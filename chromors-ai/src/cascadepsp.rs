//! CascadePSP — global/local segmentation mask refinement.
//!
//! Input: `Image2D<B>` + coarse `Image2D<B>` mask
//! Output: refined `Image2D<B>` mask

use ndarray::Array4;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use chromors::color::model::ColorModel;
use chromors::color::space::ColorSpace;
use chromors::data::image::{Image2D, RamImageTarget};
use chromors::io::Target;
use chromors::pixel::{AlphaState, PixelLayout, Storage};
use chromors::work_unit::{Lod, Region};

use crate::prelude::AiBackend;

const RGB_U8_LAYOUT: PixelLayout = PixelLayout {
    storage: Storage::U8,
    model: ColorModel::Rgb,
    alpha: AlphaState::None,
    color_space: ColorSpace::SRGB,
};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Configuration for CascadePSP inference.
#[derive(Debug, Clone)]
pub struct CascadePspConfig {
    /// Global step resolution (default: 512).
    pub global_size: usize,
    /// Pad to multiple of this for convolution alignment (default: 8).
    pub pad_multiple: usize,
    /// Local sliding-window size (default: 512).
    pub local_window: usize,
    /// Skip local patches where the mask is >90% or <10% coverage (default: true).
    pub skip_uniform_patches: bool,
}

impl Default for CascadePspConfig {
    fn default() -> Self {
        Self {
            global_size: 512,
            pad_multiple: 8,
            local_window: 512,
            skip_uniform_patches: true,
        }
    }
}

pub struct CascadePspModel {
    session: Session,
    config: CascadePspConfig,
}

impl CascadePspModel {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        Self::with_config(model_path, CascadePspConfig::default())
    }

    pub fn with_config(model_path: &str, config: CascadePspConfig) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;
        Ok(Self { session, config })
    }

    pub fn config(&self) -> &CascadePspConfig {
        &self.config
    }

    /// Refines a rough segmentation mask (global step only).
    pub fn refine<B: AiBackend>(
        &mut self,
        image: &Image2D<B>,
        mask: &Image2D<B>,
    ) -> Result<Image2D<B>, chromors::error::Error> {
        let (w, h) = image.spec.dims();
        let sw = self.config.global_size;
        let sh = self.config.global_size;

        let img_small = image
            .resize(sw as f64 / w as f64, None, Some(sh as f64 / h as f64), None)
            .convert(RGB_U8_LAYOUT);
        let mask_small = mask
            .resize(
                sw as f64 / mask.width() as f64,
                None,
                Some(sh as f64 / mask.height() as f64),
                None,
            )
            .convert(RGB_U8_LAYOUT);

        let img_bytes = img_small.pull(
            &RamImageTarget,
            Region::full((sw as i32, sh as i32), Lod(0)),
        )?;
        let mask_bytes = mask_small.pull(
            &RamImageTarget,
            Region::full((sw as i32, sh as i32), Lod(0)),
        )?;

        let image_arr = hwc_u8_to_nchw_normalized(&img_bytes, sw, sh);
        let mask_arr = hwc_u8_to_mask_nchw(&mask_bytes, sw, sh);

        let out_arr = self
            .forward_padded(image_arr, mask_arr)
            .map_err(|e| chromors::error::Error::Backend(format!("CascadePSP ORT: {e:?}")))?;

        let out_bytes = nchw_f32_to_hwc_u8_gray3(&out_arr, sw, sh);
        let refined_small = B::image_from_bytes(out_bytes, sw as i32, sh as i32, RGB_U8_LAYOUT);

        Ok(refined_small.resize(w as f64 / sw as f64, None, Some(h as f64 / sh as f64), None))
    }

    /// Refines a mask with both global and local (sliding window) steps.
    pub fn refine_with_local<B: AiBackend>(
        &mut self,
        image: &Image2D<B>,
        mask: &Image2D<B>,
    ) -> Result<Image2D<B>, chromors::error::Error> {
        let global_refined = self.refine(image, mask)?;

        let (w, h) = image.spec.dims();
        let w = w as usize;
        let h = h as usize;

        let img_bytes = image
            .clone()
            .convert(RGB_U8_LAYOUT)
            .pull(&RamImageTarget, Region::full((w as i32, h as i32), Lod(0)))?;
        let mask_bytes = global_refined
            .convert(RGB_U8_LAYOUT)
            .pull(&RamImageTarget, Region::full((w as i32, h as i32), Lod(0)))?;

        let image_arr = hwc_u8_to_nchw_normalized(&img_bytes, w, h);
        let mask_arr = hwc_u8_to_mask_nchw(&mask_bytes, w, h);

        let local_out = self
            .refine_local_internal(image_arr, mask_arr)
            .map_err(|e| chromors::error::Error::Backend(format!("CascadePSP local ORT: {e:?}")))?;

        let out_bytes = nchw_f32_to_hwc_u8_gray3(&local_out, w, h);
        Ok(B::image_from_bytes(
            out_bytes,
            w as i32,
            h as i32,
            RGB_U8_LAYOUT,
        ))
    }

    fn forward_padded(
        &mut self,
        image: Array4<f32>,
        mask: Array4<f32>,
    ) -> ort::Result<Array4<f32>> {
        use ndarray::s;

        let (b, _, h, w) = image.dim();
        let pm = self.config.pad_multiple;
        let pad_h = if h % pm != 0 { ((h / pm) + 1) * pm } else { h };
        let pad_w = if w % pm != 0 { ((w / pm) + 1) * pm } else { w };

        let mut padded_image = Array4::<f32>::zeros((b, 3, pad_h, pad_w));
        let mut padded_mask = Array4::<f32>::from_elem((b, 1, pad_h, pad_w), -1.0);
        padded_image.slice_mut(s![.., .., ..h, ..w]).assign(&image);
        padded_mask.slice_mut(s![.., .., ..h, ..w]).assign(&mask);

        let outputs = self.session.run(ort::inputs![
            "image" => Tensor::from_array(padded_image)?.into_dyn(),
            "mask"  => Tensor::from_array(padded_mask)?.into_dyn(),
        ])?;

        let out_key = outputs.keys().next().unwrap();
        let (shape, slice) = outputs[out_key].try_extract_tensor::<f32>()?;
        let full = ndarray::ArrayView4::from_shape(
            (
                shape[0] as usize,
                shape[1] as usize,
                shape[2] as usize,
                shape[3] as usize,
            ),
            slice,
        )
        .unwrap();
        Ok(full.slice(s![.., .., ..h, ..w]).to_owned())
    }

    fn refine_local_internal(
        &mut self,
        image: Array4<f32>,
        mask: Array4<f32>,
    ) -> ort::Result<Array4<f32>> {
        use ndarray::s;

        let (b, _, h, w) = image.dim();
        let l = self.config.local_window;
        let stride = l / 2;
        let padding = 16;
        let step_size = stride - padding * 2;

        let mut combined = Array4::<f32>::zeros((b, 1, h, w));
        let mut weight = Array4::<f32>::zeros((b, 1, h, w));
        let mut used_starts = std::collections::HashSet::new();

        for y_idx in 0..=(h / step_size) {
            for x_idx in 0..=(w / step_size) {
                let mut sx = x_idx * step_size;
                let mut sy = y_idx * step_size;
                let mut ex = (sx + l).min(w);
                let mut ey = (sy + l).min(h);
                if ex == w && w > l {
                    sx = w - l;
                }
                if ey == h && h > l {
                    sy = h - l;
                }
                ex = ex.min(w);
                ey = ey.min(h);

                if !used_starts.insert(sy * w + sx) {
                    continue;
                }

                let mask_part = mask.slice(s![.., .., sy..ey, sx..ex]).to_owned();
                if self.config.skip_uniform_patches {
                    let pos = mask_part.iter().filter(|&&v| v > 0.0).count();
                    let mean = pos as f32 / mask_part.len() as f32;
                    if mean > 0.9 || mean < 0.1 {
                        continue;
                    }
                }

                let im_part = image.slice(s![.., .., sy..ey, sx..ex]).to_owned();
                let pred = self.forward_padded(im_part, mask_part)?;

                let csx = if sx != 0 { sx + padding } else { sx };
                let csy = if sy != 0 { sy + padding } else { sy };
                let cex = if ex != w { ex - padding } else { ex };
                let cey = if ey != h { ey - padding } else { ey };
                let psx = csx - sx;
                let psy = csy - sy;
                let pex = psx + (cex - csx);
                let pey = psy + (cey - csy);

                let pred_slice = pred.slice(s![.., .., psy..pey, psx..pex]);
                combined.slice_mut(s![.., .., csy..cey, csx..cex]) += &pred_slice;
                weight.slice_mut(s![.., .., csy..cey, csx..cex]) += 1.0;
            }
        }

        let mut out = Array4::<f32>::zeros((b, 1, h, w));
        let mask_s = mask.as_slice().unwrap();
        let comb_s = combined.as_slice().unwrap();
        let wt_s = weight.as_slice().unwrap();
        let out_s = out.as_slice_mut().unwrap();
        for i in 0..(b * h * w) {
            let seg_norm = mask_s[i] / 2.0 + 0.5;
            out_s[i] = if wt_s[i] == 0.0 {
                seg_norm
            } else {
                comb_s[i] / wt_s[i]
            };
        }
        Ok(out)
    }
}

fn hwc_u8_to_nchw_normalized(bytes: &[u8], w: usize, h: usize) -> Array4<f32> {
    let mut arr = Array4::<f32>::zeros((1, 3, h, w));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            for c in 0..3 {
                arr[[0, c, y, x]] =
                    (bytes[idx + c] as f32 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    arr
}

fn hwc_u8_to_mask_nchw(bytes: &[u8], w: usize, h: usize) -> Array4<f32> {
    let mut arr = Array4::<f32>::zeros((1, 1, h, w));
    for y in 0..h {
        for x in 0..w {
            let bin = if bytes[(y * w + x) * 3] > 127 {
                1.0f32
            } else {
                0.0
            };
            arr[[0, 0, y, x]] = (bin - 0.5) / 0.5;
        }
    }
    arr
}

fn nchw_f32_to_hwc_u8_gray3(arr: &Array4<f32>, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let v = (arr[[0, 0, y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            let idx = (y * w + x) * 3;
            out[idx] = v;
            out[idx + 1] = v;
            out[idx + 2] = v;
        }
    }
    out
}
