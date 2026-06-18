use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use poc::backend::vips::VipsBackend;
use poc::color::model::ColorModel;
use poc::color::space::ColorSpace;
use poc::data::image::{Image2D, RamImageTarget};
use poc::pixel::{AlphaState, PixelLayout, Storage};
use poc::work_unit::{Lod, Region};

const RGB_U8_LAYOUT: PixelLayout = PixelLayout {
    storage: Storage::U8,
    model: ColorModel::Rgb,
    alpha: AlphaState::None,
    color_space: ColorSpace::SRGB,
};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct CascadePspModel {
    session: Session,
}

impl CascadePspModel {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;

        Ok(Self { session })
    }

    /// Refines a rough segmentation mask using the CascadePSP global step.
    ///
    /// Both `image` and `mask` are `Image2D<VipsBackend>` — this method handles
    /// all preprocessing (resize to 512², normalize) and postprocessing
    /// (upscale back to original resolution) internally.
    ///
    /// Returns a refined grayscale mask at the original image resolution.
    pub fn refine(
        &mut self,
        image: &Image2D<VipsBackend>,
        mask: &Image2D<VipsBackend>,
    ) -> Result<Image2D<VipsBackend>, poc::error::Error> {
        let (w, h) = image.spec.dims();

        // Preprocess: resize to 512x512, convert to RGB U8
        let img_small = image
            .resize(512.0 / w as f64, None, Some(512.0 / h as f64), None)
            .convert(RGB_U8_LAYOUT);
        let mask_small = mask
            .resize(512.0 / mask.width() as f64, None, Some(512.0 / mask.height() as f64), None)
            .convert(RGB_U8_LAYOUT);

        let sw = 512usize;
        let sh = 512usize;

        // Extract bytes via the safe RamImageTarget
        let img_bytes = img_small.pull(
            &RamImageTarget,
            Region::full((sw as i32, sh as i32), Lod(0)),
        )?;
        let mask_bytes = mask_small.pull(
            &RamImageTarget,
            Region::full((sw as i32, sh as i32), Lod(0)),
        )?;

        // Convert to NCHW f32 tensors (private — never exposed)
        let image_arr = hwc_u8_to_nchw_normalized(&img_bytes, sw, sh);
        let mask_arr = hwc_u8_to_mask_nchw(&mask_bytes, sw, sh);

        // Run inference
        let out_arr = self
            .forward_padded(image_arr, mask_arr)
            .map_err(|e| poc::error::Error::Backend(format!("CascadePSP ORT: {e:?}")))?;

        // Convert back to Image2D via from_bytes
        let out_bytes = nchw_f32_to_hwc_u8_gray3(&out_arr, sw, sh);
        let refined_small = Image2D::<VipsBackend>::from_bytes(
            out_bytes,
            sw as i32,
            sh as i32,
            RGB_U8_LAYOUT,
        );

        // Upscale back to original resolution
        let final_scale_x = w as f64 / sw as f64;
        let final_scale_y = h as f64 / sh as f64;
        Ok(refined_small.resize(final_scale_x, None, Some(final_scale_y), None))
    }

    /// Refines a mask with both global and local (sliding window) steps.
    ///
    /// The local step operates at the original image resolution for
    /// pixel-accurate edge refinement.
    pub fn refine_with_local(
        &mut self,
        image: &Image2D<VipsBackend>,
        mask: &Image2D<VipsBackend>,
    ) -> Result<Image2D<VipsBackend>, poc::error::Error> {
        // Global step first
        let global_refined = self.refine(image, mask)?;

        let (w, h) = image.spec.dims();
        let w = w as usize;
        let h = h as usize;

        // Extract full-res bytes for local step
        let img_bytes = image.clone().convert(RGB_U8_LAYOUT).pull(
            &RamImageTarget,
            Region::full((w as i32, h as i32), Lod(0)),
        )?;
        let mask_bytes = global_refined.convert(RGB_U8_LAYOUT).pull(
            &RamImageTarget,
            Region::full((w as i32, h as i32), Lod(0)),
        )?;

        let image_arr = hwc_u8_to_nchw_normalized(&img_bytes, w, h);
        let mask_arr = hwc_u8_to_mask_nchw(&mask_bytes, w, h);

        let local_out = self
            .refine_local_internal(image_arr, mask_arr)
            .map_err(|e| poc::error::Error::Backend(format!("CascadePSP local ORT: {e:?}")))?;

        let out_bytes = nchw_f32_to_hwc_u8_gray3(&local_out, w, h);
        Ok(Image2D::<VipsBackend>::from_bytes(
            out_bytes,
            w as i32,
            h as i32,
            RGB_U8_LAYOUT,
        ))
    }

    // ── Private inference methods ─────────────────────────────────────────

    /// Raw forward pass ensuring padding to multiple of 8
    fn forward_padded(&mut self, image: Array4<f32>, mask: Array4<f32>) -> ort::Result<Array4<f32>> {
        use ndarray::s;

        let (b, _, h, w) = image.dim();
        let pad_h = if h % 8 != 0 { ((h / 8) + 1) * 8 } else { h };
        let pad_w = if w % 8 != 0 { ((w / 8) + 1) * 8 } else { w };

        let mut padded_image = Array4::<f32>::zeros((b, 3, pad_h, pad_w));
        let mut padded_mask = Array4::<f32>::from_elem((b, 1, pad_h, pad_w), -1.0);

        padded_image.slice_mut(s![.., .., ..h, ..w]).assign(&image);
        padded_mask.slice_mut(s![.., .., ..h, ..w]).assign(&mask);

        let image_tensor = Tensor::from_array(padded_image)?.into_dyn();
        let mask_tensor = Tensor::from_array(padded_mask)?.into_dyn();

        let inputs = ort::inputs![
            "image" => image_tensor,
            "mask" => mask_tensor,
        ];

        let outputs = self.session.run(inputs)?;
        let out_key = outputs.keys().next().unwrap();

        let (shape, slice) = outputs[out_key].try_extract_tensor::<f32>()?;
        let output_array = ndarray::ArrayView4::from_shape(
            (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize),
            slice,
        )
        .unwrap();

        let cropped = output_array.slice(s![.., .., ..h, ..w]).to_owned();
        Ok(cropped)
    }

    /// Local step refinement (sliding window) at full resolution
    fn refine_local_internal(
        &mut self,
        image: Array4<f32>,
        mask: Array4<f32>,
    ) -> ort::Result<Array4<f32>> {
        use ndarray::s;

        let (b, _, h, w) = image.dim();
        let l = 512;
        let stride = l / 2;
        let padding = 16;
        let step_size = stride - padding * 2;

        let mut combined = Array4::<f32>::zeros((b, 1, h, w));
        let mut weight = Array4::<f32>::zeros((b, 1, h, w));

        let mut used_starts = std::collections::HashSet::new();

        for y_idx in 0..=(h / step_size) {
            for x_idx in 0..=(w / step_size) {
                let mut start_x = x_idx * step_size;
                let mut start_y = y_idx * step_size;
                let mut end_x = start_x + l;
                let mut end_y = start_y + l;

                if end_y > h {
                    end_y = h;
                    start_y = if h > l { h - l } else { 0 };
                }
                if end_x > w {
                    end_x = w;
                    start_x = if w > l { w - l } else { 0 };
                }

                end_x = end_x.min(w);
                end_y = end_y.min(h);

                let start_idx = start_y * w + start_x;
                if !used_starts.insert(start_idx) {
                    continue;
                }

                let im_part = image
                    .slice(s![.., .., start_y..end_y, start_x..end_x])
                    .to_owned();
                let mask_part = mask
                    .slice(s![.., .., start_y..end_y, start_x..end_x])
                    .to_owned();

                // Skip uninteresting regions
                let num_positive = mask_part.iter().filter(|&&v| v > 0.0).count();
                let mean = num_positive as f32 / mask_part.len() as f32;
                if mean > 0.9 || mean < 0.1 {
                    continue;
                }

                let grid_pred = self.forward_padded(im_part, mask_part)?;

                let mut pred_sx = 0;
                let mut pred_sy = 0;
                let mut pred_ex = end_x - start_x;
                let mut pred_ey = end_y - start_y;

                let mut comb_sx = start_x;
                let mut comb_sy = start_y;
                let mut comb_ex = end_x;
                let mut comb_ey = end_y;

                if start_x != 0 {
                    comb_sx += padding;
                    pred_sx += padding;
                }
                if start_y != 0 {
                    comb_sy += padding;
                    pred_sy += padding;
                }
                if end_x != w {
                    comb_ex -= padding;
                    pred_ex -= padding;
                }
                if end_y != h {
                    comb_ey -= padding;
                    pred_ey -= padding;
                }

                let pred_slice = grid_pred.slice(s![.., .., pred_sy..pred_ey, pred_sx..pred_ex]);
                let mut dest_slice =
                    combined.slice_mut(s![.., .., comb_sy..comb_ey, comb_sx..comb_ex]);
                dest_slice += &pred_slice;

                let mut weight_slice =
                    weight.slice_mut(s![.., .., comb_sy..comb_ey, comb_sx..comb_ex]);
                weight_slice += 1.0;
            }
        }

        let mut out = Array4::<f32>::zeros((b, 1, h, w));
        for i in 0..(b * h * w) {
            let w_val = weight.as_slice().unwrap()[i];
            let c_val = combined.as_slice().unwrap()[i];
            let m_val = mask.as_slice().unwrap()[i];

            let seg_norm = m_val / 2.0 + 0.5;
            out.as_slice_mut().unwrap()[i] = if w_val == 0.0 {
                seg_norm
            } else {
                c_val / w_val
            };
        }

        Ok(out)
    }
}

// ── Private tensor conversion helpers ────────────────────────────────────────

/// HWC U8 RGB bytes → NCHW f32, ImageNet-normalized.
fn hwc_u8_to_nchw_normalized(bytes: &[u8], w: usize, h: usize) -> Array4<f32> {
    let mut arr = Array4::<f32>::zeros((1, 3, h, w));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            for c in 0..3 {
                let val = bytes[idx + c] as f32 / 255.0;
                arr[[0, c, y, x]] = (val - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    arr
}

/// HWC U8 RGB bytes → NCHW f32, single-channel mask in [-1, 1].
fn hwc_u8_to_mask_nchw(bytes: &[u8], w: usize, h: usize) -> Array4<f32> {
    let mut arr = Array4::<f32>::zeros((1, 1, h, w));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let bin = if bytes[idx] > 127 { 1.0f32 } else { 0.0 };
            arr[[0, 0, y, x]] = (bin - 0.5) / 0.5; // → -1..1
        }
    }
    arr
}

/// NCHW f32 single-channel [0,1] → HWC U8 grayscale replicated to 3 channels.
fn nchw_f32_to_hwc_u8_gray3(arr: &Array4<f32>, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    for y in 0..h {
        for x in 0..w {
            let val = arr[[0, 0, y, x]];
            let v = (val * 255.0).clamp(0.0, 255.0) as u8;
            let idx = (y * w + x) * 3;
            out[idx] = v;
            out[idx + 1] = v;
            out[idx + 2] = v;
        }
    }
    out
}
