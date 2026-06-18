//! LaMa — Large Mask Inpainting.
//!
//! Input: `Image2D` + `Mask2D` (1.0 = region to inpaint)
//! Output: `Image2D` (inpainted result)

use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use poc::backend::vips::VipsBackend;
use poc::color::model::ColorModel;
use poc::color::space::ColorSpace;
use poc::data::image::{Image2D, RamImageTarget};
use poc::data::mask2d::{Mask2D, RamMaskTarget};
use poc::pixel::{AlphaState, PixelLayout, Storage};
use poc::work_unit::{Lod, Region};

const RGB_U8_LAYOUT: PixelLayout = PixelLayout {
    storage: Storage::U8,
    model: ColorModel::Rgb,
    alpha: AlphaState::None,
    color_space: ColorSpace::SRGB,
};

/// Configuration for LaMa inference.
#[derive(Debug, Clone)]
pub struct LamaConfig {
    /// Internal processing resolution (default: 512).
    /// LaMa works best at 512; larger values may improve quality but require more VRAM.
    pub input_size: usize,

    /// Pad dimensions to a multiple of this (default: 8).
    /// Required by the LaMa architecture for proper convolution alignment.
    pub pad_multiple: usize,

    /// Mask binarization threshold (default: 0.5).
    /// Mask values above this become 1.0 (inpaint), below become 0.0 (keep).
    pub mask_threshold: f32,

    /// Whether the model output is already in [0, 255] range (default: true).
    /// Set to `false` if using a model variant that outputs [0, 1].
    pub output_is_255: bool,

    /// Alpha for blending inpainted region with original (default: 1.0 = full replacement).
    /// Values < 1.0 blend original and inpainted pixels in the masked region.
    pub blend_alpha: f32,
}

impl Default for LamaConfig {
    fn default() -> Self {
        Self {
            input_size: 512,
            pad_multiple: 8,
            mask_threshold: 0.5,
            output_is_255: true,
            blend_alpha: 1.0,
        }
    }
}

pub struct LamaModel {
    session: Session,
    config: LamaConfig,
}

impl LamaModel {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        Self::with_config(model_path, LamaConfig::default())
    }

    pub fn with_config(model_path: &str, config: LamaConfig) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;

        Ok(Self { session, config })
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &LamaConfig {
        &self.config
    }

    /// Inpaints masked regions of an image.
    ///
    /// The `mask` indicates which pixels to inpaint (1.0 = inpaint, 0.0 = keep).
    /// Both image and mask can be any resolution — they are resized to `input_size`
    /// internally and the result is returned at that resolution.
    pub fn inpaint(
        &mut self,
        image: &Image2D<VipsBackend>,
        mask: &Mask2D<VipsBackend>,
    ) -> Result<Image2D<VipsBackend>, poc::error::Error> {
        let sz = self.config.input_size;
        let pm = self.config.pad_multiple;

        let (w, h) = image.spec.dims();
        let img_resized = image.resize(sz as f64 / w as f64, None, Some(sz as f64 / h as f64), None)
            .convert(RGB_U8_LAYOUT);
        let iw = img_resized.width() as usize;
        let ih = img_resized.height() as usize;

        let pad_w = if iw % pm != 0 { ((iw / pm) + 1) * pm } else { iw };
        let pad_h = if ih % pm != 0 { ((ih / pm) + 1) * pm } else { ih };

        let img_bytes = img_resized.pull(
            &RamImageTarget,
            Region::full((iw as i32, ih as i32), Lod(0)),
        )?;

        let mask_values = mask.pull(
            &RamMaskTarget,
            Region::full((mask.width(), mask.height()), Lod(0)),
        )?;
        let mw = mask.width() as usize;
        let mh = mask.height() as usize;

        // Build image tensor (normalize to [0, 1])
        let mut image_arr = Array4::<f32>::zeros((1, 3, pad_h, pad_w));
        for y in 0..ih {
            for x in 0..iw {
                let idx = (y * iw + x) * 3;
                for c in 0..3 {
                    image_arr[[0, c, y, x]] = img_bytes[idx + c] as f32 / 255.0;
                }
            }
        }

        // Build mask tensor (binarize with threshold, scale to image coords)
        let thresh = self.config.mask_threshold;
        let mut mask_arr = Array4::<f32>::zeros((1, 1, pad_h, pad_w));
        for y in 0..ih {
            for x in 0..iw {
                let mx = (x as f32 * mw as f32 / iw as f32) as usize;
                let my = (y as f32 * mh as f32 / ih as f32) as usize;
                let mi = my * mw + mx;
                if mi < mask_values.len() {
                    mask_arr[[0, 0, y, x]] = if mask_values[mi] > thresh { 1.0 } else { 0.0 };
                }
            }
        }

        let image_tensor = Tensor::from_array(image_arr)
            .map_err(|e| poc::error::Error::Backend(format!("LaMa image tensor: {e:?}")))?
            .into_dyn();
        let mask_tensor = Tensor::from_array(mask_arr)
            .map_err(|e| poc::error::Error::Backend(format!("LaMa mask tensor: {e:?}")))?
            .into_dyn();

        let outputs = self.session
            .run(ort::inputs!["image" => image_tensor, "mask" => mask_tensor])
            .map_err(|e| poc::error::Error::Backend(format!("LaMa inference: {e:?}")))?;

        let out_key = outputs.keys().next().unwrap();
        let (shape, slice) = outputs[out_key]
            .try_extract_tensor::<f32>()
            .map_err(|e| poc::error::Error::Backend(format!("LaMa output: {e:?}")))?;

        let out_h = shape[2] as usize;
        let out_w = shape[3] as usize;
        let divisor = if self.config.output_is_255 { 1.0 } else { 1.0 / 255.0 };
        let blend = self.config.blend_alpha;

        let crop_h = ih.min(out_h);
        let crop_w = iw.min(out_w);
        let mut output_bytes = vec![0u8; crop_h * crop_w * 3];
        for y in 0..crop_h {
            for x in 0..crop_w {
                let dst_idx = (y * crop_w + x) * 3;
                // Read mask to decide blending
                let mx = (x as f32 * mw as f32 / crop_w as f32) as usize;
                let my = (y as f32 * mh as f32 / crop_h as f32) as usize;
                let mi = my * mw + mx;
                let in_mask = mi < mask_values.len() && mask_values[mi] > thresh;

                for c in 0..3 {
                    let inpainted = (slice[c * out_h * out_w + y * out_w + x] * divisor)
                        .clamp(0.0, 255.0);
                    if in_mask && blend < 1.0 {
                        let original = img_bytes[(y * iw + x) * 3 + c] as f32;
                        output_bytes[dst_idx + c] =
                            (original * (1.0 - blend) + inpainted * blend) as u8;
                    } else {
                        output_bytes[dst_idx + c] = inpainted as u8;
                    }
                }
            }
        }

        Ok(Image2D::<VipsBackend>::from_bytes(
            output_bytes,
            crop_w as i32,
            crop_h as i32,
            RGB_U8_LAYOUT,
        ))
    }
}
