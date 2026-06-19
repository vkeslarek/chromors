//! ViTMatte — trimap-guided alpha matting with Vision Transformers.
//!
//! Input: `Image2D<B>` + `Mask2D<B>` (trimap: 0=bg, 0.5=unknown, 1=fg)
//! Output: `Mask2D<B>` (refined alpha matte, f32, 0..1)

use ndarray::Array4;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use chromors::color::model::ColorModel;
use chromors::color::space::ColorSpace;
use chromors::data::image::{Image2D, RamImageTarget};
use chromors::data::mask2d::{Mask2D, RamMaskTarget};
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

#[derive(Debug, Clone)]
pub struct ViTMatteConfig {
    /// Pad input dimensions to a multiple of this (ViT patch requirement, default: 32).
    pub pad_multiple: usize,
    pub normalize_mean: [f32; 3],
    pub normalize_std: [f32; 3],
    /// Alpha values above this → definite foreground (1.0) in trimap (default: 0.8).
    pub trimap_fg_threshold: f32,
    /// Alpha values below this → definite background (0.0) in trimap (default: 0.2).
    pub trimap_bg_threshold: f32,
    pub alpha_clip_low: f32,
    pub alpha_clip_high: f32,
}

impl Default for ViTMatteConfig {
    fn default() -> Self {
        Self {
            pad_multiple: 32,
            normalize_mean: [0.485, 0.456, 0.406],
            normalize_std: [0.229, 0.224, 0.225],
            trimap_fg_threshold: 0.8,
            trimap_bg_threshold: 0.2,
            alpha_clip_low: 0.0,
            alpha_clip_high: 1.0,
        }
    }
}

impl ViTMatteConfig {
    /// Convert a raw alpha `Mask2D` into a trimap using the configured thresholds.
    pub fn alpha_to_trimap<B: AiBackend>(
        &self,
        alpha: &Mask2D<B>,
    ) -> Result<Mask2D<B>, chromors::error::Error> {
        let values = alpha.pull(
            &RamMaskTarget,
            Region::full((alpha.width(), alpha.height()), Lod(0)),
        )?;
        let trimap: Vec<f32> = values
            .iter()
            .map(|&v| {
                if v > self.trimap_fg_threshold {
                    1.0
                } else if v < self.trimap_bg_threshold {
                    0.0
                } else {
                    0.5
                }
            })
            .collect();
        Ok(B::mask_from_values(&trimap, alpha.width(), alpha.height()))
    }
}

pub struct ViTMatteModel {
    session: Session,
    config: ViTMatteConfig,
}

impl ViTMatteModel {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        Self::with_config(model_path, ViTMatteConfig::default())
    }

    pub fn with_config(model_path: &str, config: ViTMatteConfig) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;
        Ok(Self { session, config })
    }

    pub fn config(&self) -> &ViTMatteConfig {
        &self.config
    }

    /// Refines a trimap into a precise alpha matte.
    ///
    /// `trimap`: 0.0 = definite background, 0.5 = unknown, 1.0 = definite foreground.
    pub fn matte<B: AiBackend>(
        &mut self,
        image: &Image2D<B>,
        trimap: &Mask2D<B>,
    ) -> Result<Mask2D<B>, chromors::error::Error> {
        let (w, h) = image.spec.dims();
        let w = w as usize;
        let h = h as usize;

        let pm = self.config.pad_multiple;
        let pad_w = if w % pm != 0 { ((w / pm) + 1) * pm } else { w };
        let pad_h = if h % pm != 0 { ((h / pm) + 1) * pm } else { h };

        let img_bytes = image
            .clone()
            .convert(RGB_U8_LAYOUT)
            .pull(&RamImageTarget, Region::full((w as i32, h as i32), Lod(0)))?;
        let trimap_values = trimap.pull(
            &RamMaskTarget,
            Region::full((trimap.width(), trimap.height()), Lod(0)),
        )?;
        let tw = trimap.width() as usize;
        let th = trimap.height() as usize;

        let mean = self.config.normalize_mean;
        let std = self.config.normalize_std;

        // 4-channel input: RGB + trimap
        let mut input = Array4::<f32>::zeros((1, 4, pad_h, pad_w));
        for y in 0..h {
            for x in 0..w {
                let img_idx = (y * w + x) * 3;
                for c in 0..3 {
                    input[[0, c, y, x]] =
                        (img_bytes[img_idx + c] as f32 / 255.0 - mean[c]) / std[c];
                }
                let tx = (x as f32 * tw as f32 / w as f32) as usize;
                let ty = (y as f32 * th as f32 / h as f32) as usize;
                let ti = ty * tw + tx;
                if ti < trimap_values.len() {
                    input[[0, 3, y, x]] = trimap_values[ti];
                }
            }
        }

        let input_tensor = Tensor::from_array(input)
            .map_err(|e| chromors::error::Error::Backend(format!("ViTMatte tensor: {e:?}")))?
            .into_dyn();

        let outputs = self
            .session
            .run(ort::inputs!["pixel_values" => input_tensor])
            .map_err(|e| chromors::error::Error::Backend(format!("ViTMatte inference: {e:?}")))?;

        let out_key = outputs.keys().next().unwrap();
        let (shape, slice) = outputs[out_key]
            .try_extract_tensor::<f32>()
            .map_err(|e| chromors::error::Error::Backend(format!("ViTMatte output: {e:?}")))?;

        let out_w = shape[3] as usize;
        let lo = self.config.alpha_clip_low;
        let hi = self.config.alpha_clip_high;

        let alpha_values: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |x| slice[y * out_w + x].clamp(lo, hi)))
            .collect();

        Ok(B::mask_from_values(&alpha_values, w as i32, h as i32))
    }
}
