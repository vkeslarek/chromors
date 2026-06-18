//! Depth Anything V2 — monocular depth estimation.
//!
//! Input: `Image2D<B>` (any size, any layout — converted internally)
//! Output: `Mask2D<B>` (depth map, 0..1 normalized, near=0 far=1 by default)

use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use chromors::color::model::ColorModel;
use chromors::color::space::ColorSpace;
use chromors::data::image::{Image2D, RamImageTarget};
use chromors::data::mask2d::Mask2D;
use chromors::pixel::{AlphaState, PixelLayout, Storage};
use chromors::work_unit::{Lod, Region};
use chromors::io::Target;

use crate::prelude::AiBackend;

const RGB_U8_LAYOUT: PixelLayout = PixelLayout {
    storage: Storage::U8,
    model: ColorModel::Rgb,
    alpha: AlphaState::None,
    color_space: ColorSpace::SRGB,
};

#[derive(Debug, Clone)]
pub struct DepthAnythingConfig {
    /// Input resolution (default: 518 = 14×37, DINOv2 patch size).
    /// Must be divisible by 14.
    pub input_size: usize,
    pub normalize_mean: [f32; 3],
    pub normalize_std: [f32; 3],
    /// Invert depth: 1.0 = near, 0.0 = far (default: false).
    pub invert: bool,
    /// Apply min-max normalization to output (default: true).
    pub normalize_output: bool,
    /// Gamma correction on output (default: 1.0 = linear).
    pub gamma: f32,
}

impl Default for DepthAnythingConfig {
    fn default() -> Self {
        Self {
            input_size: 518,
            normalize_mean: [0.485, 0.456, 0.406],
            normalize_std: [0.229, 0.224, 0.225],
            invert: false,
            normalize_output: true,
            gamma: 1.0,
        }
    }
}

pub struct DepthAnythingModel {
    session: Session,
    config: DepthAnythingConfig,
}

impl DepthAnythingModel {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        Self::with_config(model_path, DepthAnythingConfig::default())
    }

    pub fn with_config(model_path: &str, config: DepthAnythingConfig) -> ort::Result<Self> {
        assert!(config.input_size % 14 == 0,
            "input_size must be divisible by 14 (DINOv2 patch size), got {}", config.input_size);
        assert!(config.gamma > 0.0, "gamma must be positive, got {}", config.gamma);

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;

        Ok(Self { session, config })
    }

    pub fn config(&self) -> &DepthAnythingConfig { &self.config }

    /// Estimates depth from a single image.
    ///
    /// Returns a `Mask2D` where 0.0 = nearest, 1.0 = farthest (unless `invert = true`).
    pub fn estimate<B: AiBackend>(
        &mut self,
        image: &Image2D<B>,
    ) -> Result<Mask2D<B>, chromors::error::Error> {
        let (w, h) = image.spec.dims();
        let sz = self.config.input_size;

        let preprocessed = image
            .resize(sz as f64 / w as f64, None, Some(sz as f64 / h as f64), None)
            .convert(RGB_U8_LAYOUT);

        let bytes = preprocessed.pull(&RamImageTarget, Region::full((sz as i32, sz as i32), Lod(0)))?;

        let mean = self.config.normalize_mean;
        let std = self.config.normalize_std;
        let mut input = Array4::<f32>::zeros((1, 3, sz, sz));
        for y in 0..sz {
            for x in 0..sz {
                let idx = (y * sz + x) * 3;
                for c in 0..3 {
                    input[[0, c, y, x]] = (bytes[idx + c] as f32 / 255.0 - mean[c]) / std[c];
                }
            }
        }

        let input_tensor = Tensor::from_array(input)
            .map_err(|e| chromors::error::Error::Backend(format!("DepthAnything tensor: {e:?}")))?
            .into_dyn();

        let outputs = self.session
            .run(ort::inputs!["pixel_values" => input_tensor])
            .map_err(|e| chromors::error::Error::Backend(format!("DepthAnything inference: {e:?}")))?;

        let out_key = outputs.keys().next().unwrap();
        let (shape, slice) = outputs[out_key]
            .try_extract_tensor::<f32>()
            .map_err(|e| chromors::error::Error::Backend(format!("DepthAnything output: {e:?}")))?;

        // Output shape: [B, H, W] or [B, 1, H, W]
        let (oh, ow) = if shape.len() == 3 {
            (shape[1] as usize, shape[2] as usize)
        } else {
            (shape[2] as usize, shape[3] as usize)
        };

        let mut depth_values: Vec<f32> = slice[..oh * ow].to_vec();

        if self.config.normalize_output {
            let min_v = depth_values.iter().cloned().fold(f32::MAX, f32::min);
            let max_v = depth_values.iter().cloned().fold(f32::MIN, f32::max);
            let range = if (max_v - min_v).abs() > 1e-6 { max_v - min_v } else { 1.0 };
            for v in depth_values.iter_mut() { *v = (*v - min_v) / range; }
        }

        let gamma = self.config.gamma;
        for v in depth_values.iter_mut() {
            if self.config.invert { *v = 1.0 - *v; }
            if (gamma - 1.0).abs() > 1e-6 { *v = v.powf(gamma); }
            *v = v.clamp(0.0, 1.0);
        }

        Ok(B::mask_from_values(&depth_values, ow as i32, oh as i32))
    }
}
