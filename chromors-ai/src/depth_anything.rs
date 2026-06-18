//! Depth Anything V2 — monocular depth estimation.
//!
//! Input: `Image2D` (any size, any layout — converted internally)
//! Output: `Mask2D` (depth map, 0..1 normalized, near=0 far=1)

use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use poc::backend::vips::VipsBackend;
use poc::color::model::ColorModel;
use poc::color::space::ColorSpace;
use poc::data::image::{Image2D, RamImageTarget};
use poc::data::mask2d::Mask2D;
use poc::pixel::{AlphaState, PixelLayout, Storage};
use poc::work_unit::{Lod, Region};

const RGB_U8_LAYOUT: PixelLayout = PixelLayout {
    storage: Storage::U8,
    model: ColorModel::Rgb,
    alpha: AlphaState::None,
    color_space: ColorSpace::SRGB,
};

/// Configuration for Depth Anything V2 inference.
#[derive(Debug, Clone)]
pub struct DepthAnythingConfig {
    /// Input resolution for the model (default: 518).
    /// The DINOv2 backbone expects 518 = 14×37.
    /// Larger → more detail, slower. Must be divisible by 14.
    pub input_size: usize,

    /// ImageNet normalization mean (default: `[0.485, 0.456, 0.406]`).
    pub normalize_mean: [f32; 3],

    /// ImageNet normalization std (default: `[0.229, 0.224, 0.225]`).
    pub normalize_std: [f32; 3],

    /// Whether to invert the depth map (default: false).
    /// When true, 1.0 = near, 0.0 = far. When false, 0.0 = near, 1.0 = far.
    pub invert: bool,

    /// Contrast stretch: apply min-max normalization to the output (default: true).
    /// When true, the full [0, 1] range is used regardless of scene depth range.
    pub normalize_output: bool,

    /// Gamma correction applied to the depth map (default: 1.0 = linear).
    /// Values > 1.0 compress near-range, expand far-range.
    /// Values < 1.0 expand near-range, compress far-range.
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
            "input_size must be divisible by 14 (DINOv2 patch size), got {}",
            config.input_size);
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

    /// Returns the current configuration.
    pub fn config(&self) -> &DepthAnythingConfig {
        &self.config
    }

    /// Estimates depth from a single image.
    ///
    /// Returns a `Mask2D` where pixel values represent relative depth.
    /// By default: 0.0 = nearest, 1.0 = farthest.
    pub fn estimate(
        &mut self,
        image: &Image2D<VipsBackend>,
    ) -> Result<Mask2D<VipsBackend>, poc::error::Error> {
        let (w, h) = image.spec.dims();
        let sz = self.config.input_size;

        let preprocessed = image
            .resize(sz as f64 / w as f64, None, Some(sz as f64 / h as f64), None)
            .convert(RGB_U8_LAYOUT);

        let bytes = preprocessed.pull(
            &RamImageTarget,
            Region::full((sz as i32, sz as i32), Lod(0)),
        )?;

        let mean = self.config.normalize_mean;
        let std = self.config.normalize_std;
        let mut input = Array4::<f32>::zeros((1, 3, sz, sz));
        for y in 0..sz {
            for x in 0..sz {
                let idx = (y * sz + x) * 3;
                for c in 0..3 {
                    let val = bytes[idx + c] as f32 / 255.0;
                    input[[0, c, y, x]] = (val - mean[c]) / std[c];
                }
            }
        }

        let input_tensor = Tensor::from_array(input)
            .map_err(|e| poc::error::Error::Backend(format!("DepthAnything tensor: {e:?}")))?
            .into_dyn();

        let outputs = self.session
            .run(ort::inputs!["pixel_values" => input_tensor])
            .map_err(|e| poc::error::Error::Backend(format!("DepthAnything inference: {e:?}")))?;

        let out_key = outputs.keys().next().unwrap();
        let (shape, slice) = outputs[out_key]
            .try_extract_tensor::<f32>()
            .map_err(|e| poc::error::Error::Backend(format!("DepthAnything output: {e:?}")))?;

        // Output shape can be [B, H, W] or [B, 1, H, W]
        let (oh, ow) = if shape.len() == 3 {
            (shape[1] as usize, shape[2] as usize)
        } else {
            (shape[2] as usize, shape[3] as usize)
        };
        println!("  Output depth map: {}x{}", ow, oh);

        // Collect raw values and find range
        let mut depth_values = vec![0.0f32; oh * ow];
        let mut min_v = f32::MAX;
        let mut max_v = f32::MIN;
        for i in 0..(oh * ow) {
            let v = slice[i];
            depth_values[i] = v;
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }

        // Normalize + apply config transforms
        let range = if (max_v - min_v).abs() > 1e-6 { max_v - min_v } else { 1.0 };
        let gamma = self.config.gamma;
        for v in depth_values.iter_mut() {
            let mut d = if self.config.normalize_output {
                (*v - min_v) / range
            } else {
                *v
            };
            if self.config.invert {
                d = 1.0 - d;
            }
            if (gamma - 1.0).abs() > 1e-6 {
                d = d.powf(gamma);
            }
            *v = d.clamp(0.0, 1.0);
        }

        Ok(Mask2D::<VipsBackend>::from_values(ow as i32, oh as i32, &depth_values))
    }
}
