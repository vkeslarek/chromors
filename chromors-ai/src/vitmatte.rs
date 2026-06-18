//! ViTMatte — trimap-guided alpha matting with Vision Transformers.
//!
//! Input: `Image2D` + `Mask2D` (trimap: 0=bg, 0.5=unknown, 1=fg)
//! Output: `Mask2D` (refined alpha matte, f32, 0..1)

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

/// Configuration for ViTMatte inference.
#[derive(Debug, Clone)]
pub struct ViTMatteConfig {
    /// Pad input dimensions to a multiple of this (ViT patch requirement, default: 32).
    pub pad_multiple: usize,

    /// ImageNet normalization mean (default: `[0.485, 0.456, 0.406]`).
    pub normalize_mean: [f32; 3],

    /// ImageNet normalization std (default: `[0.229, 0.224, 0.225]`).
    pub normalize_std: [f32; 3],

    /// Foreground threshold for trimap generation from a raw alpha mask.
    /// Values above this → definite foreground (1.0). Default: 0.8.
    pub trimap_fg_threshold: f32,

    /// Background threshold for trimap generation from a raw alpha mask.
    /// Values below this → definite background (0.0). Default: 0.2.
    pub trimap_bg_threshold: f32,

    /// Minimum alpha in the output — values below are clamped to 0 (default: 0.0).
    pub alpha_clip_low: f32,

    /// Maximum alpha in the output — values above are clamped to 1 (default: 1.0).
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
    /// Convert a raw alpha `Mask2D` into a trimap `Mask2D` using the configured thresholds.
    ///
    /// - alpha > `trimap_fg_threshold` → 1.0 (foreground)
    /// - alpha < `trimap_bg_threshold` → 0.0 (background)
    /// - else → 0.5 (unknown — model estimates here)
    pub fn alpha_to_trimap(
        &self,
        alpha: &Mask2D<VipsBackend>,
    ) -> Mask2D<VipsBackend> {
        let values = alpha.pull(
            &RamMaskTarget,
            Region::full((alpha.width(), alpha.height()), Lod(0)),
        ).unwrap();

        let trimap_values: Vec<f32> = values.iter().map(|&v| {
            if v > self.trimap_fg_threshold {
                1.0
            } else if v < self.trimap_bg_threshold {
                0.0
            } else {
                0.5
            }
        }).collect();

        Mask2D::<VipsBackend>::from_values(alpha.width(), alpha.height(), &trimap_values)
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

    /// Returns the current configuration.
    pub fn config(&self) -> &ViTMatteConfig {
        &self.config
    }

    /// Refines a trimap into a precise alpha matte using ViTMatte.
    ///
    /// The `trimap` should be a `Mask2D` where:
    /// - 0.0 = definite background
    /// - 0.5 = unknown (model estimates alpha here)
    /// - 1.0 = definite foreground
    ///
    /// Trimap can be any resolution — it is automatically scaled to match the image.
    pub fn matte(
        &mut self,
        image: &Image2D<VipsBackend>,
        trimap: &Mask2D<VipsBackend>,
    ) -> Result<Mask2D<VipsBackend>, poc::error::Error> {
        let (w, h) = image.spec.dims();
        let w = w as usize;
        let h = h as usize;

        let pm = self.config.pad_multiple;
        let pad_w = if w % pm != 0 { ((w / pm) + 1) * pm } else { w };
        let pad_h = if h % pm != 0 { ((h / pm) + 1) * pm } else { h };

        let rgb = image.clone().convert(RGB_U8_LAYOUT);
        let img_bytes = rgb.pull(
            &RamImageTarget,
            Region::full((w as i32, h as i32), Lod(0)),
        )?;

        let trimap_values = trimap.pull(
            &RamMaskTarget,
            Region::full((trimap.width(), trimap.height()), Lod(0)),
        )?;

        let tw = trimap.width() as usize;
        let th = trimap.height() as usize;
        let mean = self.config.normalize_mean;
        let std = self.config.normalize_std;

        let mut input = Array4::<f32>::zeros((1, 4, pad_h, pad_w));
        for y in 0..h {
            for x in 0..w {
                let img_idx = (y * w + x) * 3;
                for c in 0..3 {
                    let val = img_bytes[img_idx + c] as f32 / 255.0;
                    input[[0, c, y, x]] = (val - mean[c]) / std[c];
                }
                let tx = (x as f32 * tw as f32 / w as f32) as usize;
                let ty = (y as f32 * th as f32 / h as f32) as usize;
                let trimap_idx = ty * tw + tx;
                if trimap_idx < trimap_values.len() {
                    input[[0, 3, y, x]] = trimap_values[trimap_idx];
                }
            }
        }

        let input_tensor = Tensor::from_array(input)
            .map_err(|e| poc::error::Error::Backend(format!("ViTMatte tensor: {e:?}")))?
            .into_dyn();

        let outputs = self.session
            .run(ort::inputs!["pixel_values" => input_tensor])
            .map_err(|e| poc::error::Error::Backend(format!("ViTMatte inference: {e:?}")))?;

        let out_key = outputs.keys().next().unwrap();
        let (shape, slice) = outputs[out_key]
            .try_extract_tensor::<f32>()
            .map_err(|e| poc::error::Error::Backend(format!("ViTMatte output: {e:?}")))?;

        let _out_h = shape[2] as usize;
        let out_w = shape[3] as usize;
        let lo = self.config.alpha_clip_low;
        let hi = self.config.alpha_clip_high;

        let mut alpha_values = vec![0.0f32; h * w];
        for y in 0..h {
            for x in 0..w {
                let v = slice[y * out_w + x].clamp(lo, hi);
                alpha_values[y * w + x] = v;
            }
        }

        Ok(Mask2D::<VipsBackend>::from_values(w as i32, h as i32, &alpha_values))
    }
}
