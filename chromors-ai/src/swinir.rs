//! SwinIR — image restoration / denoising / super resolution.
//!
//! Input: `Image2D` (any size)
//! Output: `Image2D` (restored, potentially upscaled depending on model variant)

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

/// Configuration for SwinIR inference.
#[derive(Debug, Clone)]
pub struct SwinIrConfig {
    /// Swin Transformer window size (default: 8).
    /// Input dimensions are padded to a multiple of this.
    /// Common values: 8 (lightweight), 16 (large models).
    pub window_size: usize,

    /// Maximum input resolution before downscaling (default: 1024).
    /// Images larger than this are downscaled first to avoid OOM.
    /// Set to `usize::MAX` to disable.
    pub max_input_size: usize,

    /// Normalization range. If true, input is [0, 1] (default: true).
    pub normalize_0_1: bool,

    /// Whether output should be clamped to [0, 1] (default: true).
    pub clamp_output: bool,
}

impl Default for SwinIrConfig {
    fn default() -> Self {
        Self {
            window_size: 8,
            max_input_size: 1024,
            normalize_0_1: true,
            clamp_output: true,
        }
    }
}

pub struct SwinIrModel {
    session: Session,
    config: SwinIrConfig,
}

impl SwinIrModel {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        Self::with_config(model_path, SwinIrConfig::default())
    }

    pub fn with_config(model_path: &str, config: SwinIrConfig) -> ort::Result<Self> {
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
    pub fn config(&self) -> &SwinIrConfig {
        &self.config
    }

    /// Restores / denoises / upscales an image using SwinIR.
    ///
    /// The output size depends on the model variant (1× for denoising,
    /// 2×/4× for super resolution).
    pub fn restore(
        &mut self,
        image: &Image2D<VipsBackend>,
    ) -> Result<Image2D<VipsBackend>, poc::error::Error> {
        let (orig_w, orig_h) = image.spec.dims();
        let mut w = orig_w as usize;
        let mut h = orig_h as usize;

        // Downscale if needed
        let max = self.config.max_input_size;
        let preprocessed = if w > max || h > max {
            let scale = max as f64 / w.max(h) as f64;
            let img = image.resize(scale, None, Some(scale), None).convert(RGB_U8_LAYOUT);
            w = img.width() as usize;
            h = img.height() as usize;
            img
        } else {
            image.clone().convert(RGB_U8_LAYOUT)
        };

        let ws = self.config.window_size;
        let pad_w = if w % ws != 0 { ((w / ws) + 1) * ws } else { w };
        let pad_h = if h % ws != 0 { ((h / ws) + 1) * ws } else { h };

        let bytes = preprocessed.pull(
            &RamImageTarget,
            Region::full((w as i32, h as i32), Lod(0)),
        )?;

        let mut input = Array4::<f32>::zeros((1, 3, pad_h, pad_w));
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                for c in 0..3 {
                    let v = bytes[idx + c] as f32;
                    input[[0, c, y, x]] = if self.config.normalize_0_1 {
                        v / 255.0
                    } else {
                        v / 127.5 - 1.0
                    };
                }
            }
        }

        let input_tensor = Tensor::from_array(input)
            .map_err(|e| poc::error::Error::Backend(format!("SwinIR tensor: {e:?}")))?
            .into_dyn();

        let outputs = self.session
            .run(ort::inputs!["input" => input_tensor])
            .map_err(|e| poc::error::Error::Backend(format!("SwinIR inference: {e:?}")))?;

        let out_key = outputs.keys().next().unwrap();
        let (shape, slice) = outputs[out_key]
            .try_extract_tensor::<f32>()
            .map_err(|e| poc::error::Error::Backend(format!("SwinIR output: {e:?}")))?;

        let out_h = shape[2] as usize;
        let out_w = shape[3] as usize;

        // Compute scale factor from input→output
        let scale = out_h / pad_h;
        let crop_h = h * scale;
        let crop_w = w * scale;

        let mut output_bytes = vec![0u8; crop_h * crop_w * 3];
        for y in 0..crop_h {
            for x in 0..crop_w {
                let dst_idx = (y * crop_w + x) * 3;
                for c in 0..3 {
                    let mut val = slice[c * out_h * out_w + y * out_w + x];
                    if self.config.clamp_output {
                        val = val.clamp(0.0, 1.0);
                    }
                    output_bytes[dst_idx + c] = (val * 255.0).clamp(0.0, 255.0) as u8;
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
