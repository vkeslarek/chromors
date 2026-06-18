//! MODNet — lightweight portrait matting.
//!
//! Input: `Image2D` (any size, resized internally)
//! Output: `Mask2D` (alpha matte at model resolution)

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

/// Configuration for MODNet inference.
#[derive(Debug, Clone)]
pub struct ModNetConfig {
    /// Internal resolution for model input (default: 512).
    /// Larger → more detail but slower.
    pub input_size: usize,

    /// Normalization center value (default: 127.5).
    /// Standard MODNet uses `(pixel - ref_val) / ref_val`.
    pub ref_val: f32,

    /// Alpha threshold: values below this are clamped to 0.0 (default: 0.0).
    /// Useful to clean up faint background noise.
    pub alpha_threshold: f32,
}

impl Default for ModNetConfig {
    fn default() -> Self {
        Self {
            input_size: 512,
            ref_val: 127.5,
            alpha_threshold: 0.0,
        }
    }
}

pub struct ModNetModel {
    session: Session,
    config: ModNetConfig,
}

impl ModNetModel {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        Self::with_config(model_path, ModNetConfig::default())
    }

    pub fn with_config(model_path: &str, config: ModNetConfig) -> ort::Result<Self> {
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
    pub fn config(&self) -> &ModNetConfig {
        &self.config
    }

    /// Produces an alpha matte from a portrait image.
    ///
    /// Returns a `Mask2D` where 1.0 = foreground, 0.0 = background.
    pub fn matte(
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

        let ref_val = self.config.ref_val;
        let mut input = Array4::<f32>::zeros((1, 3, sz, sz));
        for y in 0..sz {
            for x in 0..sz {
                let idx = (y * sz + x) * 3;
                for c in 0..3 {
                    input[[0, c, y, x]] = (bytes[idx + c] as f32 - ref_val) / ref_val;
                }
            }
        }

        let input_tensor = Tensor::from_array(input)
            .map_err(|e| poc::error::Error::Backend(format!("MODNet tensor: {e:?}")))?
            .into_dyn();

        let outputs = self.session
            .run(ort::inputs!["input" => input_tensor])
            .map_err(|e| poc::error::Error::Backend(format!("MODNet inference: {e:?}")))?;

        let out_key = outputs.keys().next().unwrap();
        let (shape, slice) = outputs[out_key]
            .try_extract_tensor::<f32>()
            .map_err(|e| poc::error::Error::Backend(format!("MODNet output: {e:?}")))?;

        let oh = shape[2] as usize;
        let ow = shape[3] as usize;
        let thresh = self.config.alpha_threshold;

        let mut mask_values = vec![0.0f32; oh * ow];
        for y in 0..oh {
            for x in 0..ow {
                let v = slice[y * ow + x].clamp(0.0, 1.0);
                mask_values[y * ow + x] = if v < thresh { 0.0 } else { v };
            }
        }

        Ok(Mask2D::<VipsBackend>::from_values(ow as i32, oh as i32, &mask_values))
    }
}
