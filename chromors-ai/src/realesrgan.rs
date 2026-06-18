//! Real-ESRGAN — super resolution.
//!
//! Input: `Image2D<B>` (any size, tiled internally with overlap)
//! Output: `Image2D<B>` (`scale`× upscaled)

use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

use chromors::color::model::ColorModel;
use chromors::color::space::ColorSpace;
use chromors::data::image::{Image2D, RamImageTarget};
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
pub struct RealEsrganConfig {
    /// Tile size in input pixels (default: 64).
    pub tile_size: usize,
    /// Upscale factor (default: 4). Depends on the model variant.
    pub scale: usize,
    /// Overlap in input pixels between adjacent tiles (default: 8).
    pub tile_pad: usize,
    /// Normalize to [0, 1] (default: true). Set false for [-1, 1] models.
    pub normalize_0_1: bool,
    /// Clamp output to [0, 1] before converting to u8 (default: true).
    pub clamp_output: bool,
}

impl Default for RealEsrganConfig {
    fn default() -> Self {
        Self { tile_size: 64, scale: 4, tile_pad: 8, normalize_0_1: true, clamp_output: true }
    }
}

pub struct RealEsrganModel {
    session: Session,
    config: RealEsrganConfig,
}

impl RealEsrganModel {
    pub fn new(model_path: &str) -> ort::Result<Self> {
        Self::with_config(model_path, RealEsrganConfig::default())
    }

    pub fn with_config(model_path: &str, config: RealEsrganConfig) -> ort::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])?
            .commit_from_file(model_path)?;
        Ok(Self { session, config })
    }

    pub fn config(&self) -> &RealEsrganConfig { &self.config }

    /// Upscales an image by `config.scale`×.
    ///
    /// Large images are processed in overlapping tiles with linear blending.
    pub fn upscale<B: AiBackend>(
        &mut self,
        image: &Image2D<B>,
    ) -> Result<Image2D<B>, chromors::error::Error> {
        let (iw, ih) = image.spec.dims();
        let iw = iw as usize;
        let ih = ih as usize;
        let tile_size = self.config.tile_size;
        let scale = self.config.scale;
        let tile_pad = self.config.tile_pad;

        let rgb = image.clone().convert(RGB_U8_LAYOUT);
        let bytes = rgb.pull(&RamImageTarget, Region::full((iw as i32, ih as i32), Lod(0)))?;

        let out_w = iw * scale;
        let out_h = ih * scale;
        let mut accum = vec![0.0f32; out_h * out_w * 3];
        let mut weight = vec![0.0f32; out_h * out_w];

        let step = if tile_pad > 0 { tile_size - 2 * tile_pad } else { tile_size };

        let mut y = 0usize;
        while y < ih {
            let mut x = 0usize;
            while x < iw {
                let tx0 = x.saturating_sub(tile_pad);
                let ty0 = y.saturating_sub(tile_pad);
                let tx1 = (x + step + tile_pad).min(iw);
                let ty1 = (y + step + tile_pad).min(ih);
                let tw = tx1 - tx0;
                let th = ty1 - ty0;

                let pad_w = tile_size.max(tw);
                let pad_h = tile_size.max(th);
                let mut tile = Array4::<f32>::zeros((1, 3, pad_h, pad_w));
                for py in 0..pad_h {
                    for px in 0..pad_w {
                        let sy = ty0 + py.min(th - 1);
                        let sx = tx0 + px.min(tw - 1);
                        let src_idx = (sy * iw + sx) * 3;
                        for c in 0..3 {
                            let v = bytes[src_idx + c] as f32;
                            tile[[0, c, py, px]] = if self.config.normalize_0_1 { v / 255.0 } else { v / 127.5 - 1.0 };
                        }
                    }
                }

                let tile_tensor = Tensor::from_array(tile)
                    .map_err(|e| chromors::error::Error::Backend(format!("ESRGAN tensor: {e:?}")))?
                    .into_dyn();

                let outputs = self.session
                    .run(ort::inputs!["input.1" => tile_tensor])
                    .map_err(|e| chromors::error::Error::Backend(format!("ESRGAN inference: {e:?}")))?;

                let out_key = outputs.keys().next().unwrap();
                let (shape, slice) = outputs[out_key]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| chromors::error::Error::Backend(format!("ESRGAN output: {e:?}")))?;

                let oth = shape[2] as usize;
                let otw = shape[3] as usize;

                let ox0 = tx0 * scale;
                let oy0 = ty0 * scale;
                // valid extents in output: x-dim from tw, y-dim from th
                let valid_x = (tw * scale).min(otw);
                let valid_y = (th * scale).min(oth);
                let pad_scaled = tile_pad * scale;

                for py in 0..valid_y.min(out_h - oy0) {
                    for px in 0..valid_x.min(out_w - ox0) {
                        let wx = if pad_scaled > 0 {
                            (px as f32 / pad_scaled as f32).min(1.0)
                                .min((valid_x - 1 - px) as f32 / pad_scaled as f32)
                        } else { 1.0 };
                        let wy = if pad_scaled > 0 {
                            (py as f32 / pad_scaled as f32).min(1.0)
                                .min((valid_y - 1 - py) as f32 / pad_scaled as f32)
                        } else { 1.0 };
                        let w_blend = wx * wy;

                        let dst = ((oy0 + py) * out_w + (ox0 + px)) * 3;
                        let wt  = (oy0 + py) * out_w + (ox0 + px);
                        for c in 0..3 {
                            accum[dst + c] += slice[c * oth * otw + py * otw + px] * w_blend;
                        }
                        weight[wt] += w_blend;
                    }
                }

                x += step;
            }
            y += step;
        }

        let mut output_bytes = vec![0u8; out_h * out_w * 3];
        for i in 0..(out_h * out_w) {
            let wt = if weight[i] > 0.0 { weight[i] } else { 1.0 };
            for c in 0..3 {
                let mut val = accum[i * 3 + c] / wt;
                if self.config.clamp_output { val = val.clamp(0.0, 1.0); }
                output_bytes[i * 3 + c] = (val * 255.0) as u8;
            }
        }

        Ok(B::image_from_bytes(output_bytes, out_w as i32, out_h as i32, RGB_U8_LAYOUT))
    }
}
