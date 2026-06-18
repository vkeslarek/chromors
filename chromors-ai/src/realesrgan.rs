//! Real-ESRGAN — 4× super resolution.
//!
//! Input: `Image2D` (any size, tiled internally with overlap)
//! Output: `Image2D` (4× upscaled)

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

/// Configuration for Real-ESRGAN inference.
#[derive(Debug, Clone)]
pub struct RealEsrganConfig {
    /// Tile size in input pixels (default: 64).
    /// Must match the model's expected input resolution.
    pub tile_size: usize,

    /// Upscale factor (default: 4). Depends on the model variant.
    pub scale: usize,

    /// Overlap in input pixels between adjacent tiles (default: 8).
    /// Larger overlap → smoother seams but slower.
    /// Set to 0 for no overlap (fastest, but visible seams).
    pub tile_pad: usize,

    /// Input normalization range. If true, normalizes to [0, 1] (default: true).
    /// Some model variants expect [-1, 1].
    pub normalize_0_1: bool,

    /// Output clamping. Clamps output values to [0, 1] before scaling to u8 (default: true).
    pub clamp_output: bool,
}

impl Default for RealEsrganConfig {
    fn default() -> Self {
        Self {
            tile_size: 64,
            scale: 4,
            tile_pad: 8,
            normalize_0_1: true,
            clamp_output: true,
        }
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

    /// Returns the current configuration.
    pub fn config(&self) -> &RealEsrganConfig {
        &self.config
    }

    /// Upscales an image by `scale`× using Real-ESRGAN.
    ///
    /// Large images are processed in overlapping tiles with linear blending
    /// to prevent visible seams.
    pub fn upscale(
        &mut self,
        image: &Image2D<VipsBackend>,
    ) -> Result<Image2D<VipsBackend>, poc::error::Error> {
        let (w, h) = image.spec.dims();
        let w = w as usize;
        let h = h as usize;
        let tile_size = self.config.tile_size;
        let scale = self.config.scale;
        let tile_pad = self.config.tile_pad;

        let rgb = image.clone().convert(RGB_U8_LAYOUT);
        let bytes = rgb.pull(
            &RamImageTarget,
            Region::full((w as i32, h as i32), Lod(0)),
        )?;

        let out_w = w * scale;
        let out_h = h * scale;
        let mut accum = vec![0.0f32; out_h * out_w * 3];
        let mut weight = vec![0.0f32; out_h * out_w];

        let step = if tile_pad > 0 { tile_size - 2 * tile_pad } else { tile_size };

        let mut y = 0usize;
        while y < h {
            let mut x = 0usize;
            while x < w {
                let tx0 = x.saturating_sub(tile_pad);
                let ty0 = y.saturating_sub(tile_pad);
                let tx1 = (x + step + tile_pad).min(w);
                let ty1 = (y + step + tile_pad).min(h);
                let tw = tx1 - tx0;
                let th = ty1 - ty0;

                let pad_w = tile_size.max(tw);
                let pad_h = tile_size.max(th);
                let mut tile = Array4::<f32>::zeros((1, 3, pad_h, pad_w));
                for py in 0..pad_h {
                    for px in 0..pad_w {
                        let sy = ty0 + py.min(th - 1);
                        let sx = tx0 + px.min(tw - 1);
                        let src_idx = (sy * w + sx) * 3;
                        for c in 0..3 {
                            let v = bytes[src_idx + c] as f32;
                            tile[[0, c, py, px]] = if self.config.normalize_0_1 {
                                v / 255.0
                            } else {
                                v / 127.5 - 1.0
                            };
                        }
                    }
                }

                let tile_tensor = Tensor::from_array(tile)
                    .map_err(|e| poc::error::Error::Backend(format!("ESRGAN tensor: {e:?}")))?
                    .into_dyn();

                let outputs = self.session
                    .run(ort::inputs!["input.1" => tile_tensor])
                    .map_err(|e| poc::error::Error::Backend(format!("ESRGAN inference: {e:?}")))?;

                let out_key = outputs.keys().next().unwrap();
                let (shape, slice) = outputs[out_key]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| poc::error::Error::Backend(format!("ESRGAN output: {e:?}")))?;

                let oth = shape[2] as usize;
                let otw = shape[3] as usize;

                let ox0 = tx0 * scale;
                let oy0 = ty0 * scale;
                let valid_h = (tw * scale).min(otw);
                let valid_w = (th * scale).min(oth);
                let pad_scaled = tile_pad * scale;

                for py in 0..valid_w.min(out_h - oy0) {
                    for px in 0..valid_h.min(out_w - ox0) {
                        let wx = if pad_scaled > 0 {
                            (px as f32 / pad_scaled as f32).min(1.0)
                                .min((valid_h - 1 - px) as f32 / pad_scaled as f32)
                                .min(1.0)
                        } else {
                            1.0
                        };
                        let wy = if pad_scaled > 0 {
                            (py as f32 / pad_scaled as f32).min(1.0)
                                .min((valid_w - 1 - py) as f32 / pad_scaled as f32)
                                .min(1.0)
                        } else {
                            1.0
                        };
                        let w_blend = wx * wy;

                        let dst_idx = ((oy0 + py) * out_w + (ox0 + px)) * 3;
                        let wt_idx = (oy0 + py) * out_w + (ox0 + px);

                        for c in 0..3 {
                            let val = slice[c * oth * otw + py * otw + px];
                            accum[dst_idx + c] += val * w_blend;
                        }
                        weight[wt_idx] += w_blend;
                    }
                }

                x += step;
            }
            y += step;
        }

        let mut output_bytes = vec![0u8; out_h * out_w * 3];
        for i in 0..(out_h * out_w) {
            let w = if weight[i] > 0.0 { weight[i] } else { 1.0 };
            for c in 0..3 {
                let mut val = accum[i * 3 + c] / w;
                if self.config.clamp_output {
                    val = val.clamp(0.0, 1.0);
                }
                output_bytes[i * 3 + c] = (val * 255.0).clamp(0.0, 255.0) as u8;
            }
        }

        Ok(Image2D::<VipsBackend>::from_bytes(
            output_bytes,
            out_w as i32,
            out_h as i32,
            RGB_U8_LAYOUT,
        ))
    }
}
