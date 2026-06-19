//! SAM3 — Segment Anything Model 3 (Meta, 2024).
//!
//! Two-stage inference: encode image once, then decode any number of prompts.
//! Supports bounding-box prompts; point prompts can be added similarly.

use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView4};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;

use chromors::color::model::ColorModel;
use chromors::color::space::ColorSpace;
use chromors::data::image::{Image2D, RamImageTarget};
use chromors::data::mask2d::Mask2D;
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
pub struct Sam3Config {
    /// Resolution the encoder expects (default: 1024).
    pub encoder_size: usize,
    /// IoU threshold below which a mask is considered low-confidence (default: 0.0 = accept all).
    pub min_iou_threshold: f32,
    /// Number of language feature tokens (default: 32, must match model).
    pub language_token_count: usize,
    /// Language feature embed dim (default: 256, must match model).
    pub language_embed_dim: usize,
}

impl Default for Sam3Config {
    fn default() -> Self {
        Self {
            encoder_size: 1024,
            min_iou_threshold: 0.0,
            language_token_count: 32,
            language_embed_dim: 256,
        }
    }
}

pub struct Sam3ImageEmbeddings {
    pub vision_pos_enc_2: Array4<f32>,
    pub backbone_fpn_0: Array4<f32>,
    pub backbone_fpn_1: Array4<f32>,
    pub backbone_fpn_2: Array4<f32>,
}

pub struct Sam3Model {
    encoder: Session,
    decoder: Session,
    config: Sam3Config,
}

impl Sam3Model {
    pub fn new(encoder_path: &str, decoder_path: &str) -> ort::Result<Self> {
        Self::with_config(encoder_path, decoder_path, Sam3Config::default())
    }

    pub fn with_config(
        encoder_path: &str,
        decoder_path: &str,
        config: Sam3Config,
    ) -> ort::Result<Self> {
        let build = || {
            Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                    ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ])
        };
        let encoder = build()?.commit_from_file(encoder_path)?;
        let decoder = build()?.commit_from_file(decoder_path)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub fn config(&self) -> &Sam3Config {
        &self.config
    }

    /// Encodes an image into SAM3 embeddings (reuse for multiple decode calls).
    ///
    /// SAM3 expects raw HWC U8 — the encoder handles normalization internally.
    pub fn encode<B: AiBackend>(
        &mut self,
        image: &Image2D<B>,
    ) -> Result<Sam3ImageEmbeddings, chromors::error::Error> {
        let (w, h) = image.spec.dims();
        let sz = self.config.encoder_size;

        let preprocessed = image
            .resize(sz as f64 / w as f64, None, Some(sz as f64 / h as f64), None)
            .convert(RGB_U8_LAYOUT);

        let bytes = preprocessed.pull(
            &RamImageTarget,
            Region::full((sz as i32, sz as i32), Lod(0)),
        )?;

        let mut image_arr = Array3::<u8>::zeros((sz, sz, 3));
        for y in 0..sz {
            for x in 0..sz {
                let idx = (y * sz + x) * 3;
                image_arr[[y, x, 0]] = bytes[idx];
                image_arr[[y, x, 1]] = bytes[idx + 1];
                image_arr[[y, x, 2]] = bytes[idx + 2];
            }
        }

        self.encode_internal(image_arr)
            .map_err(|e| chromors::error::Error::Backend(format!("SAM3 encode ORT: {e:?}")))
    }

    /// Segments an image region defined by `box_coords` = [x0, y0, x1, y1] in original pixels.
    ///
    /// Returns the best-scoring mask as `Mask2D<B>` (1.0 = foreground) and IoU scores.
    /// Returns `Err` if all masks score below `config.min_iou_threshold`.
    pub fn segment_box<B: AiBackend>(
        &mut self,
        embeddings: &Sam3ImageEmbeddings,
        box_coords: [f32; 4],
        original_size: (i32, i32),
    ) -> Result<(Mask2D<B>, Vec<f32>), chromors::error::Error> {
        let (orig_w, orig_h) = original_size;

        let (masks, ious) = self
            .decode_internal(embeddings, box_coords, orig_w as i64, orig_h as i64)
            .map_err(|e| chromors::error::Error::Backend(format!("SAM3 decode ORT: {e:?}")))?;

        let num_masks = ious.dim().1;
        let mut best_idx = 0;
        let mut best_iou = f32::NEG_INFINITY;
        let mut iou_scores = Vec::with_capacity(num_masks);
        for i in 0..num_masks {
            let score = ious[[0, i]];
            iou_scores.push(score);
            if score > best_iou {
                best_iou = score;
                best_idx = i;
            }
        }

        if best_iou < self.config.min_iou_threshold {
            return Err(chromors::error::Error::Backend(format!(
                "SAM3: best IoU {best_iou:.3} below threshold {}",
                self.config.min_iou_threshold
            )));
        }

        let mask_h = masks.dim().2;
        let mask_w = masks.dim().3;
        let values: Vec<f32> = (0..mask_h)
            .flat_map(|y| (0..mask_w).map(move |x| masks[[0, best_idx, y, x]]))
            .collect();

        Ok((
            B::mask_from_values(&values, mask_w as i32, mask_h as i32),
            iou_scores,
        ))
    }

    // ── Private inference ────────────────────────────────────────────────────

    fn encode_internal(&mut self, image: Array3<u8>) -> ort::Result<Sam3ImageEmbeddings> {
        let image_tensor = Tensor::from_array(image)?.into_dyn();
        let outputs = self.encoder.run(ort::inputs!["image" => image_tensor])?;

        let extract = |name: &str| -> ort::Result<Array4<f32>> {
            let (shape, slice) = outputs[name].try_extract_tensor::<f32>()?;
            Ok(ArrayView4::from_shape(
                (
                    shape[0] as usize,
                    shape[1] as usize,
                    shape[2] as usize,
                    shape[3] as usize,
                ),
                slice,
            )
            .unwrap()
            .to_owned())
        };

        Ok(Sam3ImageEmbeddings {
            vision_pos_enc_2: extract("vision_pos_enc_2")?,
            backbone_fpn_0: extract("backbone_fpn_0")?,
            backbone_fpn_1: extract("backbone_fpn_1")?,
            backbone_fpn_2: extract("backbone_fpn_2")?,
        })
    }

    fn decode_internal(
        &mut self,
        embeddings: &Sam3ImageEmbeddings,
        box_coords: [f32; 4],
        orig_w: i64,
        orig_h: i64,
    ) -> ort::Result<(Array4<f32>, Array2<f32>)> {
        let orig_w_t = Tensor::from_array(ndarray::arr0(orig_w))?.into_dyn();
        let orig_h_t = Tensor::from_array(ndarray::arr0(orig_h))?.into_dyn();

        let n_tok = self.config.language_token_count;
        let embed_dim = self.config.language_embed_dim;
        let language_mask = Array2::<bool>::from_elem((1, n_tok), false);
        let language_features = Array3::<f32>::zeros((n_tok, 1, embed_dim));

        let mut box_arr = Array3::<f32>::zeros((1, 1, 4));
        box_arr[[0, 0, 0]] = box_coords[0] / orig_w as f32;
        box_arr[[0, 0, 1]] = box_coords[1] / orig_h as f32;
        box_arr[[0, 0, 2]] = box_coords[2] / orig_w as f32;
        box_arr[[0, 0, 3]] = box_coords[3] / orig_h as f32;

        let box_labels = Array2::<i64>::from_elem((1, 1), 1);
        let box_masks = Array2::<bool>::from_elem((1, 1), true);

        let inputs = ort::inputs![
            "original_height" => orig_h_t,
            "original_width"  => orig_w_t,
            "vision_pos_enc_2"   => Tensor::from_array(embeddings.vision_pos_enc_2.clone())?.into_dyn(),
            "backbone_fpn_0"     => Tensor::from_array(embeddings.backbone_fpn_0.clone())?.into_dyn(),
            "backbone_fpn_1"     => Tensor::from_array(embeddings.backbone_fpn_1.clone())?.into_dyn(),
            "backbone_fpn_2"     => Tensor::from_array(embeddings.backbone_fpn_2.clone())?.into_dyn(),
            "language_mask"      => Tensor::from_array(language_mask)?.into_dyn(),
            "language_features"  => Tensor::from_array(language_features)?.into_dyn(),
            "box_coords"  => Tensor::from_array(box_arr)?.into_dyn(),
            "box_labels"  => Tensor::from_array(box_labels)?.into_dyn(),
            "box_masks"   => Tensor::from_array(box_masks)?.into_dyn()
        ];

        let outputs = self.decoder.run(inputs)?;

        let (shape_m, slice_m) = outputs["masks"].try_extract_tensor::<bool>()?;
        let masks_bool = ArrayView4::from_shape(
            (
                shape_m[0] as usize,
                shape_m[1] as usize,
                shape_m[2] as usize,
                shape_m[3] as usize,
            ),
            slice_m,
        )
        .unwrap();
        let masks = masks_bool.mapv(|b| if b { 1.0f32 } else { 0.0 });

        let (shape_i, slice_i) = outputs["scores"].try_extract_tensor::<f32>()?;
        let ious = if shape_i.len() == 1 {
            ArrayView2::from_shape((1, shape_i[0] as usize), slice_i)
                .unwrap()
                .to_owned()
        } else {
            ArrayView2::from_shape((shape_i[0] as usize, shape_i[1] as usize), slice_i)
                .unwrap()
                .to_owned()
        };

        Ok((masks, ious))
    }
}
