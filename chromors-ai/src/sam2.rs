//! SAM2 — Segment Anything Model 2.
//!
//! Input: `Image2D<B>` (any size, any layout) + point/box prompts
//! Output: `Mask2D<B>` (binary or soft segmentation mask)

use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, ArrayView4};
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

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// SAM2 point label constants.
pub mod label {
    pub const FOREGROUND: i32 = 1;
    pub const BACKGROUND: i32 = 0;
    pub const BOX_TOP_LEFT: i32 = 2;
    pub const BOX_BOTTOM_RIGHT: i32 = 3;
}

#[derive(Debug, Clone)]
pub struct Sam2Config {
    /// Encoder input resolution (default: 1024).
    pub encoder_size: usize,
    /// Sigmoid threshold for mask binarization (default: 0.5).
    pub mask_threshold: f32,
    pub mask_selection: MaskSelection,
    /// Return a soft (sigmoid) mask instead of binary (default: false).
    pub soft_mask: bool,
    pub normalize_mean: [f32; 3],
    pub normalize_std: [f32; 3],
}

#[derive(Debug, Clone)]
pub enum MaskSelection {
    BestIoU,
    Index(usize),
}

impl Default for Sam2Config {
    fn default() -> Self {
        Self {
            encoder_size: 1024,
            mask_threshold: 0.5,
            mask_selection: MaskSelection::BestIoU,
            soft_mask: false,
            normalize_mean: IMAGENET_MEAN,
            normalize_std: IMAGENET_STD,
        }
    }
}

pub struct Sam2Model {
    encoder: Session,
    decoder: Session,
    config: Sam2Config,
}

pub struct Sam2Embeddings {
    high_res_feats_0: Array4<f32>,
    high_res_feats_1: Array4<f32>,
    image_embed: Array4<f32>,
}

#[derive(Debug, Clone)]
pub enum Sam2Prompt {
    Point { x: f32, y: f32, label: i32 },
    BoundingBox { x1: f32, y1: f32, x2: f32, y2: f32 },
}

pub struct Sam2Result<B: AiBackend> {
    pub mask: Mask2D<B>,
    pub iou_scores: Vec<f32>,
    pub selected_mask_index: usize,
}

impl Sam2Model {
    pub fn new(encoder_path: &str, decoder_path: &str) -> ort::Result<Self> {
        Self::with_config(encoder_path, decoder_path, Sam2Config::default())
    }

    pub fn with_config(
        encoder_path: &str,
        decoder_path: &str,
        config: Sam2Config,
    ) -> ort::Result<Self> {
        let encoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])?
            .commit_from_file(encoder_path)?;

        let decoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
                ort::execution_providers::CoreMLExecutionProvider::default().build(),
            ])?
            .commit_from_file(decoder_path)?;

        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub fn config(&self) -> &Sam2Config {
        &self.config
    }

    pub fn encode<B: AiBackend>(
        &mut self,
        image: &Image2D<B>,
    ) -> Result<Sam2Embeddings, chromors::error::Error> {
        let (w, h) = image.spec.dims();
        let sz = self.config.encoder_size;

        let preprocessed = image
            .resize(sz as f64 / w as f64, None, Some(sz as f64 / h as f64), None)
            .convert(RGB_U8_LAYOUT);

        let bytes = preprocessed.pull(
            &RamImageTarget,
            Region::full((sz as i32, sz as i32), Lod(0)),
        )?;

        let image_arr = hwc_u8_to_nchw_normalized(
            &bytes,
            sz,
            sz,
            &self.config.normalize_mean,
            &self.config.normalize_std,
        );
        self.encode_internal(image_arr)
            .map_err(|e| chromors::error::Error::Backend(format!("SAM2 encode: {e:?}")))
    }

    pub fn segment<B: AiBackend>(
        &mut self,
        embeddings: &Sam2Embeddings,
        prompts: &[Sam2Prompt],
        original_size: (i32, i32),
    ) -> Result<Sam2Result<B>, chromors::error::Error> {
        if prompts.is_empty() {
            return Err(chromors::error::Error::Backend(
                "SAM2: at least one prompt required".into(),
            ));
        }

        let mut points = Vec::new();
        let mut labels = Vec::new();
        for prompt in prompts {
            match prompt {
                Sam2Prompt::Point { x, y, label } => {
                    points.push((*x, *y));
                    labels.push(*label);
                }
                Sam2Prompt::BoundingBox { x1, y1, x2, y2 } => {
                    points.extend([(*x1, *y1), (*x2, *y2)]);
                    labels.extend([label::BOX_TOP_LEFT, label::BOX_BOTTOM_RIGHT]);
                }
            }
        }

        self.segment_points(embeddings, &points, &labels, original_size)
    }

    pub fn segment_points<B: AiBackend>(
        &mut self,
        embeddings: &Sam2Embeddings,
        points: &[(f32, f32)],
        labels: &[i32],
        original_size: (i32, i32),
    ) -> Result<Sam2Result<B>, chromors::error::Error> {
        if points.len() != labels.len() {
            return Err(chromors::error::Error::Backend(format!(
                "SAM2: points ({}) and labels ({}) length mismatch",
                points.len(),
                labels.len()
            )));
        }

        let (masks, ious) = self
            .decode_internal(embeddings, points, labels)
            .map_err(|e| chromors::error::Error::Backend(format!("SAM2 decode: {e:?}")))?;

        let num_masks = ious.dim().1;
        let iou_scores: Vec<f32> = (0..num_masks).map(|i| ious[[0, i]]).collect();

        let selected = match &self.config.mask_selection {
            MaskSelection::BestIoU => iou_scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
            MaskSelection::Index(idx) => {
                if *idx >= num_masks {
                    return Err(chromors::error::Error::Backend(format!(
                        "SAM2: mask index {idx} out of range ({num_masks} masks)"
                    )));
                }
                *idx
            }
        };

        let mask_h = masks.dim().2;
        let mask_w = masks.dim().3;
        let thresh = self.config.mask_threshold;
        let soft = self.config.soft_mask;

        let mask_values: Vec<f32> = (0..mask_h)
            .flat_map(|y| {
                (0..mask_w).map(move |x| {
                    let sig = 1.0 / (1.0 + (-masks[[0, selected, y, x]]).exp());
                    if soft {
                        sig
                    } else if sig > thresh {
                        1.0
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        let _ = original_size; // TODO: upscale mask when Mask2D supports resize

        Ok(Sam2Result {
            mask: B::mask_from_values(&mask_values, mask_w as i32, mask_h as i32),
            iou_scores,
            selected_mask_index: selected,
        })
    }

    fn encode_internal(&mut self, image: Array4<f32>) -> ort::Result<Sam2Embeddings> {
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

        Ok(Sam2Embeddings {
            high_res_feats_0: extract("high_res_feats_0")?,
            high_res_feats_1: extract("high_res_feats_1")?,
            image_embed: extract("image_embed")?,
        })
    }

    fn decode_internal(
        &mut self,
        embeddings: &Sam2Embeddings,
        points: &[(f32, f32)],
        labels: &[i32],
    ) -> ort::Result<(Array4<f32>, Array2<f32>)> {
        let mut point_coords = Array3::<f32>::zeros((1, points.len(), 2));
        for (i, &(x, y)) in points.iter().enumerate() {
            point_coords[[0, i, 0]] = x;
            point_coords[[0, i, 1]] = y;
        }

        let mut point_labels = Array2::<f32>::zeros((1, labels.len()));
        for (i, &l) in labels.iter().enumerate() {
            point_labels[[0, i]] = l as f32;
        }

        let mask_input = Array4::<f32>::zeros((1, 1, 256, 256));
        let has_mask_input = Array1::<f32>::zeros(1);

        let inputs = ort::inputs![
            "image_embed"      => Tensor::from_array(embeddings.image_embed.clone())?.into_dyn(),
            "high_res_feats_0" => Tensor::from_array(embeddings.high_res_feats_0.clone())?.into_dyn(),
            "high_res_feats_1" => Tensor::from_array(embeddings.high_res_feats_1.clone())?.into_dyn(),
            "point_coords"     => Tensor::from_array(point_coords)?.into_dyn(),
            "point_labels"     => Tensor::from_array(point_labels)?.into_dyn(),
            "mask_input"       => Tensor::from_array(mask_input)?.into_dyn(),
            "has_mask_input"   => Tensor::from_array(has_mask_input)?.into_dyn()
        ];

        let outputs = self.decoder.run(inputs)?;

        let (shape_m, slice_m) = outputs["masks"].try_extract_tensor::<f32>()?;
        let masks = ArrayView4::from_shape(
            (
                shape_m[0] as usize,
                shape_m[1] as usize,
                shape_m[2] as usize,
                shape_m[3] as usize,
            ),
            slice_m,
        )
        .unwrap()
        .to_owned();

        let (shape_i, slice_i) = outputs["iou_predictions"].try_extract_tensor::<f32>()?;
        let ious = ArrayView2::from_shape((shape_i[0] as usize, shape_i[1] as usize), slice_i)
            .unwrap()
            .to_owned();

        Ok((masks, ious))
    }
}

fn hwc_u8_to_nchw_normalized(
    bytes: &[u8],
    w: usize,
    h: usize,
    mean: &[f32; 3],
    std: &[f32; 3],
) -> Array4<f32> {
    let mut arr = Array4::<f32>::zeros((1, 3, h, w));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            for c in 0..3 {
                arr[[0, c, y, x]] = (bytes[idx + c] as f32 / 255.0 - mean[c]) / std[c];
            }
        }
    }
    arr
}
