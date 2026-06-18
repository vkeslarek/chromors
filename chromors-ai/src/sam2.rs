//! SAM2 — Segment Anything Model 2.
//!
//! Input: `Image2D` (any size, any layout) + point/box prompts
//! Output: `Mask2D` (binary or soft segmentation mask)

use ndarray::{Array1, Array2, Array3, Array4, ArrayView2, ArrayView4};
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

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// SAM2 point label constants.
pub mod label {
    /// Positive click — include this point in the mask.
    pub const FOREGROUND: i32 = 1;
    /// Negative click — exclude this point from the mask.
    pub const BACKGROUND: i32 = 0;
    /// Top-left corner of a bounding box prompt.
    pub const BOX_TOP_LEFT: i32 = 2;
    /// Bottom-right corner of a bounding box prompt.
    pub const BOX_BOTTOM_RIGHT: i32 = 3;
}

/// Configuration for SAM2 inference.
#[derive(Debug, Clone)]
pub struct Sam2Config {
    /// Encoder input resolution (default: 1024).
    pub encoder_size: usize,

    /// Sigmoid threshold for mask binarization (default: 0.5).
    /// Lower → more permissive masks, higher → tighter masks.
    pub mask_threshold: f32,

    /// Which mask to select from multi-mask output (default: `MaskSelection::BestIoU`).
    pub mask_selection: MaskSelection,

    /// Whether to return a soft (sigmoid) mask instead of binary (default: false).
    /// When true, mask values are continuous [0, 1] instead of {0, 1}.
    pub soft_mask: bool,

    /// ImageNet normalization mean.
    pub normalize_mean: [f32; 3],

    /// ImageNet normalization std.
    pub normalize_std: [f32; 3],
}

/// How to select among the multi-mask decoder outputs.
#[derive(Debug, Clone)]
pub enum MaskSelection {
    /// Pick the mask with the highest IoU prediction (default).
    BestIoU,
    /// Pick a specific mask by index (0, 1, or 2).
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

/// Cached image embeddings from the encoder — reuse across multiple `segment` calls.
pub struct Sam2Embeddings {
    high_res_feats_0: Array4<f32>,
    high_res_feats_1: Array4<f32>,
    image_embed: Array4<f32>,
}

/// A single prompt for SAM2 segmentation.
#[derive(Debug, Clone)]
pub enum Sam2Prompt {
    /// Click at a point with the given label.
    Point { x: f32, y: f32, label: i32 },
    /// Bounding box defined by top-left and bottom-right corners.
    BoundingBox { x1: f32, y1: f32, x2: f32, y2: f32 },
}

/// Result of a SAM2 segmentation.
pub struct Sam2Result {
    /// The selected segmentation mask.
    pub mask: Mask2D<VipsBackend>,
    /// IoU prediction scores for all candidate masks.
    pub iou_scores: Vec<f32>,
    /// Index of the selected mask.
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
            .commit_from_file(encoder_path)?;

        let decoder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(decoder_path)?;

        Ok(Self { encoder, decoder, config })
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &Sam2Config {
        &self.config
    }

    /// Encodes an image into reusable embeddings.
    ///
    /// Call this once, then use `segment` multiple times with different prompts.
    pub fn encode(
        &mut self,
        image: &Image2D<VipsBackend>,
    ) -> Result<Sam2Embeddings, poc::error::Error> {
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
            &bytes, sz, sz,
            &self.config.normalize_mean,
            &self.config.normalize_std,
        );

        self.encode_internal(image_arr)
            .map_err(|e| poc::error::Error::Backend(format!("SAM2 encode: {e:?}")))
    }

    /// Segments using high-level prompt types.
    ///
    /// Returns a `Sam2Result` with the mask, IoU scores, and selected mask index.
    pub fn segment(
        &mut self,
        embeddings: &Sam2Embeddings,
        prompts: &[Sam2Prompt],
        original_size: (i32, i32),
    ) -> Result<Sam2Result, poc::error::Error> {
        assert!(!prompts.is_empty(), "at least one prompt is required");

        let mut points = Vec::new();
        let mut labels = Vec::new();

        for prompt in prompts {
            match prompt {
                Sam2Prompt::Point { x, y, label } => {
                    points.push((*x, *y));
                    labels.push(*label);
                }
                Sam2Prompt::BoundingBox { x1, y1, x2, y2 } => {
                    points.push((*x1, *y1));
                    labels.push(label::BOX_TOP_LEFT);
                    points.push((*x2, *y2));
                    labels.push(label::BOX_BOTTOM_RIGHT);
                }
            }
        }

        self.segment_points(embeddings, &points, &labels, original_size)
    }

    /// Low-level segmentation with raw points and labels.
    pub fn segment_points(
        &mut self,
        embeddings: &Sam2Embeddings,
        points: &[(f32, f32)],
        labels: &[i32],
        original_size: (i32, i32),
    ) -> Result<Sam2Result, poc::error::Error> {
        assert_eq!(points.len(), labels.len(),
            "points ({}) and labels ({}) must have the same length",
            points.len(), labels.len());

        let (masks, ious) = self
            .decode_internal(embeddings, points, labels)
            .map_err(|e| poc::error::Error::Backend(format!("SAM2 decode: {e:?}")))?;

        let num_masks = ious.dim().1;
        let mut iou_scores = Vec::with_capacity(num_masks);
        for i in 0..num_masks {
            iou_scores.push(ious[[0, i]]);
        }

        let selected = match &self.config.mask_selection {
            MaskSelection::BestIoU => {
                iou_scores.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            MaskSelection::Index(idx) => {
                assert!(*idx < num_masks,
                    "mask index {} out of range (model produced {} masks)",
                    idx, num_masks);
                *idx
            }
        };

        let mask_h = masks.dim().2;
        let mask_w = masks.dim().3;
        let thresh = self.config.mask_threshold;
        let soft = self.config.soft_mask;

        let mut mask_values = vec![0.0f32; mask_h * mask_w];
        for y in 0..mask_h {
            for x in 0..mask_w {
                let val = masks[[0, selected, y, x]];
                let sig = 1.0 / (1.0 + (-val).exp());
                mask_values[y * mask_w + x] = if soft {
                    sig
                } else {
                    if sig > thresh { 1.0 } else { 0.0 }
                };
            }
        }

        let mask_small = Mask2D::<VipsBackend>::from_values(
            mask_w as i32, mask_h as i32, &mask_values,
        );

        // TODO: upscale mask to original_size when Mask2D supports resize
        let _ = original_size;

        Ok(Sam2Result {
            mask: mask_small,
            iou_scores,
            selected_mask_index: selected,
        })
    }

    // ── Private ─────────────────────────────────────────────────────────────

    fn encode_internal(&mut self, image: Array4<f32>) -> ort::Result<Sam2Embeddings> {
        let image_tensor = Tensor::from_array(image)?.into_dyn();
        let outputs = self.encoder.run(ort::inputs!["image" => image_tensor])?;

        let extract = |name: &str| -> ort::Result<Array4<f32>> {
            let (shape, slice) = outputs[name].try_extract_tensor::<f32>()?;
            Ok(ArrayView4::from_shape(
                (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize),
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
            "image_embed" => Tensor::from_array(embeddings.image_embed.clone())?.into_dyn(),
            "high_res_feats_0" => Tensor::from_array(embeddings.high_res_feats_0.clone())?.into_dyn(),
            "high_res_feats_1" => Tensor::from_array(embeddings.high_res_feats_1.clone())?.into_dyn(),
            "point_coords" => Tensor::from_array(point_coords)?.into_dyn(),
            "point_labels" => Tensor::from_array(point_labels)?.into_dyn(),
            "mask_input" => Tensor::from_array(mask_input)?.into_dyn(),
            "has_mask_input" => Tensor::from_array(has_mask_input)?.into_dyn()
        ];

        let outputs = self.decoder.run(inputs)?;

        let (shape_m, slice_m) = outputs["masks"].try_extract_tensor::<f32>()?;
        let masks = ArrayView4::from_shape(
            (shape_m[0] as usize, shape_m[1] as usize, shape_m[2] as usize, shape_m[3] as usize),
            slice_m,
        )
        .unwrap()
        .to_owned();

        let (shape_i, slice_i) = outputs["iou_predictions"].try_extract_tensor::<f32>()?;
        let ious = ArrayView2::from_shape(
            (shape_i[0] as usize, shape_i[1] as usize),
            slice_i,
        )
        .unwrap()
        .to_owned();

        Ok((masks, ious))
    }
}

fn hwc_u8_to_nchw_normalized(
    bytes: &[u8], w: usize, h: usize,
    mean: &[f32; 3], std: &[f32; 3],
) -> Array4<f32> {
    let mut arr = Array4::<f32>::zeros((1, 3, h, w));
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            for c in 0..3 {
                let val = bytes[idx + c] as f32 / 255.0;
                arr[[0, c, y, x]] = (val - mean[c]) / std[c];
            }
        }
    }
    arr
}
