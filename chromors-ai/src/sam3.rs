use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView4};
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

pub struct Sam3Model {
    encoder: Session,
    decoder: Session,
}

pub struct Sam3ImageEmbeddings {
    vision_pos_enc_2: Array4<f32>,
    backbone_fpn_0: Array4<f32>,
    backbone_fpn_1: Array4<f32>,
    backbone_fpn_2: Array4<f32>,
}

impl Sam3Model {
    pub fn new(encoder_path: &str, decoder_path: &str) -> ort::Result<Self> {
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

        Ok(Self { encoder, decoder })
    }

    /// Encodes an `Image2D` into SAM3 embeddings.
    ///
    /// SAM3 expects raw HWC U8 (no ImageNet normalization — the encoder
    /// handles normalization internally).
    pub fn encode(
        &mut self,
        image: &Image2D<VipsBackend>,
    ) -> Result<Sam3ImageEmbeddings, poc::error::Error> {
        let (w, h) = image.spec.dims();

        let preprocessed = image
            .resize(1024.0 / w as f64, None, Some(1024.0 / h as f64), None)
            .convert(RGB_U8_LAYOUT);

        let bytes = preprocessed.pull(
            &RamImageTarget,
            Region::full((1024, 1024), Lod(0)),
        )?;

        // SAM3 encoder takes HWC u8 directly (Array3<u8>)
        let mut image_arr = Array3::<u8>::zeros((1024, 1024, 3));
        for y in 0..1024usize {
            for x in 0..1024usize {
                let idx = (y * 1024 + x) * 3;
                image_arr[[y, x, 0]] = bytes[idx];
                image_arr[[y, x, 1]] = bytes[idx + 1];
                image_arr[[y, x, 2]] = bytes[idx + 2];
            }
        }

        self.encode_internal(image_arr)
            .map_err(|e| poc::error::Error::Backend(format!("SAM3 encode ORT: {e:?}")))
    }

    /// Segments an image region defined by a bounding box.
    ///
    /// Returns the best mask as `Image2D<VipsBackend>` at original resolution,
    /// plus IoU confidence scores.
    pub fn segment_box(
        &mut self,
        embeddings: &Sam3ImageEmbeddings,
        box_coords: [f32; 4],
        original_size: (i32, i32),
    ) -> Result<(Image2D<VipsBackend>, Vec<f32>), poc::error::Error> {
        let (orig_w, orig_h) = original_size;

        let (masks, ious) = self
            .decode_internal(embeddings, box_coords, orig_w as i64, orig_h as i64)
            .map_err(|e| poc::error::Error::Backend(format!("SAM3 decode ORT: {e:?}")))?;

        // Pick best mask
        let num_masks = ious.dim().1;
        let mut best_idx = 0;
        let mut best_iou = -1.0f32;
        let mut iou_scores = Vec::with_capacity(num_masks);
        for i in 0..num_masks {
            let score = ious[[0, i]];
            iou_scores.push(score);
            if score > best_iou {
                best_iou = score;
                best_idx = i;
            }
        }

        let mask_h = masks.dim().2;
        let mask_w = masks.dim().3;
        let mut out_bytes = vec![0u8; mask_h * mask_w * 3];
        for y in 0..mask_h {
            for x in 0..mask_w {
                let v = if masks[[0, best_idx, y, x]] > 0.5 { 255u8 } else { 0u8 };
                let idx = (y * mask_w + x) * 3;
                out_bytes[idx] = v;
                out_bytes[idx + 1] = v;
                out_bytes[idx + 2] = v;
            }
        }

        let mask_img = Image2D::<VipsBackend>::from_bytes(
            out_bytes,
            mask_w as i32,
            mask_h as i32,
            RGB_U8_LAYOUT,
        );

        Ok((mask_img, iou_scores))
    }

    // ── Private inference ─────────────────────────────────────────────────

    fn encode_internal(&mut self, image: Array3<u8>) -> ort::Result<Sam3ImageEmbeddings> {
        let image_tensor = Tensor::from_array(image)?.into_dyn();
        let inputs = ort::inputs!["image" => image_tensor];
        let outputs = self.encoder.run(inputs)?;

        let extract = |name: &str| -> ort::Result<Array4<f32>> {
            let (shape, slice) = outputs[name].try_extract_tensor::<f32>()?;
            Ok(ArrayView4::from_shape(
                (shape[0] as usize, shape[1] as usize, shape[2] as usize, shape[3] as usize),
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

        let language_mask = Array2::<bool>::from_elem((1, 32), false);
        let language_features = Array3::<f32>::zeros((32, 1, 256));

        let mut box_arr = Array3::<f32>::zeros((1, 1, 4));
        box_arr[[0, 0, 0]] = box_coords[0] / orig_w as f32;
        box_arr[[0, 0, 1]] = box_coords[1] / orig_h as f32;
        box_arr[[0, 0, 2]] = box_coords[2] / orig_w as f32;
        box_arr[[0, 0, 3]] = box_coords[3] / orig_h as f32;

        let box_labels = Array2::<i64>::from_elem((1, 1), 1);
        let box_masks = Array2::<bool>::from_elem((1, 1), true);

        let inputs = ort::inputs![
            "original_height" => orig_h_t,
            "original_width" => orig_w_t,
            "vision_pos_enc_2" => Tensor::from_array(embeddings.vision_pos_enc_2.clone())?.into_dyn(),
            "backbone_fpn_0" => Tensor::from_array(embeddings.backbone_fpn_0.clone())?.into_dyn(),
            "backbone_fpn_1" => Tensor::from_array(embeddings.backbone_fpn_1.clone())?.into_dyn(),
            "backbone_fpn_2" => Tensor::from_array(embeddings.backbone_fpn_2.clone())?.into_dyn(),
            "language_mask" => Tensor::from_array(language_mask)?.into_dyn(),
            "language_features" => Tensor::from_array(language_features)?.into_dyn(),
            "box_coords" => Tensor::from_array(box_arr)?.into_dyn(),
            "box_labels" => Tensor::from_array(box_labels)?.into_dyn(),
            "box_masks" => Tensor::from_array(box_masks)?.into_dyn()
        ];

        let outputs = self.decoder.run(inputs)?;

        let (shape_m, slice_m) = outputs["masks"].try_extract_tensor::<bool>()?;
        let masks_bool = ArrayView4::from_shape(
            (shape_m[0] as usize, shape_m[1] as usize, shape_m[2] as usize, shape_m[3] as usize),
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
