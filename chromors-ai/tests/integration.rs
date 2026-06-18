//! Integration tests for all chromors-ai models.
//!
//! Run with:
//!   LD_LIBRARY_PATH=$(find target/debug/build -name "libslang-compiler*" -printf "%h\n" | head -1) \
//!   cargo test -p chromors-ai --all-features -- --ignored --nocapture
//!
//! All results are saved to target/test-output/chromors-ai/ for visual inspection.

use std::path::PathBuf;

fn output_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("target")
        .join("test-output")
        .join("chromors-ai");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("assets")
}


/// Room interior — geometric structures for ZITS.
fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn asset(name: &str) -> String {
    assets_dir().join(name).to_str().unwrap().to_string()
}

const RGB_U8: chromors::pixel::PixelLayout = chromors::pixel::PixelLayout {
    storage: chromors::pixel::Storage::U8,
    model: chromors::color::model::ColorModel::Rgb,
    alpha: chromors::pixel::AlphaState::None,
    color_space: chromors::color::space::ColorSpace::SRGB,
};

fn save_png(img: &chromors::data::image::Image2D<chromors::backend::vips::VipsBackend>, name: &str) {
    let path = output_dir().join(format!("{name}.png"));
    let config = chromors::export::ExportConfig::Png(chromors::export::png::PngExportConfig::default());
    img.save_with_config(path.to_str().unwrap(), &config).unwrap();
    println!("  ✓ Saved: {}", path.display());
}

/// Pull mask f32 values from a Mask2D.
fn pull_mask(mask: &chromors::data::mask2d::Mask2D<chromors::backend::vips::VipsBackend>) -> Vec<f32> {
    use chromors::data::mask2d::RamMaskTarget;
    use chromors::work_unit::{Lod, Region};
    mask.pull(&RamMaskTarget, Region::full((mask.width(), mask.height()), Lod(0)))
        .unwrap()
}

/// Save a mask as a grayscale PNG.
fn save_mask_as_png(
    mask: &chromors::data::mask2d::Mask2D<chromors::backend::vips::VipsBackend>,
    name: &str,
) {
    let w = mask.width();
    let h = mask.height();
    let values = pull_mask(mask);

    let mut bytes = vec![0u8; (w * h * 3) as usize];
    for i in 0..(w * h) as usize {
        let v = (values[i].clamp(0.0, 1.0) * 255.0) as u8;
        bytes[i * 3] = v;
        bytes[i * 3 + 1] = v;
        bytes[i * 3 + 2] = v;
    }

    let img = chromors::data::image::Image2D::<chromors::backend::vips::VipsBackend>::from_bytes(
        bytes, w, h, RGB_U8,
    );
    save_png(&img, name);
}

/// Multiply an image by a mask (alpha composite) and save.
/// Shows the original image where mask > 0, black where mask = 0.
fn save_mask_composite(
    image: &chromors::data::image::Image2D<chromors::backend::vips::VipsBackend>,
    mask: &chromors::data::mask2d::Mask2D<chromors::backend::vips::VipsBackend>,
    name: &str,
) {
    use chromors::data::image::RamImageTarget;
    use chromors::work_unit::{Lod, Region};

    let iw = image.width();
    let ih = image.height();
    let mw = mask.width();
    let mh = mask.height();

    let rgb = image.clone().convert(RGB_U8);
    let img_bytes = rgb.pull(
        &RamImageTarget,
        Region::full((iw, ih), Lod(0)),
    ).unwrap();
    let mask_values = pull_mask(mask);

    let mut out = vec![0u8; (iw * ih * 3) as usize];
    for y in 0..ih as usize {
        for x in 0..iw as usize {
            // Map image coords to mask coords (they might differ in size)
            let mx = (x as f32 * mw as f32 / iw as f32) as usize;
            let my = (y as f32 * mh as f32 / ih as f32) as usize;
            let mi = my * mw as usize + mx;
            let alpha = if mi < mask_values.len() {
                mask_values[mi].clamp(0.0, 1.0)
            } else {
                0.0
            };

            let src = (y * iw as usize + x) * 3;
            let dst = src;
            out[dst] = (img_bytes[src] as f32 * alpha) as u8;
            out[dst + 1] = (img_bytes[src + 1] as f32 * alpha) as u8;
            out[dst + 2] = (img_bytes[src + 2] as f32 * alpha) as u8;
        }
    }

    let composite = chromors::data::image::Image2D::<chromors::backend::vips::VipsBackend>::from_bytes(
        out, iw, ih, RGB_U8,
    );
    save_png(&composite, name);
}

/// Multiply an Image2D mask (binary white/black) with the original image.
fn save_image_mask_composite(
    image: &chromors::data::image::Image2D<chromors::backend::vips::VipsBackend>,
    mask_img: &chromors::data::image::Image2D<chromors::backend::vips::VipsBackend>,
    name: &str,
) {
    use chromors::data::image::RamImageTarget;
    use chromors::work_unit::{Lod, Region};

    let iw = image.width();
    let ih = image.height();
    let mw = mask_img.width();
    let mh = mask_img.height();

    let rgb = image.clone().convert(RGB_U8);
    let img_bytes = rgb.pull(
        &RamImageTarget,
        Region::full((iw, ih), Lod(0)),
    ).unwrap();

    let mask_rgb = mask_img.clone().convert(RGB_U8);
    let mask_bytes = mask_rgb.pull(
        &RamImageTarget,
        Region::full((mw, mh), Lod(0)),
    ).unwrap();

    let mut out = vec![0u8; (iw * ih * 3) as usize];
    for y in 0..ih as usize {
        for x in 0..iw as usize {
            let mx = (x as f32 * mw as f32 / iw as f32) as usize;
            let my = (y as f32 * mh as f32 / ih as f32) as usize;
            let mi = (my * mw as usize + mx) * 3;
            let alpha = if mi < mask_bytes.len() {
                mask_bytes[mi] as f32 / 255.0
            } else {
                0.0
            };

            let src = (y * iw as usize + x) * 3;
            out[src] = (img_bytes[src] as f32 * alpha) as u8;
            out[src + 1] = (img_bytes[src + 1] as f32 * alpha) as u8;
            out[src + 2] = (img_bytes[src + 2] as f32 * alpha) as u8;
        }
    }

    let composite = chromors::data::image::Image2D::<chromors::backend::vips::VipsBackend>::from_bytes(
        out, iw, ih, RGB_U8,
    );
    save_png(&composite, name);
}

fn init_ort() {
    let _ = ort::init().with_name("chromors-ai-tests").commit();
}

// ── MODNet ───────────────────────────────────────────────────────────────────

#[cfg(feature = "modnet")]
mod modnet_tests {
    use super::*;
    use chromors_ai::modnet::ModNetModel;
    use chromors::data::image::Image2D;
    use chromors::backend::vips::VipsBackend;

    #[test]
    #[ignore]
    fn test_modnet_portrait_matte() {
        init_ort();
        println!("\n=== MODNet Portrait Matting ===");

        let model_path = models_dir().join("modnet/modnet_photographic.onnx");
        if !model_path.exists() {
            println!("  ⏭ Skipped (model not downloaded). Run: python chromors-ai/download_models.py modnet");
            return;
        }

        let mut model = ModNetModel::new(model_path.to_str().unwrap()).unwrap();
        // Portrait with clear hair → shows matting quality
        let img = Image2D::<VipsBackend>::open(&asset("portrait.jpg")).unwrap();
        println!("  Input: {}x{}", img.width(), img.height());

        let mask = model.matte(&img).unwrap();
        println!("  Output mask: {}x{}", mask.width(), mask.height());

        save_png(&img, "modnet_input");
        save_mask_as_png(&mask, "modnet_alpha");
        save_mask_composite(&img, &mask, "modnet_cutout");
    }
}

// ── Real-ESRGAN ──────────────────────────────────────────────────────────────

#[cfg(feature = "realesrgan")]
mod realesrgan_tests {
    use super::*;
    use chromors_ai::realesrgan::RealEsrganModel;
    use chromors::data::image::Image2D;
    use chromors::backend::vips::VipsBackend;

    #[test]
    #[ignore]
    fn test_realesrgan_upscale() {
        init_ort();
        println!("\n=== Real-ESRGAN 4× Upscale ===");

        let model_path = models_dir().join("realesrgan/realesrgan_x4plus.onnx");
        if !model_path.exists() {
            println!("  ⏭ Skipped (model not downloaded). Run: python chromors-ai/download_models.py realesrgan");
            return;
        }

        let mut model = RealEsrganModel::new(model_path.to_str().unwrap()).unwrap();

        // City buildings crop → shows detail recovery in upscale
        let small = Image2D::<VipsBackend>::open(&asset("city_crop_256.jpg")).unwrap();
        println!("  Input: {}x{}", small.width(), small.height());

        let upscaled = model.upscale(&small).unwrap();
        println!("  Output: {}x{} (4× upscaled)", upscaled.width(), upscaled.height());

        save_png(&small, "realesrgan_input_small");
        save_png(&upscaled, "realesrgan_output_4x");
    }
}

// ── Depth Anything V2 ────────────────────────────────────────────────────────

#[cfg(feature = "depth_anything")]
mod depth_anything_tests {
    use super::*;
    use chromors_ai::depth_anything::DepthAnythingModel;
    use chromors::data::image::Image2D;
    use chromors::backend::vips::VipsBackend;

    #[test]
    #[ignore]
    fn test_depth_anything_road() {
        init_ort();
        println!("\n=== Depth Anything V2 — Road Scene ===");

        let model_path = models_dir().join("depth_anything/depth_anything_v2_small.onnx");
        if !model_path.exists() {
            println!("  ⏭ Skipped (model not downloaded). Run: python chromors-ai/download_models.py depth_anything");
            return;
        }

        let mut model = DepthAnythingModel::new(model_path.to_str().unwrap()).unwrap();
        let img = Image2D::<VipsBackend>::open(&asset("road.jpg")).unwrap();
        println!("  Input: {}x{}", img.width(), img.height());

        let depth = model.estimate(&img).unwrap();
        println!("  Output depth map: {}x{}", depth.width(), depth.height());

        save_png(&img, "depth_road_input");
        save_mask_as_png(&depth, "depth_road_map");
        save_mask_composite(&img, &depth, "depth_road_composite");
    }

    #[test]
    #[ignore]
    fn test_depth_anything_landscape() {
        init_ort();
        println!("\n=== Depth Anything V2 — Landscape ===");

        let model_path = models_dir().join("depth_anything/depth_anything_v2_small.onnx");
        if !model_path.exists() {
            println!("  ⏭ Skipped");
            return;
        }

        let mut model = DepthAnythingModel::new(model_path.to_str().unwrap()).unwrap();
        let img = Image2D::<VipsBackend>::open(&asset("landscape.jpg")).unwrap();
        println!("  Input: {}x{}", img.width(), img.height());

        let depth = model.estimate(&img).unwrap();
        save_mask_as_png(&depth, "depth_landscape_map");
        save_mask_composite(&img, &depth, "depth_landscape_composite");
    }
}

// ── SwinIR ───────────────────────────────────────────────────────────────────

#[cfg(feature = "swinir")]
mod swinir_tests {
    use super::*;
    use chromors_ai::swinir::SwinIrModel;
    use chromors::data::image::Image2D;
    use chromors::backend::vips::VipsBackend;

    #[test]
    #[ignore]
    fn test_swinir_denoise() {
        init_ort();
        println!("\n=== SwinIR Denoise ===");

        let model_path = models_dir().join("swinir/swinir_denoise_color_15.onnx");
        if !model_path.exists() {
            println!("  ⏭ Skipped (model not downloaded). Run: python chromors-ai/download_models.py swinir");
            return;
        }

        let mut model = SwinIrModel::new(model_path.to_str().unwrap()).unwrap();

        let img = Image2D::<VipsBackend>::open(&asset("city_street.jpg")).unwrap();
        let small = img.resize(256.0 / img.width() as f64, None, Some(256.0 / img.height() as f64), None);
        println!("  Input: {}x{}", small.width(), small.height());

        let restored = model.restore(&small).unwrap();
        println!("  Output: {}x{}", restored.width(), restored.height());

        save_png(&small, "swinir_input");
        save_png(&restored, "swinir_output_denoised");
    }
}

// ── ViTMatte ─────────────────────────────────────────────────────────────────

#[cfg(feature = "vitmatte")]
mod vitmatte_tests {
    use super::*;
    use chromors_ai::vitmatte::ViTMatteModel;
    use chromors_ai::modnet::ModNetModel;
    use chromors::data::image::Image2D;
    use chromors::data::mask2d::Mask2D;
    use chromors::backend::vips::VipsBackend;

    #[test]
    #[ignore]
    fn test_vitmatte_trimap() {
        init_ort();
        println!("\n=== ViTMatte Trimap Matting (MODNet → trimap → refine) ===");

        let modnet_path = models_dir().join("modnet/modnet_photographic.onnx");
        let vitmatte_path = models_dir().join("vitmatte/vitmatte_small.onnx");
        if !modnet_path.exists() || !vitmatte_path.exists() {
            println!("  ⏭ Skipped (models not downloaded). Run: python chromors-ai/download_models.py modnet vitmatte");
            return;
        }

        let img = Image2D::<VipsBackend>::open(&asset("portrait.jpg")).unwrap();
        let w = img.width();
        let h = img.height();
        println!("  Input: {}x{}", w, h);

        // Step 1: MODNet rough alpha
        let mut modnet = ModNetModel::new(modnet_path.to_str().unwrap()).unwrap();
        let rough_alpha = modnet.matte(&img).unwrap();
        println!("  MODNet rough alpha: {}x{}", rough_alpha.width(), rough_alpha.height());
        save_mask_as_png(&rough_alpha, "vitmatte_1_rough_alpha");

        // Step 2: Convert rough alpha to trimap via erode/dilate simulation
        //   alpha > 0.8 → foreground (1.0)
        //   alpha < 0.2 → background (0.0)
        //   else         → unknown    (0.5)
        let alpha_values = pull_mask(&rough_alpha);
        let mw = rough_alpha.width() as usize;
        let mh = rough_alpha.height() as usize;
        let mut trimap_values = vec![0.0f32; mw * mh];
        for i in 0..alpha_values.len() {
            trimap_values[i] = if alpha_values[i] > 0.8 {
                1.0
            } else if alpha_values[i] < 0.2 {
                0.0
            } else {
                0.5
            };
        }

        // Resize trimap to image dimensions
        let trimap = Mask2D::<VipsBackend>::from_values(mw as i32, mh as i32, &trimap_values);
        save_mask_as_png(&trimap, "vitmatte_2_trimap");

        // Step 3: ViTMatte refines the trimap
        let mut vitmatte = ViTMatteModel::new(vitmatte_path.to_str().unwrap()).unwrap();
        let alpha = vitmatte.matte(&img, &trimap).unwrap();
        println!("  Output alpha: {}x{}", alpha.width(), alpha.height());

        save_mask_as_png(&alpha, "vitmatte_3_refined_alpha");
        save_mask_composite(&img, &alpha, "vitmatte_4_cutout");
    }
}

// ── LaMa ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "lama")]
mod lama_tests {
    use super::*;
    use chromors_ai::lama::LamaModel;
    use chromors::data::image::Image2D;
    use chromors::data::mask2d::Mask2D;
    use chromors::backend::vips::VipsBackend;

    #[test]
    #[ignore]
    fn test_lama_inpaint() {
        init_ort();
        println!("\n=== LaMa Inpainting ===");

        let model_path = models_dir().join("lama/lama_fp32.onnx");
        if !model_path.exists() {
            println!("  ⏭ Skipped (model not downloaded). Run: python chromors-ai/download_models.py lama");
            return;
        }

        let mut model = LamaModel::new(model_path.to_str().unwrap()).unwrap();

        // City street → remove a building/object from the scene
        let img = Image2D::<VipsBackend>::open(&asset("city_street.jpg")).unwrap();
        let img_small = img.resize(512.0 / img.width() as f64, None, Some(512.0 / img.height() as f64), None);
        let w = img_small.width();
        let h = img_small.height();
        println!("  Input: {}x{}", w, h);

        // Create a hole mask: rectangle in the center
        let mut mask_values = vec![0.0f32; (w * h) as usize];
        let x0 = w / 4;
        let x1 = 3 * w / 4;
        let y0 = h / 4;
        let y1 = 3 * h / 4;
        for y in y0..y1 {
            for x in x0..x1 {
                mask_values[(y * w + x) as usize] = 1.0;
            }
        }

        let mask = Mask2D::<VipsBackend>::from_values(w, h, &mask_values);
        save_png(&img_small, "lama_input");
        save_mask_as_png(&mask, "lama_mask");

        let inpainted = model.inpaint(&img_small, &mask).unwrap();
        println!("  Output: {}x{}", inpainted.width(), inpainted.height());
        save_png(&inpainted, "lama_output");
    }
}


// ── SAM2 ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "sam2")]
mod sam2_tests {
    use super::*;
    use chromors_ai::sam2::{Sam2Model, Sam2Prompt, label};
    use chromors::data::image::Image2D;
    use chromors::backend::vips::VipsBackend;

    #[test]
    #[ignore]
    fn test_sam2_segment_center() {
        init_ort();
        println!("\n=== SAM2 Center Point Segmentation ===");

        let enc_path = models_dir().join("sam2/sam2_hiera_tiny.encoder.onnx");
        let dec_path = models_dir().join("sam2/sam2_hiera_tiny.decoder.onnx");
        if !enc_path.exists() || !dec_path.exists() {
            println!("  ⏭ Skipped (models not downloaded). Run: python chromors-ai/download_models.py sam2");
            return;
        }

        let mut model = Sam2Model::new(
            enc_path.to_str().unwrap(),
            dec_path.to_str().unwrap(),
        )
        .unwrap();

        let img = Image2D::<VipsBackend>::open(&asset("city_street.jpg")).unwrap();
        let (w, h) = img.spec.dims();
        println!("  Input: {}x{}", w, h);

        let embeddings = model.encode(&img).unwrap();
        println!("  Encoded.");

        // Point at image center (SAM2 coords are in 1024×1024 encoder space)
        let result = model
            .segment(&embeddings, &[
                Sam2Prompt::Point { x: 512.0, y: 512.0, label: label::FOREGROUND },
            ], (w, h))
            .unwrap();
        println!("  IoU scores: {:?}", result.iou_scores);
        println!("  Selected mask: {}", result.selected_mask_index);

        save_mask_as_png(&result.mask, "sam2_mask");
        save_mask_composite(&img, &result.mask, "sam2_cutout");
    }
}

// ── CascadePSP ──────────────────────────────────────────────────────────────

#[cfg(feature = "cascadepsp")]
mod cascadepsp_tests {
    use super::*;
    use chromors_ai::cascadepsp::CascadePspModel;
    use chromors::data::image::Image2D;
    use chromors::backend::vips::VipsBackend;

    #[test]
    #[ignore]
    fn test_cascadepsp_refine() {
        init_ort();
        println!("\n=== CascadePSP Mask Refinement ===");

        let model_path = models_dir().join("cascadepsp/cascadepsp_base.onnx");
        if !model_path.exists() {
            println!("  ⏭ Skipped (model not downloaded). Run: python chromors-ai/download_models.py cascadepsp");
            return;
        }

        let mut model = CascadePspModel::new(model_path.to_str().unwrap()).unwrap();
        let img = Image2D::<VipsBackend>::open(&asset("city_street.jpg")).unwrap();
        let (w, h) = img.spec.dims();
        println!("  Input: {}x{}", w, h);

        // Create a synthetic rough mask (circular blob)
        let mut mask_bytes = vec![0u8; (w * h * 3) as usize];
        let cx = w / 2;
        let cy = h / 2;
        let radius = w.min(h) / 3;
        for y in 0..h {
            for x in 0..w {
                let dx = (x - cx).abs();
                let dy = (y - cy).abs();
                if dx * dx + dy * dy < radius * radius {
                    let idx = ((y * w + x) * 3) as usize;
                    mask_bytes[idx] = 255;
                    mask_bytes[idx + 1] = 255;
                    mask_bytes[idx + 2] = 255;
                }
            }
        }
        let rough_mask = Image2D::<VipsBackend>::from_bytes(mask_bytes, w, h, RGB_U8);

        save_png(&rough_mask, "cascadepsp_rough_mask");

        let refined = model.refine(&img, &rough_mask).unwrap();
        println!("  Output: {}x{}", refined.width(), refined.height());
        save_png(&refined, "cascadepsp_refined_mask");
        save_image_mask_composite(&img, &refined, "cascadepsp_cutout");
    }
}
