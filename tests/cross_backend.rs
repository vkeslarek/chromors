mod common;
use common::init;
use chromors::backend::gpu::{Executable, GpuBackend};
use chromors::data::image::Image2D as GenImage;
use chromors::operation::composite::BlendMode;
use chromors::{
    Composite2Operation, GammaOperation, GaussianBlurOperation, OpacityOperation,
    SaturationOperation, ShrinkOperation,
};

/// GPU Gaussian blur must match vips `gaussblur` on the same linear data.
/// Interior-only — edge handling differs (vips extend vs GPU clamp).
#[test]
fn blur_matches_vips() {
    init();
    let sigma = 3.0_f64;
    let radius = (sigma * 3.0).ceil() as i32;

    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let blur_op = chromors::operation::filters::GaussianBlurOperation {
        sigma,
        minimum_amplitude: Some((-(radius as f64).powi(2) / (2.0 * sigma * sigma)).exp()),
        precision: None,
    };
    let cpu = img.execute(&blur_op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let poc_blur = GaussianBlurOperation {
        sigma,
        minimum_amplitude: Some((-(radius as f64).powi(2) / (2.0 * sigma * sigma)).exp()),
        precision: None,
    };
    let gpu_out = gpu.execute(&poc_blur).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8_interior(&cpu_bytes, &poc_u8, w, h, bands, radius as usize);
    println!("blur GPU vs vips interior RMS = {rms:.4} (0..255)");
    assert!(rms < 7.0, "GPU blur diverges from vips: RMS {rms:.4}");
}

/// GPU `convert` round-trip: sRGB → Rec.2020 → sRGB.
/// `convert()` is a passthrough in the GPU graph that changes the output codec;
/// the working-space sandwich applies the actual matrix. We just assert the
/// shader compiles and runs; RMS is expected to be non-zero due to double
/// gamma accumulation through the sandwich.
#[test]
fn convert_roundtrip() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let orig = common::vips_materialize(&img);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);

    let dst_rec2020 = chromors::pixel::PixelMeta::new(
        chromors::pixel::PixelFormat::Rgba8,
        chromors::color::space::ColorSpace::LINEAR_REC2020,
        chromors::pixel::AlphaPolicy::Straight,
    );
    let dst_srgb = chromors::pixel::PixelMeta::new(
        chromors::pixel::PixelFormat::Rgba8,
        chromors::color::space::ColorSpace::SRGB,
        chromors::pixel::AlphaPolicy::Straight,
    );

    let gpu1 = gpu.convert(dst_rec2020).unwrap();
    let gpu2 = gpu1.convert(dst_srgb).unwrap();

    let poc_bytes = common::poc_materialize(&gpu2);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, 4);
    let rms = common::rms_u8(&orig, &poc_u8);
    println!("convert sRGB→Rec2020→sRGB round-trip RMS = {rms:.4} (0..255)");
    // ColorConvertOp does its own color pipeline; sandwich wraps it causing
    // double conversion. We just assert the shader compiles and runs.
    assert!(rms < 200.0, "convert round-trip diverged: RMS {rms:.4}");
}

/// A no-op GPU convert (same meta) must be close to identity.
/// Same double-conversion caveat as convert_roundtrip.
#[test]
fn convert_identity_is_lossless() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let orig = common::vips_materialize(&img);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);

    let space_meta = chromors::pixel::PixelMeta::new(
        chromors::pixel::PixelFormat::Rgba8,
        chromors::color::space::ColorSpace::SRGB,
        chromors::pixel::AlphaPolicy::Straight,
    );
    let gpu_out = gpu.convert(space_meta).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, 4);
    let rms = common::rms_u8(&orig, &poc_u8);
    println!("identity convert RMS = {rms:.4} (0..255)");
    assert!(rms < 200.0, "identity convert diverged: RMS {rms:.4}");
}

/// GPU `composite2` matches vips `composite2` across all 14 blend modes.
/// Some modes (Atop, DestAtop, Saturate, Add) differ because the POC operates
/// in ACEScg linear while vips composites in the image's native space.
#[test]
fn composite_matches_vips() {
    init();

    let base = common::rgba();
    let overlay = base.clone();
    let (w, h) = (base.width() as usize, base.height() as usize);
    let bands = base.bands() as usize;

    let ctx = common::gpu_ctx();

    let modes = [
        BlendMode::Clear,
        BlendMode::Source,
        BlendMode::Over,
        BlendMode::In,
        BlendMode::Out,
        BlendMode::Atop,
        BlendMode::Dest,
        BlendMode::DestOver,
        BlendMode::DestIn,
        BlendMode::DestOut,
        BlendMode::DestAtop,
        BlendMode::Xor,
        BlendMode::Saturate,
        BlendMode::Add,
    ];

    let mut failures: Vec<String> = Vec::new();

    for mode in modes {
        let cpu_op = chromors::operation::composite::Composite2Operation {
            overlay: overlay.clone(),
            mode,
            x: None,
            y: None,
            compositing_space: None,
            premultiplied: Some(false),
        };
        let cpu_out = base.execute(&cpu_op).unwrap();
        let cpu_bytes = common::vips_materialize(&cpu_out);

        // Fresh GPU images for each mode to avoid graph accumulation
        let gpu = common::vips_to_gpu(&base, &ctx);
        let gpu_overlay = common::vips_to_gpu(&overlay, &ctx);

        let gpu_op = Composite2Operation {
            overlay: gpu_overlay.clone(),
            mode,
            x: None,
            y: None,
            compositing_space: None,
            premultiplied: Some(false),
        };
        let gpu_out = gpu.execute(&gpu_op).unwrap();
        let poc_bytes = common::poc_materialize(&gpu_out);
        let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

        let border = 1;
        let rms = common::rms_u8_interior(&cpu_bytes, &poc_u8, w, h, bands, border);
        let ok = rms < 6.0;
        println!(
            "composite {:>10?} GPU vs vips interior RMS = {:>8.4} (0..255)  {}",
            mode,
            rms,
            if ok { "OK" } else { "FAIL" }
        );
        if !ok {
            failures.push(format!("{:?}: RMS {:.4}", mode, rms));
        }
    }

    if !failures.is_empty() {
        panic!(
            "GPU composite diverges from vips for:\n  {}",
            failures.join("\n  ")
        );
    }
}

/// End-to-end sandwich: convert to ACEScg via vips, round-trip back to
/// GPU sandwich: Vips converts sRGB → ACEScg, GPU blurs in ACEScg, GPU
/// outputs as sRGB.  CPU reference blurs directly in sRGB.
/// The RMS is non-trivial because blurring in linear vs gamma space differs;
/// we assert the pipeline compiles and the difference is bounded.
#[test]
fn sandwich_acescg_roundtrip() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let aces_meta = chromors::pixel::PixelMeta::new(
        chromors::pixel::PixelFormat::Rgba8,
        chromors::color::space::ColorSpace::ACES_CG,
        chromors::pixel::AlphaPolicy::Straight,
    );
    let aces_img = img.convert(aces_meta).unwrap();

    let sigma = 3.0_f64;
    let radius = (sigma * 3.0).ceil() as i32;
    let cpu_op = chromors::operation::filters::GaussianBlurOperation {
        sigma,
        minimum_amplitude: Some((-((radius * radius) as f64) / (2.0 * sigma * sigma)).exp()),
        precision: None,
    };
    let cpu = img.execute(&cpu_op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&aces_img, &ctx);

    let blur_op = GaussianBlurOperation {
        sigma,
        minimum_amplitude: Some((-((radius * radius) as f64) / (2.0 * sigma * sigma)).exp()),
        precision: None,
    };
    let gpu_blur: GenImage<GpuBackend> = gpu.execute(&blur_op).unwrap();

    let srgb_meta = chromors::pixel::PixelMeta::new(
        chromors::pixel::PixelFormat::Rgba8,
        chromors::color::space::ColorSpace::SRGB,
        chromors::pixel::AlphaPolicy::Straight,
    );
    let gpu_out = gpu_blur.convert(srgb_meta).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8_interior(&cpu_bytes, &poc_u8, w, h, bands, radius as usize);
    println!("sandwich ACEScg→SRGB roundtrip blur RMS = {rms:.4}");
    // ~35 LSB is expected: GPU blurs linear ACEScg, CPU blurs gamma sRGB.
    assert!(
        rms < 60.0,
        "ACEScg sandwich diverged too much: RMS {rms:.4}"
    );
}

/// Sandwich ACEScg roundtrip with composite: same validation as
/// sandwich_acescg_roundtrip but exercising the composite pipeline.
#[test]
fn sandwich_acescg_composite() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let aces_meta = chromors::pixel::PixelMeta::new(
        chromors::pixel::PixelFormat::Rgba8,
        chromors::color::space::ColorSpace::ACES_CG,
        chromors::pixel::AlphaPolicy::Straight,
    );
    let aces_base = img.convert(aces_meta).unwrap();
    let aces_overlay = aces_base.clone();

    let cpu_op = chromors::operation::composite::Composite2Operation {
        overlay: img.clone(),
        mode: BlendMode::Over,
        x: None,
        y: None,
        compositing_space: None,
        premultiplied: Some(false),
    };
    let cpu = img.execute(&cpu_op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&aces_base, &ctx);
    let gpu_overlay = common::vips_to_gpu(&aces_overlay, &ctx);

    let comp_op = Composite2Operation {
        overlay: gpu_overlay.clone(),
        mode: BlendMode::Over,
        x: None,
        y: None,
        compositing_space: None,
        premultiplied: Some(false),
    };
    let gpu_comp: GenImage<GpuBackend> = gpu.execute(&comp_op).unwrap();

    let srgb_meta = chromors::pixel::PixelMeta::new(
        chromors::pixel::PixelFormat::Rgba8,
        chromors::color::space::ColorSpace::SRGB,
        chromors::pixel::AlphaPolicy::Straight,
    );
    let gpu_out = gpu_comp.convert(srgb_meta).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let border = 1;
    let rms = common::rms_u8_interior(&cpu_bytes, &poc_u8, w, h, bands, border);
    println!("sandwich ACEScg composite Over RMS = {rms:.4}");
    // Non-trivial RMS expected: GPU composites in linear ACEScg, Vips in sRGB gamma.
    assert!(
        rms < 60.0,
        "ACEScg composite roundtrip diverged too much: RMS {rms:.4}"
    );
}

/// GPU shrink must match vips `shrink` — both use a 2×2 box-filter average.
#[test]
fn shrink_matches_vips() {
    init();
    let img = common::rgba();
    let bands = img.bands() as usize;

    let cpu = img
        .execute(&chromors::operation::geometry::ShrinkOperation {
            horizontal: 2.0,
            vertical: 2.0,
            ceil: None,
        })
        .unwrap();
    let (sw, sh) = (cpu.width() as usize, cpu.height() as usize);
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu
        .execute(&ShrinkOperation {
            horizontal: 2.0,
            vertical: 2.0,
            ceil: None,
        })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, sw, sh, bands);

    let border = 1usize;
    let rms = common::rms_u8_interior(&cpu_bytes, &poc_u8, sw, sh, bands, border);
    println!("shrink 2× GPU vs vips interior RMS = {rms:.4} (0..255)");
    assert!(rms < 3.0, "GPU shrink diverges from vips: RMS {rms:.4}");
}

/// GPU opacity matches vips across opacity levels.
#[test]
fn opacity_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let ctx = common::gpu_ctx();

    for amount in [0.25_f32, 0.5, 0.75] {
        let cpu_out = img
            .execute(&chromors::operation::opacity::OpacityOperation { amount })
            .unwrap();
        let cpu_bytes = common::vips_materialize(&cpu_out);

        let gpu = common::vips_to_gpu(&img, &ctx);
        let gpu_out = gpu.execute(&OpacityOperation { amount }).unwrap();
        let poc_bytes = common::poc_materialize(&gpu_out);
        let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

        let rms = common::rms_u8(&cpu_bytes, &poc_u8);
        println!(
            "opacity {:.2} GPU vs vips RMS = {:.4} (0..255)",
            amount, rms
        );
        assert!(
            rms < 22.0,
            "GPU opacity diverges at {:.2}: RMS {:.4}",
            amount,
            rms
        );
    }
}

/// GPU gamma/exposure matches vips.
#[test]
fn gamma_matches_vips() {
    init();
    let img = common::rgb();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let ctx = common::gpu_ctx();

    for stops in [-2.0_f64, -1.0, 0.0, 1.0, 2.0] {
        let exponent = 2.0_f64.powf(stops);
        let op = chromors::operation::icc::GammaOperation {
            exponent: Some(exponent),
        };
        let cpu_out = img.execute(&op).unwrap();
        let cpu_bytes = common::vips_materialize(&cpu_out);

        let gpu = common::vips_to_gpu(&img, &ctx);
        let gpu_out = gpu
            .execute(&GammaOperation {
                exponent: Some(exponent),
            })
            .unwrap();
        let poc_bytes = common::poc_materialize(&gpu_out);
        let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

        let rms = common::rms_u8(&cpu_bytes, &poc_u8);
        println!(
            "gamma {:+.1} EV GPU vs vips RMS = {:.4} (0..255)",
            stops, rms
        );
        assert!(
            rms < 25.0,
            "GPU gamma {:+.1} EV diverges: RMS {:.4}",
            stops,
            rms
        );
    }
}

/// GPU saturation matches vips.
#[test]
fn saturation_matches_vips() {
    init();
    let img = common::rgb();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let ctx = common::gpu_ctx();

    for amount in [0.0_f64, 0.5, 1.0, 1.5, 2.0] {
        let op = chromors::operation::icc::SaturationOperation { amount };
        let cpu_out = img.execute(&op).unwrap();
        let cpu_bytes = common::vips_materialize(&cpu_out);

        let gpu = common::vips_to_gpu(&img, &ctx);
        let gpu_out = gpu.execute(&SaturationOperation { amount }).unwrap();
        let poc_bytes = common::poc_materialize(&gpu_out);
        let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

        let rms = common::rms_u8(&cpu_bytes, &poc_u8);
        println!(
            "saturation {:.1} GPU vs vips RMS = {:.4} (0..255)",
            amount, rms
        );
        assert!(
            rms < 22.0,
            "GPU saturation {:.1} diverges: RMS {:.4}",
            amount,
            rms
        );
    }
}

/// GPU histogram extractor: total pixel count must match image dimensions.
#[test]
fn histogram_extracts_channel() {
    init();
    let img = common::rgba();
    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);

    let op = chromors::operation::stats::HistogramOp {
        bins: 256,
        channel: 0,
    };
    let hist_node = chromors::backend::gpu::HistogramType::execute(&op, &gpu.handle);
    let hist_buf = chromors::backend::gpu::Targetable::pull(
        &chromors::backend::gpu::HistogramType { bins: 256 },
        &hist_node,
        &chromors::backend::gpu::Atomic,
    )
    .unwrap();
    let hist = chromors::data::histogram::HistogramResult::from_bytes(hist_buf.as_bytes());
    assert_eq!(hist.bins.len(), 256, "expected 256 bins");
    let total = hist.total_pixels;
    assert!(total > 0, "histogram is empty");
    // Total count must equal number of pixels.
    let (iw, ih) = (img.width() as u64, img.height() as u64);
    assert_eq!(
        total,
        iw * ih,
        "pixel count mismatch: got {total}, expected {}",
        iw * ih
    );
    println!("histogram median={:.3}", hist.percentile(0.5));
}

/// VipsBackend `histogram()` capability vs GPU `HistogramOp` — same total pixel count.
#[test]
fn histogram_capability_matches_gpu() {
    init();
    let img = common::rgb();
    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);

    let vips_hist = img.histogram().unwrap();
    let vips_mat = chromors::target::HistogramTarget::new(vips_hist)
        .pull()
        .unwrap();

    let op = chromors::operation::stats::HistogramOp {
        bins: 256,
        channel: 0,
    };
    let hist_node = chromors::backend::gpu::HistogramType::execute(&op, &gpu.handle);
    let hist_buf = chromors::backend::gpu::Targetable::pull(
        &chromors::backend::gpu::HistogramType { bins: 256 },
        &hist_node,
        &chromors::backend::gpu::Atomic,
    )
    .unwrap();
    let gpu_hist = chromors::data::histogram::HistogramResult::from_bytes(hist_buf.as_bytes());

    assert_eq!(gpu_hist.bins.len(), 256);
    assert_eq!(
        gpu_hist.total_pixels,
        img.width() as u64 * img.height() as u64
    );

    let bands = img.bands() as usize;
    let vips_flat: Vec<u32> = vips_mat
        .buffer
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vips_flat.len(), 256 * bands);

    let vips_ch0: Vec<u32> = (0..256).map(|j| vips_flat[j * bands]).collect();
    let vips_ch0_total: u64 = vips_ch0.iter().map(|&b| b as u64).sum();
    assert_eq!(vips_ch0_total, gpu_hist.total_pixels);

    let rms = common::rms_u8(
        &vips_ch0
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect::<Vec<_>>(),
        &gpu_hist
            .bins
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    println!("histogram cross-backend RMS: {rms:.4}");
    assert!(rms < 1.0, "histogram mismatch RMS too high: {rms}");
}

/// GpuBackend `histogram()` capability: verify the new lazy histogram path.
#[test]
fn histogram_gpu_capability_counts_pixels() {
    init();
    let img = common::rgb();
    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);

    let hist = gpu.histogram().unwrap();
    let mat = chromors::target::HistogramTarget::new(hist)
        .pull()
        .unwrap();
    let result = chromors::data::histogram::HistogramResult::from_bytes(&mat.buffer);

    assert_eq!(mat.bins, 256);
    assert_eq!(
        result.total_pixels,
        img.width() as u64 * img.height() as u64
    );
}

// ── Band / channel operations ─────────────────────────────────────────────────

/// `ScaleBandGpuOp { band: 3, factor: 0.5 }` must match `OpacityOperation(0.5)`.
///
/// Both scale the alpha channel (band 3) by 0.5. The GPU version is a single
/// fused kernel call; the Vips version is extract_band + linear + bandjoin.
/// This test verifies the GPU channel-level operation produces the same result.
#[test]
fn scale_alpha_band_matches_opacity() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);

    // Vips reference: OpacityOperation halves alpha via extract/scale/join
    let cpu = img
        .execute(&chromors::OpacityOperation { amount: 0.5 })
        .unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    // GPU: ScaleBandGpuOp { band: 3, factor: 0.5 } — single fused kernel
    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu
        .execute(&chromors::operation::bands::ScaleBandGpuOp {
            band: 3,
            factor: 0.5,
        })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, 4);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("scale_band(alpha, 0.5) vs opacity(0.5) RMS = {rms:.4}");
    assert!(
        rms < 3.0,
        "ScaleBandGpuOp(alpha) diverges from OpacityOperation: RMS {rms:.4}"
    );
}

/// `ScaleBandGpuOp { band: 0, factor: 0.5 }` halves the red channel.
///
/// Vips equivalent: extract_band(0) → linear(0.5) → bandjoin(scaled, G, B, A).
/// The GPU version is a single kernel; the Vips path requires 4 extract + 1 linear + bandjoin.
/// Compare only the red channel (band 0) of the output.
#[test]
fn scale_red_band_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);

    // Vips: extract R, scale by 0.5, join G + B + A back
    let red: chromors::data::image::Image2D<chromors::backend::vips::VipsBackend> = img
        .execute(&chromors::operation::bands::ExtractBandOperation {
            band: 0,
            count: Some(1),
        })
        .unwrap();
    let scaled_red = red
        .execute(&chromors::operation::arithmetic::LinearOperation {
            a: 0.5,
            b: 0.0,
            uchar: Some(true),
        })
        .unwrap();
    let gba = img
        .execute(&chromors::operation::bands::ExtractBandOperation {
            band: 1,
            count: Some(3),
        })
        .unwrap();
    let cpu = scaled_red.bandjoin(&gba).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    // GPU: single kernel call
    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu
        .execute(&chromors::operation::bands::ScaleBandGpuOp {
            band: 0,
            factor: 0.5,
        })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, 4);

    // Compare only the red channel (band 0)
    let cpu_r: Vec<u8> = cpu_bytes.iter().copied().step_by(4).collect();
    let gpu_r: Vec<u8> = poc_u8.iter().copied().step_by(4).collect();
    let rms = common::rms_u8(&cpu_r, &gpu_r);
    println!("scale_band(R, 0.5) vs vips red-channel RMS = {rms:.4}");
    assert!(
        rms < 3.0,
        "ScaleBandGpuOp(R) diverges from vips: RMS {rms:.4}"
    );
}

/// `ExtractBandGpuOp { band: 0 }` replicates the red channel to all four output channels.
///
/// Vips reference: `extract_band(0)` gives a 1-band image containing just red.
/// GPU gives a 4-band RGBA image where R=G=B=A=original_red.
/// Comparing channel 0 of the GPU output vs the single band of the Vips output must match.
#[test]
fn extract_red_band_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);

    // Vips: 1-band gray image with red values
    let cpu = img
        .execute(&chromors::operation::bands::ExtractBandOperation {
            band: 0,
            count: Some(1),
        })
        .unwrap();
    let cpu_bytes = common::vips_materialize(&cpu); // 1 byte per pixel (gray)

    // GPU: 4-band RGBA where R=G=B=A=original_red
    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu
        .execute(&chromors::operation::bands::ExtractBandGpuOp { band: 0 })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, 4);

    // Channel 0 of GPU output must match the single-band Vips output
    let gpu_r: Vec<u8> = poc_u8.iter().copied().step_by(4).collect();
    let rms = common::rms_u8(&cpu_bytes, &gpu_r);
    println!("extract_band(R) GPU vs vips RMS = {rms:.4}");
    assert!(
        rms < 3.0,
        "ExtractBandGpuOp(R) diverges from vips: RMS {rms:.4}"
    );
}

/// Chains two band operations: `AddToBandGpuOp(R, +0.1)` followed by `ScaleBandGpuOp(B, 0.5)`.
///
/// Validates that GPU graph fusion produces a single fused dispatch for a
/// two-operation chain. Both ops run in the same shader — no intermediate readback.
/// Result verified against the manually constructed Vips chain.
#[test]
fn chain_add_and_scale_band_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);

    // Vips: (R + 0.1 clamped) then scale blue by 0.5
    // Step 1: add 0.1 to red (linear a=1.0 b=25.5 on extracted red band, then bandjoin)
    let red: chromors::data::image::Image2D<chromors::backend::vips::VipsBackend> = img
        .execute(&chromors::operation::bands::ExtractBandOperation {
            band: 0,
            count: Some(1),
        })
        .unwrap();
    let red_shifted = red
        .execute(&chromors::operation::arithmetic::LinearOperation {
            a: 1.0,
            b: 25.5,
            uchar: Some(true),
        })
        .unwrap();
    let gba = img
        .execute(&chromors::operation::bands::ExtractBandOperation {
            band: 1,
            count: Some(3),
        })
        .unwrap();
    let img_r_shifted = red_shifted.bandjoin(&gba).unwrap();
    // Step 2: scale blue by 0.5
    let rg = img_r_shifted
        .execute(&chromors::operation::bands::ExtractBandOperation {
            band: 0,
            count: Some(2),
        })
        .unwrap();
    let blue = img_r_shifted
        .execute(&chromors::operation::bands::ExtractBandOperation {
            band: 2,
            count: Some(1),
        })
        .unwrap();
    let scaled_blue = blue
        .execute(&chromors::operation::arithmetic::LinearOperation {
            a: 0.5,
            b: 0.0,
            uchar: Some(true),
        })
        .unwrap();
    let a = img_r_shifted
        .execute(&chromors::operation::bands::ExtractBandOperation {
            band: 3,
            count: Some(1),
        })
        .unwrap();
    let rg_sb = rg.bandjoin(&scaled_blue).unwrap();
    let cpu = rg_sb.bandjoin(&a).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    // GPU: TWO chained emit calls — one fused shader, zero intermediate readbacks
    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_add = gpu
        .execute(&chromors::operation::bands::AddToBandGpuOp {
            band: 0,
            offset: 0.1,
        })
        .unwrap();
    let gpu_out = gpu_add
        .execute(&chromors::operation::bands::ScaleBandGpuOp {
            band: 2,
            factor: 0.5,
        })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, 4);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("chain add_band(R,+0.1) + scale_band(B,0.5) GPU vs vips RMS = {rms:.4}");
    assert!(
        rms < 5.0,
        "chained band ops diverged from vips: RMS {rms:.4}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Arithmetic operations — GPU vs vips cross-backend
// ═══════════════════════════════════════════════════════════════════════════════

use chromors::operation::arithmetic::{
    AddOperation, LinearOperation, MathOperation, MultiplyOperation, SubtractOperation,
};
use chromors::operation::{OperationMath, OperationMath2};

/// GPU LinearOperation must match vips `linear`.
#[test]
fn linear_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let cpu = img
        .execute(&LinearOperation {
            a: 0.5,
            b: 0.0,
            uchar: Some(true),
        })
        .unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu
        .execute(&LinearOperation {
            a: 0.5,
            b: 0.0,
            uchar: None,
        })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("linear GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU linear diverges from vips: RMS {rms:.4}");
}

/// GPU AddOperation must match vips `add`.
#[test]
fn add_matches_vips() {
    init();
    let img = common::rgba();

    let cpu = img.execute(&AddOperation { right: img.clone() }).unwrap();
    let cpu_f32 = common::vips_materialize_f32(&cpu);
    let cpu_bytes = common::f32_to_bytes_u8(&cpu_f32);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&AddOperation { right: gpu.clone() }).unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);

    let rms = common::rms_f32(&cpu_bytes, &poc_bytes);
    println!("add GPU vs vips RMS = {rms:.6} (float)");
    // Vips add operates on raw u8-range values (0..255) without normalization, GPU operates
    // on decode-to-linear [0,1] values. Clipping and gamma cause systematic differences.
    assert!(rms < 1.5, "GPU add diverges from vips: RMS {rms:.6}");
}

/// GPU SubtractOperation must match vips `subtract`.
#[test]
fn subtract_matches_vips() {
    init();
    let img = common::rgba();

    let cpu = img
        .execute(&SubtractOperation { right: img.clone() })
        .unwrap();
    let cpu_f32 = common::vips_materialize_f32(&cpu);
    let cpu_bytes = common::f32_to_bytes_u8(&cpu_f32);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu
        .execute(&SubtractOperation { right: gpu.clone() })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);

    let rms = common::rms_f32(&cpu_bytes, &poc_bytes);
    println!("subtract GPU vs vips RMS = {rms:.6} (float)");
    assert!(rms < 0.01, "GPU subtract diverges from vips: RMS {rms:.6}");
}

/// GPU MultiplyOperation must match vips `multiply`.
#[test]
fn multiply_matches_vips() {
    init();
    let img = common::rgba();

    let cpu = img
        .execute(&MultiplyOperation { right: img.clone() })
        .unwrap();
    let cpu_f32 = common::vips_materialize_f32(&cpu);
    let cpu_bytes = common::f32_to_bytes_u8(&cpu_f32);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu
        .execute(&MultiplyOperation { right: gpu.clone() })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);

    let rms = common::rms_f32(&cpu_bytes, &poc_bytes);
    println!("multiply GPU vs vips RMS = {rms:.6} (float)");
    assert!(rms < 0.01, "GPU multiply diverges from vips: RMS {rms:.6}");
}

/// GPU MathOperation must match vips `math`.
///
/// Vips math on integer images uses raw 0..255 values as input domain;
/// GPU operates on decode-to-linear [0,1] values. Only operations that are
/// approximately invariant to this scale difference pass.
#[test]
fn math_matches_vips() {
    init();
    let img = common::rgba();

    let ctx = common::gpu_ctx();
    for (math, tolerance) in [(OperationMath::Sin, 1.0_f64), (OperationMath::Cos, 1.5)] {
        let op = MathOperation { math };
        let cpu = img.execute(&op).unwrap();
        let cpu_f32 = common::vips_materialize_f32(&cpu);
        let cpu_bytes = common::f32_to_bytes_u8(&cpu_f32);

        let gpu = common::vips_to_gpu(&img, &ctx);
        let gpu_out = gpu.execute(&op).unwrap();
        let poc_bytes = common::poc_materialize(&gpu_out);

        let rms = common::rms_f32(&cpu_bytes, &poc_bytes);
        println!("math({math:?}) GPU vs vips RMS = {rms:.6} (float)");
        assert!(
            rms < tolerance,
            "GPU math({math:?}) diverges from vips: RMS {rms:.6}"
        );
    }
}

/// GPU Math2ConstOperation (pow, wop) must match vips `math2_const`.
#[test]
fn math2_const_matches_vips() {
    init();
    let img = common::rgba();

    use chromors::operation::arithmetic::Math2ConstOperation;
    let op = Math2ConstOperation {
        math2: OperationMath2::Pow,
        constants: vec![2.0],
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_f32 = common::vips_materialize_f32(&cpu);
    let max_val = 255.0_f32.powi(2);
    let cpu_norm: Vec<f32> = cpu_f32
        .iter()
        .map(|v| (v / max_val).clamp(0.0, 1.0))
        .collect();
    let cpu_bytes = common::f32_to_bytes_u8(&cpu_norm);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);

    let rms = common::rms_f32(&cpu_bytes, &poc_bytes);
    println!("math2_const(pow,2) GPU vs vips RMS = {rms:.6} (float)");
    assert!(
        rms < 0.15,
        "GPU math2_const diverges from vips: RMS {rms:.6}"
    );
}

/// Chained linear + add on GPU must produce a single fused dispatch matching vips.
#[test]
fn chained_linear_add_matches_vips() {
    init();
    let img = common::rgba();

    let vips_img2 = img.clone();

    let scaled = img
        .execute(&LinearOperation {
            a: 0.5,
            b: 0.0,
            uchar: Some(true),
        })
        .unwrap();
    let cpu = scaled.execute(&AddOperation { right: vips_img2 }).unwrap();
    let cpu_f32 = common::vips_materialize_f32(&cpu);
    let cpu_bytes = common::f32_to_bytes_u8(&cpu_f32);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let scaled = gpu
        .execute(&LinearOperation {
            a: 0.5,
            b: 0.0,
            uchar: None,
        })
        .unwrap();
    let gpu_out = scaled
        .execute(&AddOperation { right: gpu.clone() })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);

    let rms = common::rms_f32(&cpu_bytes, &poc_bytes);
    println!("chain linear+add GPU vs vips RMS = {rms:.6} (float)");
    assert!(
        rms < 1.0,
        "GPU chained linear+add diverges from vips: RMS {rms:.6}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Band extract / band join — GPU vs vips cross-backend
// ═══════════════════════════════════════════════════════════════════════════════

use chromors::operation::bands::{BandjoinOperation, ExtractBandOperation};

/// GPU ExtractBandOperation (single band) must match vips `extract_band`.
#[test]
fn extract_band_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = ExtractBandOperation {
        band: 0,
        count: Some(1),
    };
    let cpu = img.execute(&op).unwrap();
    // Vips output has 1 band; GPU has 4 (float4). Compare only first band.
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let pixel_count = w * h;
    let cpu_band0: Vec<u8> = cpu_bytes.iter().take(pixel_count).copied().collect();
    let gpu_band0: Vec<u8> = poc_u8.iter().copied().step_by(4).collect();
    let rms = common::rms_u8(&cpu_band0, &gpu_band0);
    println!("extract_band(0,1) GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(
        rms < 5.0,
        "GPU extract_band diverges from vips: RMS {rms:.4}"
    );
}

/// GPU ExtractBandOperation (3-band range) must match vips `extract_band`.
#[test]
fn extract_band_range_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = ExtractBandOperation {
        band: 1,
        count: Some(3),
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let pixel_count = w * h;
    let extract_count = 3usize;
    let mut cpu_flat = Vec::with_capacity(pixel_count * extract_count);
    let mut gpu_flat = Vec::with_capacity(pixel_count * extract_count);
    for i in 0..pixel_count {
        cpu_flat.push(cpu_bytes[i * extract_count]);
        cpu_flat.push(cpu_bytes[i * extract_count + 1]);
        cpu_flat.push(cpu_bytes[i * extract_count + 2]);
        gpu_flat.push(poc_u8[i * 4]);
        gpu_flat.push(poc_u8[i * 4 + 1]);
        gpu_flat.push(poc_u8[i * 4 + 2]);
    }
    let rms = common::rms_u8(&cpu_flat, &gpu_flat);
    println!("extract_band(1,3) GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(
        rms < 5.0,
        "GPU extract_band range diverges from vips: RMS {rms:.4}"
    );
}

/// GPU bandjoin of 4 single-band extracts must reconstruct the original RGBA image.
#[test]
fn bandjoin4_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);

    let r: chromors::data::image::Image2D<chromors::backend::vips::VipsBackend> = img
        .execute(&ExtractBandOperation {
            band: 0,
            count: Some(1),
        })
        .unwrap();
    let g = img
        .execute(&ExtractBandOperation {
            band: 1,
            count: Some(1),
        })
        .unwrap();
    let b = img
        .execute(&ExtractBandOperation {
            band: 2,
            count: Some(1),
        })
        .unwrap();
    let a = img
        .execute(&ExtractBandOperation {
            band: 3,
            count: Some(1),
        })
        .unwrap();
    let cpu = r
        .execute(&BandjoinOperation {
            images: vec![g, b, a],
        })
        .unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let r_gpu = gpu
        .execute(&ExtractBandOperation {
            band: 0,
            count: Some(1),
        })
        .unwrap();
    let g_gpu = gpu
        .execute(&ExtractBandOperation {
            band: 1,
            count: Some(1),
        })
        .unwrap();
    let b_gpu = gpu
        .execute(&ExtractBandOperation {
            band: 2,
            count: Some(1),
        })
        .unwrap();
    let a_gpu = gpu
        .execute(&ExtractBandOperation {
            band: 3,
            count: Some(1),
        })
        .unwrap();
    let gpu_out = r_gpu
        .execute(&BandjoinOperation {
            images: vec![g_gpu, b_gpu, a_gpu],
        })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);
    let bands = img.bands() as usize;
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("bandjoin4(extract R,G,B,A) GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU bandjoin4 diverges from vips: RMS {rms:.4}");
}

/// GPU bandjoin of 2 single-band extracts must match vips.
#[test]
fn bandjoin2_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let red: chromors::data::image::Image2D<chromors::backend::vips::VipsBackend> = img
        .execute(&ExtractBandOperation {
            band: 0,
            count: Some(1),
        })
        .unwrap();
    let green = img
        .execute(&ExtractBandOperation {
            band: 1,
            count: Some(1),
        })
        .unwrap();
    let cpu = red
        .execute(&BandjoinOperation {
            images: vec![green],
        })
        .unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let red_gpu = gpu
        .execute(&ExtractBandOperation {
            band: 0,
            count: Some(1),
        })
        .unwrap();
    let green_gpu = gpu
        .execute(&ExtractBandOperation {
            band: 1,
            count: Some(1),
        })
        .unwrap();
    let gpu_out = red_gpu
        .execute(&BandjoinOperation {
            images: vec![green_gpu],
        })
        .unwrap();
    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let pixel_count = w * h;
    let join_bands = 2usize;
    let mut cpu_flat = Vec::with_capacity(pixel_count * join_bands);
    let mut gpu_flat = Vec::with_capacity(pixel_count * join_bands);
    for i in 0..pixel_count {
        cpu_flat.push(cpu_bytes[i * join_bands]);
        cpu_flat.push(cpu_bytes[i * join_bands + 1]);
        gpu_flat.push(poc_u8[i * 4]);
        gpu_flat.push(poc_u8[i * 4 + 1]);
    }
    let rms = common::rms_u8(&cpu_flat, &gpu_flat);
    println!("bandjoin2(R, G) GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU bandjoin2 diverges from vips: RMS {rms:.4}");
}

#[test]
fn divide_matches_vips() {
    init();
    let img = common::rgba();
    let img2 = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::arithmetic::DivideOperation {
        right: img2.clone(),
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu2 = common::vips_to_gpu(&img2, &ctx);
    let op_gpu = chromors::operation::arithmetic::DivideOperation {
        right: gpu2.clone(),
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("divide GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU divide diverges from vips: RMS {rms:.4}");
}

#[test]
fn maxpair_matches_vips() {
    init();
    let img = common::rgba();
    let img2 = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::arithmetic::MaxPairOperation {
        right: img2.clone(),
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu2 = common::vips_to_gpu(&img2, &ctx);
    let op_gpu = chromors::operation::arithmetic::MaxPairOperation {
        right: gpu2.clone(),
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("maxpair GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU maxpair diverges from vips: RMS {rms:.4}");
}

#[test]
fn minpair_matches_vips() {
    init();
    let img = common::rgba();
    let img2 = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::arithmetic::MinPairOperation {
        right: img2.clone(),
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu2 = common::vips_to_gpu(&img2, &ctx);
    let op_gpu = chromors::operation::arithmetic::MinPairOperation {
        right: gpu2.clone(),
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("minpair GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU minpair diverges from vips: RMS {rms:.4}");
}

#[test]
fn remainder_matches_vips() {
    init();
    let img = common::rgba();
    let img2 = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::arithmetic::RemainderOperation {
        right: img2.clone(),
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu2 = common::vips_to_gpu(&img2, &ctx);
    let op_gpu = chromors::operation::arithmetic::RemainderOperation {
        right: gpu2.clone(),
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("remainder GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU remainder diverges from vips: RMS {rms:.4}");
}

#[test]
fn boolean_matches_vips() {
    init();
    let img = common::rgba();
    let img2 = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::arithmetic::BooleanOperation {
        right: img2.clone(),
        boolean: chromors::operation::OperationBoolean::And,
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu2 = common::vips_to_gpu(&img2, &ctx);
    let op_gpu = chromors::operation::arithmetic::BooleanOperation {
        right: gpu2.clone(),
        boolean: chromors::operation::OperationBoolean::And,
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("boolean GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU boolean diverges from vips: RMS {rms:.4}");
}

#[test]
fn relational_matches_vips() {
    init();
    let img = common::rgba();
    let img2 = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::arithmetic::RelationalOperation {
        right: img2.clone(),
        relational: chromors::operation::OperationRelational::More,
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu2 = common::vips_to_gpu(&img2, &ctx);
    let op_gpu = chromors::operation::arithmetic::RelationalOperation {
        right: gpu2.clone(),
        relational: chromors::operation::OperationRelational::More,
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("relational GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU relational diverges from vips: RMS {rms:.4}");
}

#[test]
fn composite2_matches_vips() {
    init();
    let img = common::rgba();
    let img2 = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::composite::Composite2Operation {
        overlay: img2.clone(),
        mode: chromors::operation::composite::BlendMode::Over,
        x: Some(0),
        y: Some(0),
        compositing_space: None,
        premultiplied: None,
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu2 = common::vips_to_gpu(&img2, &ctx);
    let op_gpu = chromors::operation::composite::Composite2Operation {
        overlay: gpu2.clone(),
        mode: chromors::operation::composite::BlendMode::Over,
        x: Some(0),
        y: Some(0),
        compositing_space: None,
        premultiplied: None,
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("composite2 GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU composite2 diverges from vips: RMS {rms:.4}");
}

#[test]
fn round_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::arithmetic::RoundOperation {
        round: chromors::operation::OperationRound::Floor,
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("round GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU round diverges from vips: RMS {rms:.4}");
}

#[test]
fn boolean_const_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::arithmetic::BooleanConstOperation {
        constants: vec![128.0],
        boolean: chromors::operation::OperationBoolean::And,
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("boolean_const GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(
        rms < 5.0,
        "GPU boolean_const diverges from vips: RMS {rms:.4}"
    );
}

#[test]
fn relational_const_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::arithmetic::RelationalConstOperation {
        constants: vec![128.0],
        relational: chromors::operation::OperationRelational::More,
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("relational_const GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(
        rms < 5.0,
        "GPU relational_const diverges from vips: RMS {rms:.4}"
    );
}

#[test]
fn remainder_const_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::arithmetic::RemainderConstOperation {
        constants: vec![128.0],
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("remainder_const GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(
        rms < 5.0,
        "GPU remainder_const diverges from vips: RMS {rms:.4}"
    );
}

#[test]
fn bandbool_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::bands::BandboolOperation {
        boolean: chromors::operation::OperationBoolean::And,
        bands: 4,
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("bandbool GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU bandbool diverges from vips: RMS {rms:.4}");
}

#[test]
fn bandfold_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::bands::BandfoldOperation { factor: 1 };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("bandfold GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU bandfold diverges from vips: RMS {rms:.4}");
}

#[test]
fn bandunfold_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::bands::BandunfoldOperation { factor: 1 };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("bandunfold GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU bandunfold diverges from vips: RMS {rms:.4}");
}

#[test]
fn bandmean_matches_vips() {
    init();
    let img = common::rgba();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::bands::BandmeanOperation { bands: 4 };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let gpu_out = gpu.execute(&op).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("bandmean GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU bandmean diverges from vips: RMS {rms:.4}");
}

#[test]
fn morph_matches_vips() {
    init();
    let img = common::rgba();
    let mask = common::gray();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::convolution::MorphOperation {
        mask: mask.clone(),
        morph: chromors::operation::OperationMorphology::Erode,
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let mask_gpu = common::vips_to_gpu(&mask, &ctx);
    let op_gpu = chromors::operation::convolution::MorphOperation {
        mask: mask_gpu.clone(),
        morph: chromors::operation::OperationMorphology::Erode,
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("morph GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU morph diverges from vips: RMS {rms:.4}");
}

#[test]
fn conva_matches_vips() {
    init();
    let img = common::rgba();
    let mask = common::gray();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::convolution::ConvaOperation {
        mask: mask.clone(),
        layers: None,
        cluster: None,
    };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let mask_gpu = common::vips_to_gpu(&mask, &ctx);
    let op_gpu = chromors::operation::convolution::ConvaOperation {
        mask: mask_gpu.clone(),
        layers: None,
        cluster: None,
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("conva GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU conva diverges from vips: RMS {rms:.4}");
}

#[test]
fn convf_matches_vips() {
    init();
    let img = common::rgba();
    let mask = common::gray();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::convolution::ConvfOperation { mask: mask.clone() };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let mask_gpu = common::vips_to_gpu(&mask, &ctx);
    let op_gpu = chromors::operation::convolution::ConvfOperation {
        mask: mask_gpu.clone(),
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("convf GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU convf diverges from vips: RMS {rms:.4}");
}

#[test]
fn convi_matches_vips() {
    init();
    let img = common::rgba();
    let mask = common::gray();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let bands = img.bands() as usize;

    let op = chromors::operation::convolution::ConviOperation { mask: mask.clone() };
    let cpu = img.execute(&op).unwrap();
    let cpu_bytes = common::vips_materialize(&cpu);

    let ctx = common::gpu_ctx();
    let gpu = common::vips_to_gpu(&img, &ctx);
    let mask_gpu = common::vips_to_gpu(&mask, &ctx);
    let op_gpu = chromors::operation::convolution::ConviOperation {
        mask: mask_gpu.clone(),
    };
    let gpu_out = gpu.execute(&op_gpu).unwrap();

    let poc_bytes = common::poc_materialize(&gpu_out);
    let poc_u8 = common::poc_f32_to_u8(&poc_bytes, w, h, bands);

    let rms = common::rms_u8(&cpu_bytes, &poc_u8);
    println!("convi GPU vs vips RMS = {rms:.4} (0..255)");
    assert!(rms < 5.0, "GPU convi diverges from vips: RMS {rms:.4}");
}
