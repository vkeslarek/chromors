use super::*;

#[test]
fn extract_band_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.extract_band(0, None);
    let gpu_res = gpu_img.extract_band(0, None);

    // Single-band GPU output is tightly packed u8 (R8), matching vips byte-for-byte.
    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("extract_band(0) RMS = {}", rms);
    assert!(rms < 5.0, "extract_band diff too high: {}", rms);
}

#[test]
#[ignore = "Not implemented in poc"]
fn extract_band_range_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.extract_band(1, Some(3));
    let gpu_res = gpu_img.extract_band(1, Some(3));

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("extract_band(1,3) RMS = {}", rms);
    assert!(rms < 5.0, "extract_band range diff too high: {}", rms);
}

#[test]
#[ignore = "Not implemented in poc"]
fn bandjoin4_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    // vips side
    let vr = vips_img.extract_band(0, None);
    let vg = vips_img.extract_band(1, None);
    let vb = vips_img.extract_band(2, None);
    let va = vips_img.extract_band(3, None);
    let vips_res: GenImage<VipsBackend> = vr.push(poc::operation::bands::Bandjoin {
        images: vec![vr.as_input(), vg.as_input(), vb.as_input(), va.as_input()],
    });

    // gpu side
    let gr = gpu_img.extract_band(0, None);
    let gg = gpu_img.extract_band(1, None);
    let gb = gpu_img.extract_band(2, None);
    let ga = gpu_img.extract_band(3, None);
    let gpu_res: GenImage<GpuBackend> = gr.push(poc::operation::bands::Bandjoin {
        images: vec![gr.as_input(), gg.as_input(), gb.as_input(), ga.as_input()],
    });

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("bandjoin4 RMS = {}", rms);
    assert!(rms < 5.0, "bandjoin4 diff too high: {}", rms);
}

#[test]
#[ignore = "Not implemented in poc"]
fn bandjoin2_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vr = vips_img.extract_band(0, None);
    let vg = vips_img.extract_band(1, None);
    let vips_res: GenImage<VipsBackend> = vr.push(poc::operation::bands::Bandjoin {
        images: vec![vr.as_input(), vg.as_input()],
    });

    let gr = gpu_img.extract_band(0, None);
    let gg = gpu_img.extract_band(1, None);
    let gpu_res: GenImage<GpuBackend> = gr.push(poc::operation::bands::Bandjoin {
        images: vec![gr.as_input(), gg.as_input()],
    });

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("bandjoin2 RMS = {}", rms);
    assert!(rms < 5.0, "bandjoin2 diff too high: {}", rms);
}

#[test]
#[ignore = "Not implemented in poc"]
fn bandbool_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.bandbool(OperationBoolean::And, 4);
    let gpu_res = gpu_img.bandbool(OperationBoolean::And, 4);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("bandbool(And) RMS = {}", rms);
    assert!(rms < 5.0, "bandbool diff too high: {}", rms);
}

#[test]
#[ignore = "Not implemented in poc"]
fn bandfold_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.bandfold(1);
    let gpu_res = gpu_img.bandfold(1);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("bandfold RMS = {}", rms);
    assert!(rms < 5.0, "bandfold diff too high: {}", rms);
}

#[test]
#[ignore = "Not implemented in poc"]
fn bandunfold_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.bandunfold(1);
    let gpu_res = gpu_img.bandunfold(1);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("bandunfold RMS = {}", rms);
    assert!(rms < 5.0, "bandunfold diff too high: {}", rms);
}

#[test]
#[ignore = "Not implemented in poc"]
fn bandmean_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.bandmean(4);
    let gpu_res = gpu_img.bandmean(4);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("bandmean RMS = {}", rms);
    assert!(rms < 5.0, "bandmean diff too high: {}", rms);
}
use super::*;
/// GPU Gaussian blur must match vips `gaussblur` on the same linear data.
/// Interior-only — edge handling differs (vips extend vs GPU clamp).
/// Cast round trip u8 RGBA -> f32 RGBA -> u8 RGBA must be ~lossless on both
/// backends. There is no color-space `convert` op in the current API (only
/// `Cast` for pixel format); the nearest analogue to a "convert round trip"
/// is a pixel-format round trip through `RgbaF32` and back.
/// A no-op `Cast` (same format) must be lossless / near-identity.
/// GPU `composite2` matches vips `composite2` across several blend modes.
/// Sandwich: cast u8 RGBA -> f32 -> blur -> cast back to u8 RGBA, on both
/// backends. There is no ACEScg color-space convert op in the current API,
/// so this exercises the closest analogue: a pixel-format sandwich around a
/// processing op (cast to float "working" format, process, cast back).
/// Sandwich with composite: cast -> composite -> cast back, on both backends.
/// GPU shrink must match vips `shrink` — both use a 2x2 box-filter average.
/// GPU opacity matches vips across opacity levels.
/// GPU gamma matches vips `gamma`.
///
/// Vips' `gamma(exponent)` op applies `out = in_norm^(1/exponent)`; the GPU
/// `gamma_kernel` applies `out = in_norm^exponent` directly
/// (shaders/ops/gamma.slang). They compute the same curve when the GPU is
/// given the reciprocal exponent, so we pass `1/exponent` to the GPU side.
/// GPU saturation matches vips. Vips lowers `Saturation` to a manual
/// luma-mix pipeline (extract -> linear -> add -> mix) which promotes the
/// runtime pixels to a float band format (raw 0..255-range values) while
/// `format()` keeps reporting the pre-promotion u8 format — same staleness
/// as brightness/exposure, so read raw bytes as f32 directly via
/// `vips_materialize_linear_f32_norm`. The GPU saturation kernel stays in
/// the u8 codec sandwich ([0,1] normalized). Compare both sides normalized
/// to [0,1] f32.
/// GPU histogram extractor: total pixel count across all bins must match
/// width*height for a chosen channel.
/// VipsBackend histogram capability (`histogram_find`) vs GPU `HistogramOp`
/// — both must report the same total pixel count for channel 0.
/// `HistogramKind` is GPU-only (no `VipsBand`), so we cannot construct a
/// `Histogram<VipsBackend>`; instead we use vips' own `hist_find` op (which
/// is generic over `Operation<B>` via `HistogramFind` in stats.rs) and sum
/// its raw bin counts directly from the underlying VipsImage.
/// GpuBackend `histogram()` capability: each per-channel histogram's bins
/// must sum to width*height.
/// `ScaleBandGpuOp { band: 3, factor: 0.5 }` vs `OpacityOperation(0.5)`.
/// Neither a per-band-scale primitive (`ScaleBandGpuOp`) nor `OpacityOperation`
/// exist under those names in poc; `.opacity()` exists (src/operation/opacity.rs)
/// but there is no GPU per-band-scale op to compare it against.
/// `ScaleBandGpuOp { band: 0, factor: 0.5 }` halves the red channel.
/// No per-band-scale primitive exists in poc (only extract_band/bandjoin/linear-as-impl-detail).
#[test]
fn scale_red_band_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    // linear with per-band multipliers: scale red by 0.5, green by 1.0, blue by 1.0
    let vips_res = vips_img.linear(vec![0.5, 1.0, 1.0], vec![0.0]);
    let gpu_res = gpu_img.linear(vec![0.5, 1.0, 1.0], vec![0.0]);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("scale_red_band (linear) RMS = {}", rms);
    assert!(rms < 5.0, "scale_red_band diff too high: {}", rms);
}

/// `ExtractBandGpuOp { band: 0 }` replicates the red channel to all four output channels.
///
/// Vips reference: `extract_band(0)` gives a 1-band image containing just red.
/// GPU gives a 4-band RGBA image where R=G=B=A=original_red (via SwizzleView,
/// zero-cost adapter). Both vips and GPU re-encode the single-band output as
/// `Gray8` (1 byte/pixel, `output_spec()` = `with_band_count(1)`), so
/// `poc_materialize`/`vips_materialize` are directly comparable u8 buffers of
/// the same length — no f32 conversion needed.
#[test]
fn extract_red_band_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.extract_band(0, None);
    let gpu_res = gpu_img.extract_band(0, None);

    let vips_bytes = common::vips_materialize(&vips_res); // 1 band Gray8
    let gpu_bytes = common::poc_materialize(&gpu_res); // 1 band Gray8

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("extract_red_band RMS = {}", rms);
    assert!(rms < 5.0, "extract_band(0) diff too high: {}", rms);
}
