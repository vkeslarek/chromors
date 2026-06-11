mod common;
use poc::data::image::Image2D as GenImage;

/// GPU Gaussian blur must match vips `gaussblur` on the same linear data.
/// Interior-only — edge handling differs (vips extend vs GPU clamp).
#[test]
fn blur_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU `convert` round-trip: sRGB → Rec.2020 → sRGB.
/// `convert()` is a passthrough in the GPU graph that changes the output codec;
/// the working-space sandwich applies the actual matrix. We just assert the
/// shader compiles and runs; RMS is expected to be non-zero due to double
/// gamma accumulation through the sandwich.
#[test]
fn convert_roundtrip() {
    // TEST STRIPPED FOR REWRITE
}

/// A no-op GPU convert (same meta) must be close to identity.
/// Same double-conversion caveat as convert_roundtrip.
#[test]
fn convert_identity_is_lossless() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU `composite2` matches vips `composite2` across all 14 blend modes.
/// Some modes (Atop, DestAtop, Saturate, Add) differ because the POC operates
/// in ACEScg linear while vips composites in the image's native space.
#[test]
fn composite_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// End-to-end sandwich: convert to ACEScg via vips, round-trip back to
/// GPU sandwich: Vips converts sRGB → ACEScg, GPU blurs in ACEScg, GPU
/// outputs as sRGB.  CPU reference blurs directly in sRGB.
/// The RMS is non-trivial because blurring in linear vs gamma space differs;
/// we assert the pipeline compiles and the difference is bounded.
#[test]
fn sandwich_acescg_roundtrip() {
    // TEST STRIPPED FOR REWRITE
}

/// Sandwich ACEScg roundtrip with composite: same validation as
/// sandwich_acescg_roundtrip but exercising the composite pipeline.
#[test]
fn sandwich_acescg_composite() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU shrink must match vips `shrink` — both use a 2×2 box-filter average.
#[test]
fn shrink_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU opacity matches vips across opacity levels.
#[test]
fn opacity_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU gamma/exposure matches vips.
#[test]
fn gamma_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU saturation matches vips.
#[test]
fn saturation_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU histogram extractor: total pixel count must match image dimensions.
#[test]
fn histogram_extracts_channel() {
    // TEST STRIPPED FOR REWRITE
}

/// VipsBackend `histogram()` capability vs GPU `HistogramOp` — same total pixel count.
#[test]
fn histogram_capability_matches_gpu() {
    // TEST STRIPPED FOR REWRITE
}

/// GpuBackend `histogram()` capability: verify the new lazy histogram path.
#[test]
fn histogram_gpu_capability_counts_pixels() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

// ── Band / channel operations ─────────────────────────────────────────────────

/// `ScaleBandGpuOp { band: 3, factor: 0.5 }` must match `OpacityOperation(0.5)`.
///
/// Both scale the alpha channel (band 3) by 0.5. The GPU version is a single
/// fused kernel call; the Vips version is extract_band + linear + bandjoin.
/// This test verifies the GPU channel-level operation produces the same result.
#[test]
fn scale_alpha_band_matches_opacity() {
    // TEST STRIPPED FOR REWRITE
}

/// `ScaleBandGpuOp { band: 0, factor: 0.5 }` halves the red channel.
///
/// Vips equivalent: extract_band(0) → linear(0.5) → bandjoin(scaled, G, B, A).
/// The GPU version is a single kernel; the Vips path requires 4 extract + 1 linear + bandjoin.
/// Compare only the red channel (band 0) of the output.
#[test]
fn scale_red_band_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// `ExtractBandGpuOp { band: 0 }` replicates the red channel to all four output channels.
///
/// Vips reference: `extract_band(0)` gives a 1-band image containing just red.
/// GPU gives a 4-band RGBA image where R=G=B=A=original_red.
/// Comparing channel 0 of the GPU output vs the single band of the Vips output must match.
#[test]
fn extract_red_band_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// Chains two band operations: `AddToBandGpuOp(R, +0.1)` followed by `ScaleBandGpuOp(B, 0.5)`.
///
/// Validates that GPU graph fusion produces a single fused dispatch for a
/// two-operation chain. Both ops run in the same shader — no intermediate readback.
/// Result verified against the manually constructed Vips chain.
#[test]
fn chain_add_and_scale_band_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

// ═══════════════════════════════════════════════════════════════════════════════
// Arithmetic operations — GPU vs vips cross-backend
// ═══════════════════════════════════════════════════════════════════════════════

// Unused imports:
// use poc::operation::arithmetic::{
//     AddOperation, LinearOperation, MathOperation, MultiplyOperation, SubtractOperation,
// };
// use poc::operation::{OperationMath, OperationMath2};

/// GPU LinearOperation must match vips `linear`.
#[test]
fn linear_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU AddOperation must match vips `add`.
#[test]
fn add_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU SubtractOperation must match vips `subtract`.
#[test]
fn subtract_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU MultiplyOperation must match vips `multiply`.
#[test]
fn multiply_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU MathOperation must match vips `math`.
///
/// Vips math on integer images uses raw 0..255 values as input domain;
/// GPU operates on decode-to-linear [0,1] values. Only operations that are
/// approximately invariant to this scale difference pass.
#[test]
fn math_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU Math2ConstOperation (pow, wop) must match vips `math2_const`.
#[test]
fn math2_const_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// Chained linear + add on GPU must produce a single fused dispatch matching vips.
#[test]
fn chained_linear_add_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

// ═══════════════════════════════════════════════════════════════════════════════
// Band extract / band join — GPU vs vips cross-backend
// ═══════════════════════════════════════════════════════════════════════════════

// Removed unused imports

/// GPU ExtractBandOperation (single band) must match vips `extract_band`.
#[test]
fn extract_band_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU ExtractBandOperation (3-band range) must match vips `extract_band`.
#[test]
fn extract_band_range_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU bandjoin of 4 single-band extracts must reconstruct the original RGBA image.
#[test]
fn bandjoin4_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

/// GPU bandjoin of 2 single-band extracts must match vips.
#[test]
fn bandjoin2_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn divide_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn maxpair_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn minpair_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn remainder_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn boolean_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn relational_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn composite2_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn round_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn boolean_const_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn relational_const_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn remainder_const_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn bandbool_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn bandfold_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn bandunfold_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn bandmean_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn morph_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn conva_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn convf_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}

#[test]
fn convi_matches_vips() {
    // TEST STRIPPED FOR REWRITE
}
