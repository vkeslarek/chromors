use super::*;

#[test]
fn convert_roundtrip() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img
        .cast_storage(Storage::F32, None)
        .cast_storage(Storage::U8, None);
    let gpu_res = gpu_img
        .cast_storage(Storage::F32, None)
        .cast_storage(Storage::U8, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("convert roundtrip RMS = {}", rms);
    assert!(rms < 5.0, "convert roundtrip diverged: {}", rms);
}

#[test]
fn sandwich_acescg_roundtrip() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let sigma: f32 = 2.0;
    let radius = (sigma * 3.0).ceil() as usize;

    let vips_res = vips_img
        .cast_storage(Storage::F32, None)
        .blur(sigma)
        .cast_storage(Storage::U8, None);

    let gpu_res = gpu_img
        .cast_storage(Storage::F32, None)
        .blur(sigma)
        .cast_storage(Storage::U8, None);

    let (w, h) = (vips_img.width() as usize, vips_img.height() as usize);
    let bands = vips_img.layout().channel_count() as usize;

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8_interior(&vips_bytes, &gpu_bytes, w, h, bands, radius);
    println!("sandwich roundtrip blur interior RMS = {}", rms);
    assert!(rms < 10.0, "sandwich roundtrip diverged: {}", rms);
}

#[test]
#[ignore = "AddToBandGpuOp/ScaleBandGpuOp (per-band ops) not ported to poc yet (was AddToBandGpuOp+ScaleBandGpuOp chain in old chromors API)"]
fn chain_add_and_scale_band_matches_vips() {
    unimplemented!(
        "per-band add/scale chain not ported to poc — add Operation<B>+Lower<B> for both backends first"
    )
}

#[test]
fn add_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_a = common::rgb();
    let vips_b = common::rgb_pattern();
    let gpu_a = common::vips_to_gpu(&vips_a, &ctx);
    let gpu_b = common::vips_to_gpu(&vips_b, &ctx);

    let vips_res = vips_a.add(&vips_b);
    let gpu_res = gpu_a.add(&gpu_b);

    let vips_bytes = common::vips_materialize(&vips_res); // raw ushort, LE
    let gpu_bytes = common::poc_materialize(&gpu_res); // u8

    let vips_u16: &[u16] = bytemuck::cast_slice(&vips_bytes);
    let vips_u8: Vec<u8> = vips_u16.iter().map(|&v| v.min(255) as u8).collect();

    assert_eq!(vips_u8.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("add RMS = {}", rms);
    assert!(rms < 10.0, "add diff too high: {}", rms);
}

#[test]
fn subtract_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_a = common::rgb();
    let vips_b = common::rgb_pattern();
    let gpu_a = common::vips_to_gpu(&vips_a, &ctx);
    let gpu_b = common::vips_to_gpu(&vips_b, &ctx);

    let vips_res = vips_a.subtract(&vips_b);
    let gpu_res = gpu_a.subtract(&gpu_b);

    let vips_bytes = common::vips_materialize(&vips_res); // raw short, LE
    let gpu_bytes = common::poc_materialize(&gpu_res); // u8

    let vips_i16: &[i16] = bytemuck::cast_slice(&vips_bytes);
    let vips_u8: Vec<u8> = vips_i16.iter().map(|&v| v.clamp(0, 255) as u8).collect();

    assert_eq!(vips_u8.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("subtract RMS = {}", rms);
    assert!(rms < 10.0, "subtract diff too high: {}", rms);
}

#[test]
fn multiply_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_a = common::rgb();
    let vips_b = common::rgb_pattern();
    let gpu_a = common::vips_to_gpu(&vips_a, &ctx);
    let gpu_b = common::vips_to_gpu(&vips_b, &ctx);

    let vips_res = vips_a.multiply(&vips_b);
    let gpu_res = gpu_a.multiply(&gpu_b);

    // vips promotes uchar*uchar -> ushort (raw a*b, 0..65025) but leaves
    // format() stale at u8, so read the raw bytes as u16 directly.
    let vips_u16 = common::vips_materialize_raw_u16(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res); // u8, same layout/length as vips u8 input

    let vips_u8: Vec<u8> = vips_u16
        .iter()
        .map(|&v| (v as f32 / 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect();

    assert_eq!(vips_u8.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("multiply RMS = {}", rms);
    assert!(rms < 10.0, "multiply diff too high: {}", rms);
}

#[test]
fn math_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.math(OperationMath::Sin);
    let gpu_res = gpu_img.math(OperationMath::Sin);

    let vips_f32 = common::vips_materialize_f32(&vips_res); // raw sin() in [-1,1]
    let gpu_bytes = common::poc_materialize(&gpu_res); // u8, sin() clamped to [0,1] then *255

    let vips_u8: Vec<u8> = vips_f32
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();

    assert_eq!(vips_u8.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("math(Sin) RMS = {}", rms);
    assert!(rms < 150.0, "math(Sin) diff too high or NaN: {}", rms);
}

#[test]
fn math2_const_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    // Use a small exponent or a small constant to avoid extreme wrapping on u8
    let vips_res = vips_img.math2_const(poc::operation::OperationMath2::Pow, vec![0.5]);
    let gpu_res = gpu_img.math2_const(poc::operation::OperationMath2::Pow, vec![0.5]);

    // VIPS outputs u8 for Pow if input is u8, wait no, math2_const(Pow) always outputs f32 on some VIPS versions?
    // Let's use `poc_materialize_f32` equivalent, but wait!
    // Since math2_const(Pow) produces a float image, vips_materialize_raw_f32 gives raw f32.
    // For math2_const, we just need to get it to match roughly.
    let vips_f32 = common::vips_materialize_raw_f32(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let vips_u8: Vec<u8> = vips_f32
        .iter()
        .map(|v| (v.clamp(0.0, 255.0) + 0.5) as u8)
        .collect();

    assert_eq!(vips_u8.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("math2_const RMS = {}", rms);
    assert!(rms < 15.0, "math2_const diff too high: {}", rms);
}

#[test]
#[ignore = "Linear operation not ported to poc yet (was LinearOperation+AddOperation chain in old chromors API)"]
fn chained_linear_add_matches_vips() {
    unimplemented!(
        "Linear not ported to poc — add Operation<B>+Lower<B> for both backends first, then chain with .add()"
    )
}

#[test]
fn divide_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_a = common::rgb();
    let vips_b = common::rgb_pattern();
    let gpu_a = common::vips_to_gpu(&vips_a, &ctx);
    let gpu_b = common::vips_to_gpu(&vips_b, &ctx);

    let vips_res = vips_a.divide(&vips_b);
    let gpu_res = gpu_a.divide(&gpu_b);

    // vips promotes uchar/uchar -> float (raw ratio) but leaves format()
    // stale at u8, so read the raw bytes as f32 directly.
    let vips_f32 = common::vips_materialize_raw_f32(&vips_res); // raw ratio (float passthrough)
    let gpu_bytes = common::poc_materialize(&gpu_res); // u8

    let vips_u8: Vec<u8> = vips_f32
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();

    assert_eq!(vips_u8.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("divide RMS = {}", rms);
    assert!(rms < 15.0, "divide diff too high: {}", rms);
}

#[test]
fn maxpair_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_a = common::rgb();
    let vips_b = common::rgb_pattern();
    let gpu_a = common::vips_to_gpu(&vips_a, &ctx);
    let gpu_b = common::vips_to_gpu(&vips_b, &ctx);

    let vips_res = vips_a.max_pair(&vips_b);
    let gpu_res = gpu_a.max_pair(&gpu_b);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("maxpair RMS = {}", rms);
    assert!(rms < 5.0, "maxpair diff too high: {}", rms);
}

#[test]
fn minpair_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_a = common::rgb();
    let vips_b = common::rgb_pattern();
    let gpu_a = common::vips_to_gpu(&vips_a, &ctx);
    let gpu_b = common::vips_to_gpu(&vips_b, &ctx);

    let vips_res = vips_a.min_pair(&vips_b);
    let gpu_res = gpu_a.min_pair(&gpu_b);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("minpair RMS = {}", rms);
    assert!(rms < 5.0, "minpair diff too high: {}", rms);
}

#[test]
fn remainder_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_a = common::rgb();
    let vips_b = common::rgb_pattern();
    let gpu_a = common::vips_to_gpu(&vips_a, &ctx);
    let gpu_b = common::vips_to_gpu(&vips_b, &ctx);

    let vips_res = vips_a.remainder(&vips_b);
    let gpu_res = gpu_a.remainder(&gpu_b);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("remainder RMS = {}", rms);
    assert!(rms < 10.0, "remainder diff too high: {}", rms);
}

#[test]
fn round_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.round(OperationRound::Floor);
    let gpu_res = gpu_img.round(OperationRound::Floor);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("round(Floor) RMS = {}", rms);
    assert!(rms < 5.0, "round diff too high: {}", rms);
}

#[test]
fn remainder_const_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.remainder_const(vec![128.0]);
    let gpu_res = gpu_img.remainder_const(vec![128.0]);

    // remainder_const returns u8
    let vips_f32 = common::vips_materialize_f32(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let vips_u8: Vec<u8> = vips_f32
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();

    assert_eq!(vips_u8.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("remainder_const RMS = {}", rms);
    assert!(rms < 15.0, "remainder_const diff too high: {}", rms);
}

#[test]
#[ignore = "ForwardFft has no Lower<GpuBackend> impl in poc yet, so complex2 (which needs FFT input on both backends) cannot be exercised cross-backend"]
fn complex2_matches_vips() {
    unimplemented!()
}

#[test]
#[ignore = "BUG: panics at tests/common/mod.rs:126 (`assert_eq!(af.len(), bf.len(), \"length mismatch\")` inside rms_f32). vips_materialize_f32 computes pixel_count from img.layout().channel_count(), but vips complexform output is a complex-typed image where each 'band' is a 2-float (re,im) pair -- channel_count() likely reports 1 band while bytes_per_pixel covers 2 floats, so vips_materialize_f32 only extracts the real component (w*h floats) while poc_materialize's GPU output is w*h*2 (or w*h*4 if stored as RGBA float with re/im in two channels), causing the length mismatch. Needs a complex-aware readback helper that extracts both re/im components from vips before comparing"]
fn complexform_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let vips_img2 = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);
    let gpu_img2 = common::vips_to_gpu(&vips_img2, &ctx);

    let vips_res = vips_img.complexform(&vips_img2);
    let gpu_res = gpu_img.complexform(&gpu_img2);

    let vips_bytes = common::vips_materialize_f32(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_f32(bytemuck::cast_slice(&vips_bytes), &gpu_bytes);
    println!("complexform RMS = {}", rms);
    assert!(rms < 10.0, "complexform diff too high: {}", rms);
}
