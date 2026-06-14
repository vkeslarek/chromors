use super::*;
use poc::{OperationBoolean, OperationRelational};

#[test]
fn saturation_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let amount = 0.5f32;
    let vips_res = vips_img.saturation(amount);
    let gpu_res = gpu_img.saturation(amount);

    let vips_norm = vips_materialize_linear_f32_norm(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);
    let gpu_norm: Vec<f32> = gpu_bytes.iter().map(|&b| b as f32 / 255.0).collect();

    let rms = common::rms_f32(
        bytemuck::cast_slice(&vips_norm),
        bytemuck::cast_slice(&gpu_norm),
    );
    println!("saturation RMS = {}", rms);
    assert!(rms < 0.05, "saturation diff too high: {}", rms);
}

#[test]
fn boolean_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_a = common::rgb();
    let vips_b = common::rgb_pattern();
    let gpu_a = common::vips_to_gpu(&vips_a, &ctx);
    let gpu_b = common::vips_to_gpu(&vips_b, &ctx);

    let vips_res = vips_a.boolean(&vips_b, OperationBoolean::And);
    let gpu_res = gpu_a.boolean(&gpu_b, OperationBoolean::And);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("boolean RMS = {}", rms);
    assert!(rms < 5.0, "boolean diff too high: {}", rms);
}

#[test]
fn relational_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_a = common::rgb();
    let vips_b = common::rgb_pattern();
    let gpu_a = common::vips_to_gpu(&vips_a, &ctx);
    let gpu_b = common::vips_to_gpu(&vips_b, &ctx);

    let vips_res = vips_a.relational(&vips_b, OperationRelational::Equal);
    let gpu_res = gpu_a.relational(&gpu_b, OperationRelational::Equal);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("relational RMS = {}", rms);
    assert!(rms < 5.0, "relational diff too high: {}", rms);
}

#[test]
fn boolean_const_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.boolean_const(OperationBoolean::And, vec![128.0]);
    let gpu_res = gpu_img.boolean_const(OperationBoolean::And, vec![128.0]);

    let vips_f32 = common::vips_materialize_f32(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let vips_u8: Vec<u8> = vips_f32
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();

    assert_eq!(vips_u8.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("boolean_const RMS = {}", rms);
    assert!(rms < 5.0, "boolean_const diff too high: {}", rms);
}

#[test]
fn relational_const_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.relational_const(OperationRelational::Equal, vec![128.0]);
    let gpu_res = gpu_img.relational_const(OperationRelational::Equal, vec![128.0]);

    let vips_f32 = common::vips_materialize_f32(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let vips_u8: Vec<u8> = vips_f32
        .iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect();

    assert_eq!(vips_u8.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("relational_const RMS = {}", rms);
    assert!(rms < 5.0, "relational_const diff too high: {}", rms);
}

#[test]
#[ignore = "Lut<VipsBackend> has no constant-source constructor yet"]
fn maplut_matches_vips() {}

#[test]
fn recomb_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let bands = vips_img.layout().channel_count() as i32;

    let vips_matrix = Mask2D::<VipsBackend>::identity(bands);
    let gpu_matrix = Mask2D::<GpuBackend>::identity(ctx.clone(), bands);

    let vips_res = vips_img.recomb(vips_matrix.as_input());
    let gpu_res = gpu_img.recomb(gpu_matrix.as_input());

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("recomb RMS = {}", rms);

    assert!(rms < 5.0, "recomb diff too high: {}", rms);
}

#[test]
fn case_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_case0 = vips_img.invert();
    let gpu_case0 = gpu_img.invert();

    let vips_res = vips_img.case(vec![vips_case0.as_input()]);
    let gpu_res = gpu_img.case(vec![gpu_case0.as_input()]);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("case RMS = {}", rms);
    assert!(rms < 5.0, "case diff too high: {}", rms);
}

#[test]
fn ifthenelse_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.ifthenelse(vips_img.as_input(), vips_img.as_input(), None);
    let gpu_res = gpu_img.ifthenelse(gpu_img.as_input(), gpu_img.as_input(), None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("ifthenelse RMS = {}", rms);
    assert!(rms < 10.0, "ifthenelse diff too high: {}", rms);
}

// ── Misc tone ops ────────────────────────────────────────────────────────────

#[test]
fn exposure_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.exposure(1.0, 0.0);
    let gpu_res = gpu_img.exposure(1.0, 0.0);

    // vips `linear` promotes to float (gain can exceed 255); GPU output stays u8.
    let vips_f32 = common::vips_materialize_f32(&vips_res);
    let vips_u8: Vec<u8> = vips_f32
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect();
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("exposure RMS = {}", rms);
    assert!(rms < 20.0, "exposure diff too high: {}", rms);
}

#[test]
fn brightness_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.brightness(0.1);
    let gpu_res = gpu_img.brightness(0.1);

    // vips `linear` promotes to float; GPU output stays u8.
    let vips_f32 = common::vips_materialize_f32(&vips_res);
    let vips_u8: Vec<u8> = vips_f32
        .iter()
        .map(|&v| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8)
        .collect();
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_u8, &gpu_bytes);
    println!("brightness RMS = {}", rms);
    assert!(rms < 20.0, "brightness diff too high: {}", rms);
}
