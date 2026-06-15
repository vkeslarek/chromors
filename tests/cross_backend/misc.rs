use super::*;
use poc::{OperationBoolean, OperationRelational};

/// `GpuContext::from_device` (the windowed-viewer shared-device path) must
/// produce a context the DAG can materialize through just like
/// `GpuContext::new`, and `GpuBufferTarget` (the viewport tile-fetch exit)
/// must hand back a still-resident `Arc<GpuBuffer>` whose bytes match
/// `RamImageTarget`'s download of the same region.
#[test]
fn gpu_context_from_device_and_buffer_target() {
    use poc::backend::gpu::context::GpuContext;
    use poc::data::image::GpuBufferTarget;
    use poc::io::Target;
    use poc::work_unit::{Lod, Region};
    use std::sync::Arc;

    let _g = common::vips_serial();

    // Build a standalone device/queue the way a windowed app would, then
    // wrap it via `from_device` instead of `GpuContext::new`.
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("GPU adapter required for GPU tests");
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("from_device test"),
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .expect("GPU device required for GPU tests");

    let ctx = GpuContext::from_device(Arc::new(device), Arc::new(queue), &limits);

    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);
    // A bare source root can't be pulled directly (pre-existing
    // `emit.rs` "fused pass needs an output" limitation, unrelated to this
    // test) — apply a no-op `Convert` so the DAG has an operation node.
    let gpu_img = gpu_img.cast_storage(poc::pixel::Storage::U8, None);
    let (w, h) = (gpu_img.width(), gpu_img.height());
    let region = Region {
        x: 0,
        y: 0,
        w: w as i32,
        h: h as i32,
        lod: Lod(0),
    };

    let gpu_buf = gpu_img.pull(&GpuBufferTarget, region.clone()).unwrap();
    let extracted = gpu_buf.read_to_cpu(&ctx).unwrap();
    let downloaded = gpu_img
        .pull(&poc::data::image::RamImageTarget, region)
        .unwrap();

    assert_eq!(extracted, downloaded);
}

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

#[test]
fn copy_matches_vips() {
    // `copy` is a pixel-identity passthrough (metadata-only). The GPU lowers it
    // to a zero-cost `forward()` alias, so the output must be byte-identical to
    // the input (and to the vips `copy`), modulo storage round-trip.
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    // A metadata-only copy: tweak resolution + offset, leave pixels untouched.
    let vips_res = vips_img.copy(
        None,
        None,
        None,
        None,
        None,
        Some(72.0),
        Some(72.0),
        Some(0),
        Some(0),
    );
    let gpu_res = gpu_img.copy(
        None,
        None,
        None,
        None,
        None,
        Some(72.0),
        Some(72.0),
        Some(0),
        Some(0),
    );

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("copy RMS = {}", rms);
    assert!(
        rms < 1.0,
        "copy must be a pixel-identity passthrough: {}",
        rms
    );
}
