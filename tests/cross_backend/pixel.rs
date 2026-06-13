use super::*;

#[test]
fn convert_roundtrip() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.cast(PixelFormat::RgbaF32, None).cast(PixelFormat::Rgba8, None);
    let gpu_res = gpu_img.cast(PixelFormat::RgbaF32, None).cast(PixelFormat::Rgba8, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("convert roundtrip RMS = {}", rms);
    assert!(rms < 5.0, "convert roundtrip diverged: {}", rms);
}

/// A no-op `Cast` (same format) must be lossless / near-identity.
#[test]
fn convert_identity_is_lossless() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.cast(PixelFormat::Rgba8, None);
    let gpu_res = gpu_img.cast(PixelFormat::Rgba8, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("convert identity RMS = {}", rms);
    assert!(rms < 5.0, "identity convert diverged: {}", rms);
}

/// GPU `composite2` matches vips `composite2` across several blend modes.