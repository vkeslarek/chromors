use super::*;

#[test]
fn sobel_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.sobel();
    let gpu_res = gpu_img.sobel();

    let vips_bytes = common::vips_materialize_f32(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_f32(bytemuck::cast_slice(&vips_bytes), &gpu_bytes);
    println!("sobel RMS = {}", rms);
    assert!(rms < 10.0, "sobel diff too high: {}", rms);
}

#[test]
fn prewitt_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.prewitt();
    let gpu_res = gpu_img.prewitt();

    let vips_bytes = common::vips_materialize_f32(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_f32(bytemuck::cast_slice(&vips_bytes), &gpu_bytes);
    println!("prewitt RMS = {}", rms);
    assert!(rms < 10.0, "prewitt diff too high: {}", rms);
}

#[test]
fn scharr_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.scharr();
    let gpu_res = gpu_img.scharr();

    let vips_bytes = common::vips_materialize_f32(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_f32(bytemuck::cast_slice(&vips_bytes), &gpu_bytes);
    println!("scharr RMS = {}", rms);
    assert!(rms < 10.0, "scharr diff too high: {}", rms);
}
#[test]
fn abs_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.abs();
    let gpu_res = gpu_img.abs();

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("abs RMS = {}", rms);
    assert!(rms < 5.0, "abs diff too high: {}", rms);
}

#[test]
fn sign_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.sign();
    let gpu_res = gpu_img.sign();

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("sign RMS = {}", rms);
    assert!(rms < 5.0, "sign diff too high: {}", rms);
}

// ── Geometry ─────────────────────────────────────────────────────────────────

#[test]
fn invertlut_matches_vips() {
    use chromors::data::lut::{Lut, RawLutTarget};
    use chromors::work_unit::Range;

    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();

    #[rustfmt::skip]
    let vals: [f32; 12] = [
        0.1, 0.2, 0.3, 0.1,
        0.2, 0.4, 0.4, 0.2,
        0.7, 0.5, 0.6, 0.3,
    ];
    let vips_lut = <Lut<VipsBackend>>::from_values(3, 4, &vals);
    let gpu_lut = <Lut<GpuBackend>>::from_values(ctx.clone(), 3, 4, &vals);

    let vips_res = vips_lut.invertlut(None);
    let gpu_res = gpu_lut.invertlut(None);

    let size = 256usize;
    let wu = Range {
        start: 0,
        end: size as i32,
    };
    let vips_bytes = vips_res.pull(&RawLutTarget, wu.clone()).unwrap();
    let gpu_bytes = gpu_res.pull(&RawLutTarget, wu).unwrap();

    let bands = 3usize;
    let mut sum_sq = 0.0f64;
    let mut n = 0usize;
    for i in 0..size {
        for b in 0..bands {
            let vips_off = (i * bands + b) * 8;
            let vips_val =
                f64::from_le_bytes(vips_bytes[vips_off..vips_off + 8].try_into().unwrap());
            let gpu_off = (i * 4 + b) * 4;
            let gpu_val =
                f32::from_le_bytes(gpu_bytes[gpu_off..gpu_off + 4].try_into().unwrap()) as f64;
            let diff = vips_val - gpu_val;
            sum_sq += diff * diff;
            n += 1;
        }
    }
    let rms = (sum_sq / n as f64).sqrt();
    println!("invertlut RMS = {}", rms);
    assert!(rms < 0.01, "invertlut diff too high: {}", rms);
}
