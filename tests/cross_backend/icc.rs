use super::*;

#[test]
fn gamma_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let exponent = 2.2;
    let vips_res = vips_img.gamma(Some(exponent));
    let gpu_res = gpu_img.gamma(Some(1.0 / exponent));

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("gamma RMS = {}", rms);
    assert!(rms < 5.0, "gamma diff too high: {}", rms);
}
