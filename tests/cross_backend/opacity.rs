use super::*;

#[test]
fn opacity_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    for amount in [0.5f32, 0.25f32] {
        let vips_res = vips_img.opacity(amount);
        let gpu_res = gpu_img.opacity(amount);

        let vips_bytes = common::vips_materialize(&vips_res);
        let gpu_bytes = common::poc_materialize(&gpu_res);

        let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
        println!("opacity({}) RMS = {}", amount, rms);
        assert!(rms < 5.0, "opacity({}) diff too high: {}", amount, rms);
    }
}

#[test]
#[ignore = "ScaleBandGpuOp (per-band scale) not ported to poc yet (was ScaleBandGpuOp in old chromors API)"]
fn scale_alpha_band_matches_opacity() {
    unimplemented!("per-band scale not ported to poc — add Operation<B>+Lower<B> for both backends first")
}
