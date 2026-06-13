use super::*;

#[test]
fn composite_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let base = common::rgba();
    let overlay = common::rgba();

    let modes = [BlendMode::Over, BlendMode::In, BlendMode::Xor, BlendMode::Dest];

    for mode in modes {
        let vips_res = base.push(Composite2 {
            base: base.as_input(),
            overlay: overlay.as_input(),
            mode,
            x: None,
            y: None,
            premultiplied: Some(false),
        });

        let gpu_base = common::vips_to_gpu(&base, &ctx);
        let gpu_overlay = common::vips_to_gpu(&overlay, &ctx);
        let gpu_res = gpu_base.push(Composite2 {
            base: gpu_base.as_input(),
            overlay: gpu_overlay.as_input(),
            mode,
            x: None,
            y: None,
            premultiplied: Some(false),
        });

        let vips_bytes = common::vips_materialize(&vips_res);
        let gpu_bytes = common::poc_materialize(&gpu_res);

        let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
        println!("composite {:?} RMS = {}", mode, rms);
        assert!(rms < 10.0, "composite {:?} diff too high: {}", mode, rms);
    }
}

#[test]
fn sandwich_acescg_composite() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let base = common::rgba();
    let overlay = common::rgba();

    let base_f = base.cast(PixelFormat::RgbaF32, None);
    let overlay_f = overlay.cast(PixelFormat::RgbaF32, None);
    let vips_comp = base_f.push(Composite2 {
        base: base_f.as_input(),
        overlay: overlay_f.as_input(),
        mode: BlendMode::Over,
        x: None,
        y: None,
        premultiplied: Some(false),
    });
    let vips_res = vips_comp.cast(PixelFormat::Rgba8, None);

    let gpu_base = common::vips_to_gpu(&base, &ctx);
    let gpu_overlay = common::vips_to_gpu(&overlay, &ctx);
    let gpu_base_f = gpu_base.cast(PixelFormat::RgbaF32, None);
    let gpu_overlay_f = gpu_overlay.cast(PixelFormat::RgbaF32, None);
    let gpu_comp = gpu_base_f.push(Composite2 {
        base: gpu_base_f.as_input(),
        overlay: gpu_overlay_f.as_input(),
        mode: BlendMode::Over,
        x: None,
        y: None,
        premultiplied: Some(false),
    });
    let gpu_res = gpu_comp.cast(PixelFormat::Rgba8, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("sandwich composite RMS = {}", rms);
    assert!(rms < 10.0, "sandwich composite diverged: {}", rms);
}

#[test]
fn composite2_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let vips_img2 = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);
    let gpu_img2 = common::vips_to_gpu(&vips_img2, &ctx);

    let vips_res: GenImage<VipsBackend> = vips_img.push(Composite2 {
        base: vips_img.as_input(),
        overlay: vips_img2.as_input(),
        mode: BlendMode::Over,
        x: Some(0),
        y: Some(0),
        premultiplied: None,
    });
    let gpu_res: GenImage<GpuBackend> = gpu_img.push(Composite2 {
        base: gpu_img.as_input(),
        overlay: gpu_img2.as_input(),
        mode: BlendMode::Over,
        x: Some(0),
        y: Some(0),
        premultiplied: None,
    });

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("composite2(Over) RMS = {}", rms);
    assert!(rms < 180.0, "composite2 diff too high: {}", rms);
}
