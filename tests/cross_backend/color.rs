use super::*;
use chromors::color::pipeline::convert_matrices;
use chromors::color::space::ColorSpace;
use chromors::pixel::{AlphaState, PixelLayout};

/// Reference CPU implementation of `color_convert` for the RGB(A) straight-
/// alpha, same-storage case: decode -> linear, src->XYZ(D50)->dst matrices,
/// linear -> encode. Alpha passes through unchanged (both sides `Straight`).
fn cpu_convert_rgba8(bytes: &[u8], src: PixelLayout, dst: PixelLayout) -> Vec<u8> {
    let (a, b) = convert_matrices(src, dst).expect("convert_matrices");
    let src_tf = src.color_space.transfer();
    let dst_tf = dst.color_space.transfer();
    bytes
        .chunks_exact(4)
        .flat_map(|px| {
            let lin = [
                src_tf.decode(px[0] as f32 / 255.0),
                src_tf.decode(px[1] as f32 / 255.0),
                src_tf.decode(px[2] as f32 / 255.0),
            ];
            let xyz = a.mul_vec(lin);
            let dst_lin = b.mul_vec(xyz);
            let mut out = [0u8; 4];
            for c in 0..3 {
                out[c] = (dst_tf.encode(dst_lin[c]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
            out[3] = px[3];
            out
        })
        .collect()
}

/// `Convert`'s GPU `Lower` (copy_kernel + ColorReadView, §6.1.2) must match a
/// Rust CPU reference built from the same XYZ(D50)-hub matrices
/// (`convert_matrices`) and `TransferFn` decode/encode.
#[test]
fn convert_srgb_to_p3_matches_reference() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let src_layout = gpu_img.spec.layout;
    let dst_layout = PixelLayout {
        color_space: ColorSpace::DISPLAY_P3,
        alpha: AlphaState::Straight,
        ..src_layout
    };

    let converted = gpu_img.convert(dst_layout);
    let gpu_bytes = common::poc_materialize(&converted);
    let (w, h) = (gpu_img.width() as usize, gpu_img.height() as usize);
    let gpu_u8 = common::poc_f32_to_u8(&gpu_bytes, w, h, 4);

    let src_bytes = common::vips_materialize(&vips_img);
    let reference = cpu_convert_rgba8(&src_bytes, src_layout, dst_layout);

    let rms = common::rms_u8(&gpu_u8, &reference);
    println!("srgb->p3 convert RMS = {}", rms);
    assert!(
        rms < 2.0,
        "GPU Convert diverged from CPU reference: {}",
        rms
    );
}

/// `Convert`'s vips `Lower` (§6.1.4) for a destination color space with no
/// faithful vips interpretation (Display-P3) takes the CPU custom-region
/// fallback (`cpu_convert_region`). It must agree with the GPU `Lower`
/// (`color_read_wrap`, same test target as above) — both share the same
/// `convert_matrices` + `TransferFn` XYZ(D50)-hub math.
#[test]
fn vips_cpu_convert_matches_gpu() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let src_layout = vips_img.spec.layout;
    let dst_layout = PixelLayout {
        color_space: ColorSpace::DISPLAY_P3,
        alpha: AlphaState::Straight,
        ..src_layout
    };
    assert_eq!(
        src_layout.model,
        chromors::color::model::ColorModel::Rgb,
        "source layout must be Rgb for the CPU fallback"
    );

    let vips_converted = vips_img.convert(dst_layout);
    let vips_bytes = common::vips_materialize(&vips_converted);

    let gpu_converted = gpu_img.convert(dst_layout);
    let gpu_bytes = common::poc_materialize(&gpu_converted);
    let (w, h) = (gpu_img.width() as usize, gpu_img.height() as usize);
    let gpu_u8 = common::poc_f32_to_u8(&gpu_bytes, w, h, 4);

    let rms = common::rms_u8(&vips_bytes, &gpu_u8);
    println!("vips CPU convert vs GPU convert RMS = {}", rms);
    assert!(rms < 2.0, "vips CPU Convert diverged from GPU: {}", rms);
}

/// `to_color_space`/`to_storage`/`to_model`/`linearize`
/// (`docs/native-color-management.md` §6.2) are thin `Convert` wrappers that
/// mutate exactly one `PixelLayout` axis, keeping the others fixed.
#[test]
fn ergonomic_convert_wrappers_mutate_one_axis() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);
    let base = gpu_img.spec.layout;

    let cs = gpu_img.to_color_space(ColorSpace::DISPLAY_P3);
    assert_eq!(cs.spec.layout.color_space, ColorSpace::DISPLAY_P3);
    assert_eq!(cs.spec.layout.storage, base.storage);
    assert_eq!(cs.spec.layout.model, base.model);
    assert_eq!(cs.spec.layout.alpha, base.alpha);

    let st = gpu_img.to_storage(chromors::pixel::Storage::F32);
    assert_eq!(st.spec.layout.storage, chromors::pixel::Storage::F32);
    assert_eq!(st.spec.layout.color_space, base.color_space);
    assert_eq!(st.spec.layout.model, base.model);

    let model = gpu_img.to_model(chromors::color::model::ColorModel::Gray);
    assert_eq!(
        model.spec.layout.model,
        chromors::color::model::ColorModel::Gray
    );
    assert_eq!(model.spec.layout.storage, base.storage);
    assert_eq!(model.spec.layout.color_space, base.color_space);

    let lin = gpu_img.linearize();
    assert!(lin.spec.layout.color_space.is_linear());
    assert_eq!(
        lin.spec.layout.color_space.primaries(),
        base.color_space.primaries()
    );
    assert_eq!(lin.spec.layout.storage, base.storage);
    assert_eq!(lin.spec.layout.model, base.model);

    // Cross-check linearize against GPU vs vips materialization. Both sides
    // first move to F32 storage so the comparison is over linear float
    // values, not a degenerate cast-to-u8 of a linear signal.
    let gpu_lin = gpu_img
        .to_storage(chromors::pixel::Storage::F32)
        .linearize();
    let vips_lin = vips_img
        .to_storage(chromors::pixel::Storage::F32)
        .linearize();
    let gpu_bytes = common::poc_materialize(&gpu_lin);
    let vips_f32 = common::vips_materialize_raw_f32(&vips_lin);
    let vips_bytes = common::f32_to_bytes_u8(&vips_f32);
    let rms = common::rms_f32(&vips_bytes, &gpu_bytes);
    println!("linearize RMS = {}", rms);
    assert!(rms < 0.05, "linearize diverged: {}", rms);
}
