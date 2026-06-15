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

/// A no-op `Cast` (same format) must be lossless / near-identity.
#[test]
fn convert_identity_is_lossless() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.cast_storage(Storage::U8, None);
    let gpu_res = gpu_img.cast_storage(Storage::U8, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("convert identity RMS = {}", rms);
    assert!(rms < 5.0, "identity convert diverged: {}", rms);
}

/// `FileImageSource` must detect a faithful `PixelLayout` from the vips
/// interpretation + band count on import (`docs/native-color-management.md`
/// §7/§9): an sRGB JPEG is `Rgb`/no-alpha, an sRGB PNG with an alpha channel
/// is `Rgb`/`Straight`, and a Lab TIFF is `Lab`.
#[test]
fn file_image_source_detects_layout() {
    use poc::color::model::ColorModel;
    use poc::pixel::AlphaState;

    let rgb = poc::data::image::Image2D::<VipsBackend>::open("tests/fixtures/rgb.jpg").unwrap();
    let rgb_layout = rgb.layout();
    assert_eq!(rgb_layout.model, ColorModel::Rgb);
    assert_eq!(rgb_layout.alpha, AlphaState::None);
    assert_eq!(rgb_layout.storage, Storage::U8);

    let rgba = poc::data::image::Image2D::<VipsBackend>::open("tests/fixtures/rgba.png").unwrap();
    let rgba_layout = rgba.layout();
    assert_eq!(rgba_layout.model, ColorModel::Rgb);
    assert_eq!(rgba_layout.alpha, AlphaState::Straight);
    assert_eq!(rgba_layout.storage, Storage::U8);

    let lab = poc::data::image::Image2D::<VipsBackend>::open("tests/fixtures/lab.tif").unwrap();
    let lab_layout = lab.layout();
    assert_eq!(lab_layout.model, ColorModel::Lab);
    assert_eq!(lab_layout.alpha, AlphaState::None);
    assert_eq!(lab_layout.storage, Storage::F32);
}
