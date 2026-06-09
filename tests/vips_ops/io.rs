//! vips backend — I/O: open, save, buffers, memory, sources.

use crate::common::rgb;
use pixors_engine::backend::vips::VipsBackend;
use pixors_engine::data::image::Image;
use pixors_engine::*;

#[test]
fn open_and_properties() {
    let img = rgb();
    assert_eq!(img.width(), 200);
    assert_eq!(img.height(), 200);
    assert_eq!(img.bands(), 3);
}

#[test]
fn save_buffer() {
    let png = rgb().write_to_buffer(".png").unwrap();
    assert!(!png.is_empty());
}

#[test]
fn round_trip_buffer() {
    let png = rgb().write_to_buffer(".png").unwrap();
    let decoded = Image::<VipsBackend>::from_buffer(&png).unwrap();
    assert_eq!(decoded.width(), 200);
}

#[test]
fn from_memory_roundtrip() {
    crate::common::init();
    let buf = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let img = Image::<VipsBackend>::from_memory(&buf, 2, 2, 3, PixelFormat::Rgb8).unwrap();
    assert_eq!(img.width(), 2);
    assert_eq!(img.bands(), 3);
}

#[test]
fn from_memory_rejects_short_buffer() {
    crate::common::init();
    let buf = vec![0u8; 5];
    assert!(Image::<VipsBackend>::from_memory(&buf, 2, 2, 3, PixelFormat::Rgb8).is_err());
}

#[test]
fn from_memory_rejects_bad_dims() {
    crate::common::init();
    let buf = vec![0u8; 12];
    assert!(Image::<VipsBackend>::from_memory(&buf, 0, 2, 3, PixelFormat::Rgb8).is_err());
}

#[test]
fn source_memory_outlives_buffer() {
    crate::common::init();
    let png = rgb().write_to_buffer(".png").unwrap();
    let source = {
        let owned = png.clone();
        Source::new_from_memory(&owned).unwrap()
    };
    let img = Image::<VipsBackend>::new_from_source(&source).unwrap();
    assert_eq!(img.width(), 200);
}

#[test]
fn clone_and_drop() {
    let img = rgb();
    let cloned = img.clone();
    assert_eq!(cloned.width(), img.width());
    drop(img);
    assert_eq!(cloned.width(), 200);
}

#[test]
fn thumbnail_loaders() {
    crate::common::init();
    let t =
        Image::<VipsBackend>::thumbnail("tests/fixtures/rgb.jpg", 64, &ThumbnailParams::default())
            .unwrap();
    assert!(t.width() <= 64);

    let buf = rgb().write_to_buffer(".png").unwrap();
    let tb = Image::<VipsBackend>::thumbnail_buffer(&buf, 32, &ThumbnailParams::default()).unwrap();
    assert!(tb.width() <= 32);
}
