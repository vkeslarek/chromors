//! vips backend — geometry / resampling ops.

use crate::common::rgb;
use pixors_engine::*;

#[test]
fn resize() {
    let half = rgb()
        .execute(&ResizeOperation {
            scale: 0.5,
            kernel: None,
            vertical_scale: None,
            gap: None,
        })
        .unwrap();
    assert_eq!(half.width(), 100);
}

#[test]
fn resize_with_kernel() {
    let half = rgb()
        .execute(&ResizeOperation {
            scale: 0.5,
            kernel: Some(Kernel::Lanczos3),
            vertical_scale: None,
            gap: None,
        })
        .unwrap();
    assert_eq!(half.width(), 100);
}

#[test]
fn crop() {
    let cropped = rgb()
        .execute(&CropOperation {
            left: 10,
            top: 10,
            width: 50,
            height: 50,
        })
        .unwrap();
    assert_eq!(cropped.width(), 50);
}

#[test]
fn flip() {
    let flipped = rgb()
        .execute(&FlipOperation {
            direction: Direction::Horizontal,
        })
        .unwrap();
    assert_eq!(flipped.width(), 200);
}

#[test]
fn rot90() {
    let r = rgb()
        .execute(&Rot90Operation { angle: Angle::D90 })
        .unwrap();
    assert_eq!(r.width(), 200);
}

#[test]
fn embed() {
    let e = rgb()
        .execute(&EmbedOperation {
            x: 10,
            y: 10,
            width: 300,
            height: 300,
            extend: None,
            background: None,
        })
        .unwrap();
    assert_eq!(e.width(), 300);
}

#[test]
fn find_trim() {
    let e = rgb()
        .execute(&EmbedOperation {
            x: 10,
            y: 10,
            width: 300,
            height: 300,
            extend: None,
            background: None,
        })
        .unwrap();
    let bounds = e
        .execute(&FindTrimOperation {
            background: None,
            threshold: None,
            line_art: None,
        })
        .unwrap();
    assert!(bounds.width <= 300 && bounds.height <= 300);
}

#[test]
fn insert_and_join() {
    let a = rgb();
    let small = a
        .execute(&ResizeOperation {
            scale: 0.25,
            kernel: None,
            vertical_scale: None,
            gap: None,
        })
        .unwrap();
    let ins = a
        .execute(&InsertOperation {
            sub: small.clone(),
            x: 50,
            y: 50,
            expand: None,
            background: None,
        })
        .unwrap();
    assert_eq!(ins.width(), 200);
    let joined = a
        .execute(&JoinOperation {
            right: a.clone(),
            direction: Direction::Horizontal,
            expand: None,
            shim: None,
            background: None,
            align: None,
        })
        .unwrap();
    assert_eq!(joined.width(), 400);
}

#[test]
fn extract_area() {
    let out = rgb()
        .execute(&ExtractAreaOperation {
            left: 10,
            top: 10,
            width: 50,
            height: 50,
        })
        .unwrap();
    assert_eq!(out.width(), 50);
}

#[test]
fn replicate() {
    let out = rgb()
        .execute(&ReplicateOperation { across: 2, down: 3 })
        .unwrap();
    assert_eq!(out.width(), 400);
    assert_eq!(out.height(), 600);
}

#[test]
fn zoom() {
    let out = rgb()
        .execute(&ZoomOperation {
            horizontal: 2,
            vertical: 2,
        })
        .unwrap();
    assert_eq!(out.width(), 400);
}

#[test]
fn subsample() {
    let out = rgb()
        .execute(&SubsampleOperation {
            horizontal: 2,
            vertical: 2,
            point: None,
        })
        .unwrap();
    assert_eq!(out.width(), 100);
}

#[test]
fn grid() {
    let out = rgb()
        .execute(&GridOperation {
            tile_height: 100,
            across: 2,
            down: 1,
        })
        .unwrap();
    assert_eq!(out.width(), 400);
}

#[test]
fn affine_identity() {
    let out = rgb()
        .execute(&AffineOperation {
            matrix: vec![1.0, 0.0, 0.0, 1.0],
            interpolate: None,
            output_area: None,
            offset_input_x: None,
            offset_input_y: None,
            offset_output_x: None,
            offset_output_y: None,
            background: None,
            premultiplied: None,
            extend: None,
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn similarity_scale() {
    let out = rgb()
        .execute(&SimilarityOperation {
            scale: Some(0.5),
            angle: None,
            interpolate: None,
            background: None,
            offset_input_x: None,
            offset_input_y: None,
            offset_output_x: None,
            offset_output_y: None,
        })
        .unwrap();
    assert_eq!(out.width(), 100);
}

#[test]
fn reduce_shrink_1d() {
    let img = rgb();
    let rh = img
        .execute(&ReduceHorizontalOperation {
            shrink: 2.0,
            kernel: None,
            gap: None,
        })
        .unwrap();
    assert_eq!(rh.width(), 100);
    let sv = img
        .execute(&ShrinkVerticalOperation {
            shrink: 2,
            ceil: None,
        })
        .unwrap();
    assert_eq!(sv.height(), 100);
}

#[test]
fn thumbnail() {
    let thumb = rgb()
        .execute(&ThumbnailOperation {
            width: 64,
            height: None,
            size: None,
            crop: None,
            linear: None,
            auto_rotate: None,
            no_rotate: None,
            import_profile: None,
            export_profile: None,
            intent: None,
            fail_on: None,
        })
        .unwrap();
    assert!(thumb.width() <= 64);
}
