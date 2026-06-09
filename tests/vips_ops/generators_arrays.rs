//! vips backend — generators and array (multi-input) ops.

use crate::common::rgb;
use pixors_engine::backend::vips::VipsBackend;
use pixors_engine::data::image::Image;
use pixors_engine::*;

#[test]
fn freq_masks() {
    crate::common::init();
    let ideal = Image::<VipsBackend>::generate(&MaskIdeal {
        width: 64,
        height: 64,
        frequency_cutoff: 0.5,
        uchar: None,
        nodc: None,
        reject: None,
        optical: None,
    })
    .unwrap();
    assert_eq!(ideal.width(), 64);
    let bw = Image::<VipsBackend>::generate(&MaskButterworth {
        width: 32,
        height: 32,
        order: 2.0,
        frequency_cutoff: 0.5,
        amplitude_cutoff: 0.5,
        uchar: None,
        nodc: None,
        reject: None,
        optical: None,
    })
    .unwrap();
    assert_eq!(bw.width(), 32);
}

#[test]
fn array_join_and_switch() {
    let a = rgb();
    let joined = Image::<VipsBackend>::array_join(
        &[&a, &a],
        &ArrayJoinParams {
            across: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(joined.width(), 400);

    let g = a
        .execute(&ExtractBandOperation {
            band: 0,
            count: None,
        })
        .unwrap();
    let mask = g
        .execute(&RelationalConstOperation {
            relational: OperationRelational::More,
            constants: vec![128.0],
        })
        .unwrap();
    let sw = Image::<VipsBackend>::switch(&[&mask]).unwrap();
    assert_eq!(sw.width(), 200);
}

#[test]
fn sum_and_bandrank() {
    let a = rgb();
    let summed = Image::<VipsBackend>::sum(&[&a, &a]).unwrap();
    assert_eq!(summed.width(), 200);
    let ranked = Image::<VipsBackend>::band_rank(&[&a, &a, &a], 1).unwrap();
    assert_eq!(ranked.width(), 200);
}

#[test]
fn composite_stack() {
    let a = rgb();
    let b = a
        .execute(&ResizeOperation {
            scale: 0.5,
            kernel: None,
            vertical_scale: None,
            gap: None,
        })
        .unwrap();
    let out =
        Image::<VipsBackend>::composite(&[&a, &b], &[BlendMode::Over], &CompositeParams::default())
            .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn merge_mosaic() {
    let a = rgb();
    let merged = a
        .execute(&MergeOperation {
            secondary: &a,
            direction: Direction::Horizontal,
            dx: 200,
            dy: 0,
            max_blend: None,
        })
        .unwrap();
    assert!(merged.width() >= 200);
}
