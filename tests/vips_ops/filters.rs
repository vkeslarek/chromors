//! vips backend — convolution / filter / detection ops.

use crate::common::rgb;
use pixors_engine::backend::vips::VipsBackend;
use pixors_engine::data::image::Image;
use pixors_engine::*;

#[test]
fn gaussian_blur() {
    let blurred = rgb()
        .execute(&GaussianBlurOperation {
            sigma: 5.0,
            minimum_amplitude: None,
            precision: None,
        })
        .unwrap();
    assert_eq!(blurred.width(), 200);
}

#[test]
fn sobel() {
    let edges = rgb().execute(&SobelOperation).unwrap();
    assert_eq!(edges.width(), 200);
}

#[test]
fn invert() {
    let inv = rgb().execute(&InvertOperation).unwrap();
    assert_eq!(inv.width(), 200);
}

#[test]
fn sharpen() {
    let s = rgb()
        .execute(&SharpenOperation {
            sigma: None,
            flat: None,
            jagged: None,
            edge: None,
            smooth: None,
            maximum: None,
        })
        .unwrap();
    assert_eq!(s.width(), 200);
}

#[test]
fn median() {
    let m = rgb().execute(&MedianOperation { size: 3 }).unwrap();
    assert_eq!(m.width(), 200);
}

#[test]
fn convolution_variants() {
    let img = rgb();
    let mask = Image::<VipsBackend>::generate(&GaussMat {
        sigma: 1.0,
        minimum_amplitude: 0.2,
    })
    .unwrap();
    assert_eq!(
        img.execute(&ConvfOperation { mask: mask.clone() })
            .unwrap()
            .width(),
        200
    );
    assert_eq!(
        img.execute(&ConviOperation { mask: mask.clone() })
            .unwrap()
            .width(),
        200
    );

    let row = mask
        .execute(&CropOperation {
            left: 0,
            top: 0,
            width: mask.width(),
            height: 1,
        })
        .unwrap();
    let sep = img
        .execute(&ConvsepOperation {
            mask: row.clone(),
            precision: Some(Precision::Float),
            layers: None,
            cluster: None,
        })
        .unwrap();
    assert_eq!(sep.width(), 200);
}

#[test]
fn correlation() {
    let img = rgb();
    let reference = img
        .execute(&CropOperation {
            left: 0,
            top: 0,
            width: 20,
            height: 20,
        })
        .unwrap();
    let out = img
        .execute(&FastcorOperation {
            reference: reference.clone(),
        })
        .unwrap();
    assert!(out.width() > 0);
}

#[test]
fn hough_line() {
    let edges = rgb()
        .execute(&ExtractBandOperation {
            band: 0,
            count: None,
        })
        .unwrap()
        .execute(&CannyOperation {
            sigma: None,
            precision: None,
        })
        .unwrap();
    let out = edges
        .execute(&HoughLineOperation {
            width: Some(128),
            height: Some(128),
        })
        .unwrap();
    assert_eq!(out.width(), 128);
}
