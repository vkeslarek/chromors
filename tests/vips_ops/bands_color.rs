//! vips backend — band manipulation, format casts, colour ops.

use crate::common::rgb;
use pixors_engine::*;

#[test]
fn extract_band() {
    let r = rgb()
        .execute(&ExtractBandOperation {
            band: 0,
            count: None,
        })
        .unwrap();
    assert_eq!(r.bands(), 1);
}

#[test]
fn bandjoin() {
    let r = rgb()
        .execute(&ExtractBandOperation {
            band: 0,
            count: None,
        })
        .unwrap();
    let joined = r.bandjoin(&r).unwrap();
    assert_eq!(joined.bands(), 2);
}

#[test]
fn bandjoin_const() {
    let joined = rgb().bandjoin_const(&[0.5]).unwrap();
    assert_eq!(joined.bands(), 4);
}

#[test]
fn bandjoin_const_rejects_empty() {
    assert!(rgb().bandjoin_const(&[]).is_err());
}

#[test]
fn copy_and_cast() {
    let img = rgb();
    let cp = img
        .execute(&CopyOperation {
            width: None,
            height: None,
            bands: None,
            format: None,
            interpretation: None,
            horizontal_resolution: None,
            vertical_resolution: None,
            offset_x: None,
            offset_y: None,
        })
        .unwrap();
    assert_eq!(cp.width(), 200);

    let casted = img
        .execute(&CastOperation {
            format: PixelFormat::RgbF32,
            shift: None,
        })
        .unwrap();
    assert!(matches!(casted.pixel_format(), PixelFormat::RgbF32));
}

#[test]
fn falsecolour() {
    let g = rgb()
        .execute(&ExtractBandOperation {
            band: 0,
            count: None,
        })
        .unwrap();
    let out = g.execute(&FalsecolourOperation).unwrap();
    assert_eq!(out.bands(), 3);
}

#[test]
fn ifthenelse() {
    let a = rgb();
    let mask = a
        .execute(&RelationalConstOperation {
            relational: OperationRelational::More,
            constants: vec![128.0],
        })
        .unwrap();
    let out = mask
        .execute(&IfthenelseOperation {
            if_true: &a,
            if_false: &a,
            blend: None,
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn rad_roundtrip() {
    let f = rgb().execute(&Float2radOperation).unwrap();
    let back = f.execute(&Rad2floatOperation).unwrap();
    assert_eq!(back.width(), 200);
}

#[test]
fn composite2() {
    let a = rgb();
    let half = a
        .execute(&ResizeOperation {
            scale: 0.5,
            kernel: None,
            vertical_scale: None,
            gap: None,
        })
        .unwrap();
    let c = a
        .execute(&Composite2Operation {
            overlay: half,
            mode: BlendMode::Over,
            x: None,
            y: None,
            compositing_space: None,
            premultiplied: None,
        })
        .unwrap();
    assert_eq!(c.width(), 200);
}
