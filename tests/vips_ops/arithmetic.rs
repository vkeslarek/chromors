//! vips backend — arithmetic / logic / reduction ops.

use crate::common::rgb;
use pixors_engine::*;

#[test]
fn linear() {
    let bright = rgb()
        .execute(&LinearOperation {
            a: 2.0,
            b: 0.0,
            uchar: None,
        })
        .unwrap();
    assert_eq!(bright.width(), 200);
}

#[test]
fn add() {
    let img = rgb();
    let result = img.execute(&AddOperation { right: img.clone() }).unwrap();
    assert_eq!(result.width(), 200);
}

#[test]
fn avg_min_max() {
    let img = rgb();
    let avg: f64 = img.execute(&AverageOperation).unwrap();
    let min: f64 = img
        .execute(&MinimumOperation {
            size: None,
            x: None,
            y: None,
        })
        .unwrap();
    let max: f64 = img
        .execute(&MaximumOperation {
            size: None,
            x: None,
            y: None,
        })
        .unwrap();
    assert!(min <= avg && avg <= max);
}

#[test]
fn math_unary() {
    let out = rgb()
        .execute(&MathOperation {
            math: OperationMath::Sin,
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn round_op() {
    let out = rgb()
        .execute(&RoundOperation {
            round: OperationRound::Floor,
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn boolean_binary() {
    let img = rgb();
    let out = img
        .execute(&BooleanOperation {
            right: img.clone(),
            boolean: OperationBoolean::And,
        })
        .unwrap();
    assert_eq!(out.bands(), 3);
}

#[test]
fn relational_const() {
    let out = rgb()
        .execute(&RelationalConstOperation {
            relational: OperationRelational::More,
            constants: vec![128.0],
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn bandbool_and() {
    let out = rgb()
        .execute(&BandboolOperation {
            boolean: OperationBoolean::And,
            bands: 3,
        })
        .unwrap();
    assert_eq!(out.bands(), 1);
}

#[test]
fn math2_const_pow() {
    let out = rgb()
        .execute(&Math2ConstOperation {
            math2: OperationMath2::Pow,
            constants: vec![2.0],
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}
