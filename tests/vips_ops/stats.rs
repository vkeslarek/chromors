//! vips backend — statistics, histogram, region analysis ops.

use crate::common::rgb;
use chromors::backend::vips::VipsBackend;
use chromors::data::image::Image2D;
use chromors::*;

#[test]
fn stats_and_project() {
    let img = rgb();
    let stats = img.execute(&StatsOperation).unwrap();
    assert!(stats.width() > 0);
    let proj = img.execute(&ProjectOperation).unwrap();
    assert_eq!(proj.columns.width(), 200);
    assert_eq!(proj.rows.height(), 200);
}

#[test]
fn labelregions_and_fill() {
    let img = rgb();
    let labels = img.execute(&LabelregionsOperation).unwrap();
    assert!(labels.segments >= 0);
    assert_eq!(labels.mask.width(), 200);
    let filled = img.execute(&FillNearestOperation).unwrap();
    assert_eq!(filled.value.width(), 200);
    assert_eq!(filled.distance.width(), 200);
}

#[test]
fn hist_ismonotonic_on_identity() {
    crate::common::init();
    let lut = Image2D::<VipsBackend>::generate(&Identity).unwrap();
    let m = lut.execute(&HistIsmonotonicOperation).unwrap();
    assert!(m.0);
}

#[test]
fn getpoint_reads_pixel() {
    let values = rgb()
        .execute(&GetpointOperation {
            x: 0,
            y: 0,
            unpack_complex: None,
        })
        .unwrap();
    assert_eq!(values.0.len(), 3);
}

#[test]
fn percent_threshold() {
    let t = rgb().execute(&PercentOperation { percent: 50.0 }).unwrap();
    assert!(t.0 >= 0 && t.0 <= 255);
}
