//! Embedded custom vips operations — run inside the lazy pipeline, no download.
//!
//! Placeholder ops until real algorithms land: [`Invert`] (image output) and
//! [`HistogramSink`] (arbitrary Rust value output).

use crate::common::rgb;
use chromors::*;

// ── Invert (VipsCustomOperation → Image2D<VipsBackend>) ────────────────────────────────────

#[test]
fn invert_matches_vips() {
    let img = rgb();
    let (w, _h) = (img.width(), img.height());

    // Embedded custom op (lazy, region-by-region — never downloaded here).
    let custom = img.custom(Invert).unwrap();
    assert_eq!(custom.width(), w);
    assert_eq!(custom.bands(), img.bands());
    let custom_bytes = crate::common::vips_materialize(&custom);

    // Reference: native vips invert.
    let vips = img.execute(&InvertOperation).unwrap();
    let vips_bytes = crate::common::vips_materialize(&vips);

    assert_eq!(
        custom_bytes, vips_bytes,
        "custom invert must match vips invert"
    );
}

#[test]
fn invert_is_chainable() {
    let img = rgb();
    let twice = img.custom(Invert).unwrap().custom(Invert).unwrap();
    let (_w, _h) = (img.width(), img.height());
    let orig = crate::common::vips_materialize(&img);
    let back = crate::common::vips_materialize(&twice);
    assert_eq!(orig, back, "invert∘invert is identity");
}

// ── Histogram (VipsCustomSink → Rust value) ─────────────────────────────────

#[test]
fn histogram_counts_every_pixel() {
    let img = rgb();
    let (w, h) = (img.width(), img.height());
    let bands = img.bands() as usize;

    let hist = img.sink(HistogramSink).unwrap();
    assert_eq!(hist.bins.len(), bands);

    // Every band's bins sum to the pixel count — proves the sink visited all.
    let pixels = w as u64 * h as u64;
    for b in 0..bands {
        assert_eq!(hist.count(b), pixels, "band {b} bin total != pixel count");
    }
}

#[test]
fn histogram_matches_manual_count() {
    let img = rgb();
    let (_w, _h) = (img.width(), img.height());
    let bands = img.bands() as usize;

    let hist = img.sink(HistogramSink).unwrap();

    // Reference: download + count band 0.
    let bytes = crate::common::vips_materialize(&img);
    let mut reference = [0u32; 256];
    for px in bytes.chunks_exact(bands) {
        reference[px[0] as usize] += 1;
    }
    assert_eq!(hist.bins[0], reference, "histogram band 0 mismatch");
}

// ── unified execute() API via Custom/Reduce wrappers ────────────────────────

#[test]
fn custom_ops_via_execute() {
    let img = rgb();
    let (w, h) = (img.width(), img.height());

    // Image2D<VipsBackend>-output custom op through the same execute() as vips ops.
    let inv = img.execute(&Custom(Invert)).unwrap();
    let inv_bytes = crate::common::vips_materialize(&inv);
    let ref_out = img.execute(&InvertOperation).unwrap();
    let ref_bytes = crate::common::vips_materialize(&ref_out);
    assert_eq!(inv_bytes, ref_bytes);

    // Value-output reduction through execute().
    let hist: Histogram = img.execute(&Reduce(HistogramSink)).unwrap();
    assert_eq!(hist.count(0), w as u64 * h as u64);
}
