//! Embedded custom vips operations — run inside the lazy pipeline, no download.
//!
//! Placeholder ops until real algorithms land: [`Invert`] (image output) and
//! [`HistogramSink`] (arbitrary Rust value output).

use crate::common::rgb;
use poc::*;

// ── Invert (VipsCustom → Image2D<VipsBackend>) ────────────────────────────────────

#[test]
fn invert_matches_vips() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn invert_is_chainable() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

// ── Histogram (VipsCustomSink → Rust value) ─────────────────────────────────

#[test]
fn histogram_counts_every_pixel() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn histogram_matches_manual_count() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

// ── unified execute() API via Custom/Reduce wrappers ────────────────────────

#[test]
fn custom_ops_via_execute() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}
