//! End-to-end cache boundary tests on real GPU pipelines.
//!
//! Proves the `.cache()` boundary: a downstream branch that consumes a cached
//! tip serves from the store, and a second branch off the SAME boundary reuses
//! those tiles instead of recomputing the upstream chain.

use poc::work_unit::{Lod, Region};

use crate::common;

fn full(w: i32, h: i32) -> Region {
    Region { x: 0, y: 0, w, h, lod: Lod(0) }
}

/// A cached chain consumed by a downstream op matches the same chain without a
/// cache (modulo the boundary's storage re-encode), and the store is populated.
#[test]
fn cached_chain_matches_uncached_and_populates_store() {
    let ctx = common::gpu_ctx();
    let vips = common::rgb();
    let gpu = common::vips_to_gpu(&vips, &ctx);

    let cached = gpu.exposure(0.3, 0.0).blur(4.0).cache();
    let out = cached.handle().saturation(1.2);
    let reference = gpu.exposure(0.3, 0.0).blur(4.0).saturation(1.2);

    let got = common::poc_materialize(&out);
    let want = common::poc_materialize(&reference);

    let rms = common::rms_u8(&got, &want);
    assert!(rms < 3.0, "cached path diverged from uncached: rms {rms}");

    let stats = cached.store().stats();
    assert!(stats.entries >= 1, "boundary should hold a tile, got {stats:?}");
    assert!(stats.misses >= 1, "first pull is a miss, got {stats:?}");
}

/// Two downstream branches off one boundary: the second pull is a cache HIT
/// (the exposure+blur upstream runs once, not twice).
#[test]
fn second_branch_hits_cache() {
    let ctx = common::gpu_ctx();
    let vips = common::rgb();
    let gpu = common::vips_to_gpu(&vips, &ctx);

    let cached = gpu.exposure(0.3, 0.0).blur(4.0).cache();

    let branch_a = cached.handle().saturation(1.2);
    let branch_b = cached.handle().invert();

    let _ = common::poc_materialize(&branch_a); // miss → fills the boundary
    let hits_before = cached.store().stats().hits;
    let _ = common::poc_materialize(&branch_b); // same boundary region → hit
    let stats = cached.store().stats();

    assert!(
        stats.hits > hits_before,
        "second branch must reuse the cached boundary, stats {stats:?}"
    );
}

/// `prime` warms a region without a consumer; a later pull of that region hits.
#[test]
fn prime_warms_then_pull_hits() {
    let ctx = common::gpu_ctx();
    let vips = common::rgb();
    let (w, h) = (vips.width(), vips.height());
    let gpu = common::vips_to_gpu(&vips, &ctx);

    let cached = gpu.exposure(0.3, 0.0).cache();
    cached.prime(&[full(w, h)]).unwrap();

    let primed = cached.store().stats();
    assert_eq!(primed.entries, 1, "prime should cache the region, {primed:?}");

    // A downstream pull demanding the same region serves from the warm store.
    let out = cached.handle().invert();
    let _ = common::poc_materialize(&out);
    let stats = cached.store().stats();
    assert!(stats.hits >= 1, "primed region must be a hit, {stats:?}");
}
