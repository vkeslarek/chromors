//! Integration tests for GPU pass splitting (CutFinder) and demand tiling.
//!
//! These tests verify:
//! 1. `WorkUnit::split` correctly subdivides Region/Range work units and
//!    errors on Atomic.
//! 2. `pass::binding_count` correctly computes the number of storage buffer
//!    bindings a fused pass requires.
//! 3. `pass::exceeds_binding_limit` detects when a pass would exceed the
//!    device's `max_storage_buffers` limit.

use poc::work_unit::*;

// ── WorkUnit::split ──────────────────────────────────────────────────────────

#[test]
fn split_region_fits_returns_single_tile() {
    let wu = WorkUnit::Region(Region { x: 0, y: 0, w: 100, h: 100, lod: Lod(0) });
    let tiles = wu.split(1_000_000, |wu| match wu {
        WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 4, // 4 bpp
        _ => 0,
    }).unwrap();
    assert_eq!(tiles.len(), 1);
    assert_eq!(tiles[0], wu);
}

#[test]
fn split_region_bisects_landscape() {
    // 200×100 image, 4 bpp = 80,000 bytes. Limit = 50,000 → must split.
    let wu = WorkUnit::Region(Region { x: 0, y: 0, w: 200, h: 100, lod: Lod(0) });
    let tiles = wu.split(50_000, |wu| match wu {
        WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 4,
        _ => 0,
    }).unwrap();
    assert!(tiles.len() >= 2, "expected at least 2 tiles, got {}", tiles.len());

    // All tiles should be within the limit.
    for tile in &tiles {
        let bytes = match tile {
            WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 4,
            _ => panic!("expected Region"),
        };
        assert!(bytes <= 50_000, "tile {} bytes exceeds 50,000 limit", bytes);
    }

    // Tiles should cover the full region.
    let total_pixels: i64 = tiles.iter().map(|t| match t {
        WorkUnit::Region(r) => r.w as i64 * r.h as i64,
        _ => 0,
    }).sum();
    assert_eq!(total_pixels, 200 * 100);
}

#[test]
fn split_region_bisects_portrait() {
    // 100×200 image, 4 bpp = 80,000 bytes. Limit = 50,000 → splits along h.
    let wu = WorkUnit::Region(Region { x: 0, y: 0, w: 100, h: 200, lod: Lod(0) });
    let tiles = wu.split(50_000, |wu| match wu {
        WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 4,
        _ => 0,
    }).unwrap();
    assert!(tiles.len() >= 2);

    // First tile should start at y=0, second at y=100.
    let WorkUnit::Region(first) = &tiles[0] else { panic!() };
    let WorkUnit::Region(second) = &tiles[1] else { panic!() };
    assert_eq!(first.y, 0);
    assert_eq!(first.h, 100);
    assert_eq!(second.y, 100);
    assert_eq!(second.h, 100);
}

#[test]
fn split_region_recursive_many_tiles() {
    // 1000×1000 image, 16 bpp = 16 MB. Limit = 1 MB → needs 16+ tiles.
    let wu = WorkUnit::Region(Region { x: 0, y: 0, w: 1000, h: 1000, lod: Lod(0) });
    let tiles = wu.split(1_000_000, |wu| match wu {
        WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 16,
        _ => 0,
    }).unwrap();
    assert!(tiles.len() >= 16, "expected at least 16 tiles, got {}", tiles.len());

    // Every tile fits.
    for tile in &tiles {
        let bytes = match tile {
            WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 16,
            _ => panic!(),
        };
        assert!(bytes <= 1_000_000);
    }

    // Coverage: total pixels preserved.
    let total: i64 = tiles.iter().map(|t| match t {
        WorkUnit::Region(r) => r.w as i64 * r.h as i64,
        _ => 0,
    }).sum();
    assert_eq!(total, 1_000_000);
}

#[test]
fn split_region_preserves_offset_and_lod() {
    // Region with a non-zero origin and LOD.
    let wu = WorkUnit::Region(Region { x: 50, y: 100, w: 200, h: 200, lod: Lod(2) });
    let tiles = wu.split(50_000, |wu| match wu {
        WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 4,
        _ => 0,
    }).unwrap();

    for tile in &tiles {
        let WorkUnit::Region(r) = tile else { panic!() };
        assert_eq!(r.lod, Lod(2), "LOD not preserved");
        assert!(r.x >= 50, "x origin out of bounds");
        assert!(r.y >= 100, "y origin out of bounds");
    }
}

#[test]
fn split_range_bisects() {
    let wu = WorkUnit::Range(Range { start: 0, end: 1000 });
    let tiles = wu.split(500, |wu| match wu {
        WorkUnit::Range(r) => (r.end - r.start) as u64,
        _ => 0,
    }).unwrap();
    assert!(tiles.len() >= 2);
    for tile in &tiles {
        let bytes = match tile {
            WorkUnit::Range(r) => (r.end - r.start) as u64,
            _ => panic!(),
        };
        assert!(bytes <= 500);
    }
}

#[test]
fn split_atomic_returns_error() {
    let wu = WorkUnit::Atomic;
    let result = wu.split(100, |_| 200);
    assert!(result.is_err(), "Atomic split should return an error");
}

#[test]
fn split_atomic_below_limit_returns_single() {
    let wu = WorkUnit::Atomic;
    let tiles = wu.split(100, |_| 50).unwrap();
    assert_eq!(tiles.len(), 1);
}

// ── pass::binding_count ──────────────────────────────────────────────────────

#[test]
fn binding_count_scratch_output() {
    use poc::backend::gpu::pass::binding_count;
    // 3 steps, 2 sources, scratch output → 2 + 3 + 2 = 7
    assert_eq!(binding_count(3, 2, true), 7);
}

#[test]
fn binding_count_direct_output() {
    use poc::backend::gpu::pass::binding_count;
    // 3 steps, 2 sources, direct output → 2 + 2 + 2 = 6
    assert_eq!(binding_count(3, 2, false), 6);
}

#[test]
fn binding_count_zero_steps() {
    use poc::backend::gpu::pass::binding_count;
    // 0 steps (bare source passthrough), 1 source, scratch → 2 + 0 + 1 = 3
    assert_eq!(binding_count(0, 1, true), 3);
    // 0 steps, 1 source, direct → 2 + 0 + 1 = 3 (saturating_sub(1) of 0 = 0)
    assert_eq!(binding_count(0, 1, false), 3);
}

#[test]
fn exceeds_binding_limit_check() {
    use poc::backend::gpu::pass::exceeds_binding_limit;
    // 10 steps + 5 sources + scratch = 2 + 10 + 5 = 17 bindings.
    // With a limit of 16: exceeds.
    assert!(exceeds_binding_limit(10, 5, true, 16));
    // With a limit of 17: fits.
    assert!(!exceeds_binding_limit(10, 5, true, 17));
    // With a limit of 18: fits.
    assert!(!exceeds_binding_limit(10, 5, true, 18));
}
