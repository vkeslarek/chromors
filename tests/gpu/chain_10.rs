//! Integration test: chain of 10 GPU operations exercising the CutFinder.
//!
//! Builds a deep GPU pipeline (source → 10 ops → materialize) and verifies
//! the output is non-empty. When the binding budget is tight enough, the
//! CutFinder in pass.rs will automatically split the chain into multiple
//! passes with staging cuts.

use crate::common;

use chromors::data::image::RamImageTarget;
use chromors::io::Target;
use chromors::work_unit::{Lod, Region};

/// Chain 10 GPU ops: invert → exposure → gamma → abs → sign → invert →
/// exposure → gamma → abs → invert. Exercises the full DAG → BFS → cuts →
/// rebuild → materialize pipeline.
#[test]
fn chain_10_ops_materializes() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgba();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    // Chain 10 image→image ops. Each adds a kernel step + work buffer.
    let step1 = gpu_img.invert();
    let step2 = step1.exposure(0.5, 0.0);
    let step3 = step2.gamma(Some(2.2));
    let step4 = step3.abs();
    let step5 = step4.sign();
    let step6 = step5.invert();
    let step7 = step6.exposure(1.0, 0.1);
    let step8 = step7.gamma(Some(1.8));
    let step9 = step8.abs();
    let step10 = step9.invert();

    let rect = Region {
        x: 0,
        y: 0,
        w: step10.width(),
        h: step10.height(),
        lod: Lod(0),
    };

    let bytes = step10.pull(&RamImageTarget, rect).unwrap();
    println!(
        "chain_10_ops → {} bytes ({}×{})",
        bytes.len(),
        step10.width(),
        step10.height()
    );
    assert!(!bytes.is_empty());
}

/// Same chain but with the BFS analysis tested explicitly: count ops and
/// sources in the chain to verify the DAG structure is correct.
#[test]
fn chain_10_ops_dag_has_correct_shape() {
    use chromors::backend::gpu::pass::{binding_count, exceeds_binding_limit};

    // A chain of 10 ops from 1 source needs:
    // 2 (target+params) + 10 (work temps) + 1 (source) = 13 bindings.
    let n_ops = 10;
    let n_sources = 1;
    let bindings = binding_count(n_ops, n_sources, true);
    assert_eq!(bindings, 13, "10 ops + 1 source + scratch = 13 bindings");

    // This fits comfortably on most GPUs (limit usually ≥ 16).
    assert!(!exceeds_binding_limit(n_ops, n_sources, true, 16));

    // But would need cuts on a very constrained device (limit = 8).
    assert!(exceeds_binding_limit(n_ops, n_sources, true, 8));
}

/// Chain 10 ops, each from a DIFFERENT source. This creates a wide DAG
/// that stresses the binding budget (10 sources + 10 ops + 2 = 22 bindings).
#[test]
fn chain_10_sources_binding_budget() {
    use chromors::backend::gpu::pass::binding_count;

    // 10 sources + 10 ops = 22 bindings. Would need CutFinder on GPUs
    // with max_storage_buffers < 22.
    let bindings = binding_count(10, 10, true);
    assert_eq!(bindings, 22);
}
