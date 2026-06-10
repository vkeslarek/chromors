//! GPU handle types and the LOD abstraction.
//!
//! Extracted from `mod.rs` to keep handle definitions co-located with their
//! `Send`/`Sync` assertions and helper methods.

use std::sync::{Arc, Mutex};

use crate::geometry::Rect;

use super::context::GpuContext;
use super::graph::{Graph, NodeId};

// ── Lod ──────────────────────────────────────────────────────────────────────

/// Level-of-detail level. `Lod(0)` = full resolution, `Lod(n)` = `1/2^n`.
/// Carried on `GpuRegion`, not on `Image2D` — the image handle is LOD-agnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Lod(pub u32);

impl Lod {
    pub const FULL: Lod = Lod(0);

    /// Returns `2^n` as an `f64`.  Used for LOD-scaled dimension computation.
    ///
    /// Prefer this over `(1 << lod.0) as f64` — the `u64` cast avoids integer
    /// overflow for large LOD values.
    pub fn scale_factor(self) -> f64 {
        (1u64 << self.0) as f64
    }

    pub fn is_full(self) -> bool {
        self.0 == 0
    }
}

// ── GraphNodeHandle ───────────────────────────────────────────────────────────

/// A typed reference into a shared graph, tied to a specific root node.
///
/// All `Image2D<GpuBackend>` and typed data handles (`Histogram<GpuBackend>`, …)
/// are thin wrappers around this handle.
///
/// The materialized-tile cache lives on [`GpuContext::cache`] — shared across
/// all graph handles for the same device.
#[derive(Clone)]
pub struct GraphNodeHandle {
    pub graph: Arc<Mutex<Graph>>,
    pub root_id: NodeId,
    pub ctx: Arc<GpuContext>,
}

impl GraphNodeHandle {
    /// Speculatively compile the pipeline for `root_id` at `lod` on a
    /// background thread, bounded by the compile-slot semaphore.
    ///
    /// Call this once after the full filter chain is assembled (not after every
    /// individual `execute()`), and only for the final image that will actually
    /// be rendered.
    pub fn warmup(&self, lod: Lod) {
        let rect = Rect::new(0, 0, 1, 1);
        let (plan, ir) = {
            let g = self.graph.lock().unwrap();
            let plan = g.materialize(
                &[(
                    self.root_id,
                    crate::backend::gpu::WorkUnit::Region { rect, lod },
                )],
                lod,
            );
            let (ir, _) = plan.emit_ir_with_layout(&g, self.ctx.wg_dim, lod);
            (plan, ir)
        };
        let (shader_dir, out_dir) = crate::backend::gpu::compile::shader_paths();
        let hash_val = ir.cache_key();
        if !self.ctx.pipeline_cache.read().unwrap().contains(&hash_val) {
            let _ = crate::backend::gpu::compile::DispatchPass::compile(
                ir,
                &plan,
                &shader_dir,
                &out_dir,
                &self.ctx,
            );
        }
    }
}
