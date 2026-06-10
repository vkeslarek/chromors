//! GPU region — a pending or materialised tile from an `Image2D<GpuBackend>`.
//!
//! Lifecycle:
//!   1. `image.new_region()` or `image.new_region_at_mip(mip)`
//!   2. `region.prepare(rect)` — rect is in MIP space
//!   3. `region.materialize()` → compile + dispatch the fused graph
//!   4. Read bytes from the returned [`MaterializedValue`]

use std::sync::Arc;

use super::value::MaterializedValue;
use crate::geometry::Rect;

// ── GpuRegion ────────────────────────────────────────────────────────────────

pub struct GpuRegion {
    pub(crate) graph: Arc<std::sync::Mutex<crate::backend::gpu::Graph>>,
    pub(crate) cache: crate::backend::gpu::RegionCache,
    pub(crate) node_id: crate::backend::gpu::NodeId,
    pub(crate) rect: std::sync::Mutex<Option<Rect>>,
    pub(crate) ctx: Arc<crate::backend::gpu::GpuContext>,
    pub(crate) lod: crate::backend::gpu::Lod,
    /// Content hash of the subgraph rooted at `node_id` — the cache identity of
    /// this region's output, stable across graph forks/sessions.
    pub(crate) content: u64,
}

impl GpuRegion {
    pub fn new(
        graph: Arc<std::sync::Mutex<crate::backend::gpu::Graph>>,
        cache: crate::backend::gpu::RegionCache,
        node_id: crate::backend::gpu::NodeId,
        ctx: Arc<crate::backend::gpu::GpuContext>,
        lod: crate::backend::gpu::Lod,
    ) -> Self {
        let content = graph.lock().unwrap().content_hash(node_id);
        Self {
            graph,
            cache,
            node_id,
            rect: std::sync::Mutex::new(None),
            ctx,
            lod,
            content,
        }
    }

    pub fn prepare(&self, rect: Rect) {
        *self.rect.lock().unwrap() = Some(rect);
    }

    /// Compile the fused DAG and dispatch it on the GPU.
    pub fn materialize(&self) -> Result<Arc<MaterializedValue>, crate::error::Error> {
        let _sw = crate::utils::Stopwatch::new("gpu.materialize");
        let rect = self
            .rect
            .lock()
            .unwrap()
            .expect("GpuRegion::materialize called before prepare()");
        let mut results = super::materialize::execute_batch(self, &[rect])?;
        Ok(results.pop().unwrap())
    }

    /// Process multiple disjoint rects in a single WGPU CommandEncoder to avoid driver overhead.
    pub fn materialize_batch(
        &self,
        rects: &[Rect],
    ) -> Result<Vec<Arc<MaterializedValue>>, crate::error::Error> {
        super::materialize::execute_batch(self, rects)
    }
}
