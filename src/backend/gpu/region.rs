//! GPU region — a pending or materialised tile from an `Image2D<GpuBackend>`.
//!
//! Lifecycle:
//!   1. `image.new_region()` or `image.new_region_at_mip(mip)`
//!   2. `region.prepare(wu)` — `wu: Region` (rect + lod), rect is in MIP space
//!   3. `region.materialize()` → compile + dispatch the fused graph
//!   4. Read bytes from the returned [`MaterializedValue`]

use std::sync::Arc;

use super::value::MaterializedValue;
use super::work_unit::Region;

// ── GpuRegion ────────────────────────────────────────────────────────────────

pub struct GpuRegion {
    pub(crate) graph: Arc<std::sync::Mutex<crate::backend::gpu::Graph>>,
    pub(crate) cache: crate::backend::gpu::RegionCache,
    pub(crate) node_id: crate::backend::gpu::NodeId,
    pub(crate) wu: std::sync::Mutex<Option<Region>>,
    pub(crate) ctx: Arc<crate::backend::gpu::GpuContext>,
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
    ) -> Self {
        let content = graph.lock().unwrap().content_hash(node_id);
        Self {
            graph,
            cache,
            node_id,
            wu: std::sync::Mutex::new(None),
            ctx,
            content,
        }
    }

    pub fn prepare(&self, wu: Region) {
        *self.wu.lock().unwrap() = Some(wu);
    }

    /// Compile the fused DAG and dispatch it on the GPU.
    pub fn materialize(&self) -> Result<Arc<MaterializedValue>, crate::error::Error> {
        let _sw = crate::utils::Stopwatch::new("gpu.materialize");
        let wu = self
            .wu
            .lock()
            .unwrap()
            .clone()
            .expect("GpuRegion::materialize called before prepare()");
        let mut results = super::materialize::execute_batch(self, &[wu])?;
        Ok(results.pop().unwrap())
    }

    /// Process multiple disjoint work units in a single WGPU CommandEncoder to
    /// avoid driver overhead.
    pub fn materialize_batch(
        &self,
        wus: &[Region],
    ) -> Result<Vec<Arc<MaterializedValue>>, crate::error::Error> {
        super::materialize::execute_batch(self, wus)
    }
}
