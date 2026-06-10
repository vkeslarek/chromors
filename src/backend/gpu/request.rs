//! Typed materialization request — replaces the image-centric [`super::region::GpuRegion`].
//!
//! Use `GpuRequest<ImageType>` for image tiles, `GpuRequest<HistogramType>` for
//! histogram reads, etc.  No fake rects, no stringly-typed requests.

use std::sync::Arc;

use crate::error::Error;

use super::Lod;
use super::RegionCache;
use super::context::GpuContext;
use super::datatype::TypedData;
use super::graph::Graph;
use super::region::GpuRegion;

/// A pending materialisation request from a typed graph handle.
///
/// Generic over `D: TypedData` — callers provide the DataType and its `WorkUnit`
/// to specify *what* they want without faking image rects or encoding
/// metadata in a stringly-typed request.
pub struct GpuRequest<D: TypedData> {
    pub(crate) graph: Arc<std::sync::Mutex<Graph>>,
    pub(crate) cache: RegionCache,
    pub(crate) node_id: super::graph::NodeId,
    pub(crate) ctx: Arc<GpuContext>,
    pub(crate) lod: Lod,
    pub(crate) data: D,
    pub(crate) wu: D::WorkUnit,
}

impl<D: TypedData> GpuRequest<D> {
    pub fn new(
        graph: Arc<std::sync::Mutex<Graph>>,
        cache: RegionCache,
        node_id: super::graph::NodeId,
        ctx: Arc<GpuContext>,
        lod: Lod,
        data: D,
        wu: D::WorkUnit,
    ) -> Self {
        Self {
            graph,
            cache,
            node_id,
            ctx,
            lod,
            data,
            wu,
        }
    }

    /// Compile the fused DAG, dispatch on the GPU, and return the typed value.
    pub fn materialize(&self) -> Result<D::Value, Error> {
        let _sw = crate::utils::Stopwatch::new("gpu.materialize_typed");
        let rects = self.compute_rects()?;

        let region = GpuRegion::new(
            self.graph.clone(),
            self.cache.clone(),
            self.node_id,
            self.ctx.clone(),
            self.lod,
        );

        if rects.len() == 1 {
            region.prepare(rects[0]);
            let mat = region.materialize()?;
            self.data.finish(&mat, self.lod, &self.wu, &self.ctx)
        } else {
            let mats = region.materialize_batch(&rects)?;
            self.data.finish(&mats[0], self.lod, &self.wu, &self.ctx)
        }
    }

    fn compute_rects(&self) -> Result<Vec<crate::geometry::Rect>, Error> {
        use super::work_unit::AnyWorkUnit;
        let wu = self.wu.to_work_unit();
        let plan = self
            .graph
            .lock()
            .unwrap()
            .materialize(&[(self.node_id, wu)], self.lod);
        Ok(plan.targets.iter().map(|t| t.rect).collect())
    }
}
