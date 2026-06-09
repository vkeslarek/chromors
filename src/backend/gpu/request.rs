//! Typed materialization request — replaces the image-centric [`super::region::GpuRegion`].
//!
//! Use `GpuRequest<ImageData>` for image tiles, `GpuRequest<HistogramData>` for
//! histogram reads, etc.  No fake rects needed.

use std::sync::Arc;

use crate::error::Error;

use super::Lod;
use super::RegionCache;
use super::context::GpuContext;
use super::data::GpuData;
use super::graph::Graph;
use super::region::GpuRegion;

/// A pending materialisation request from a typed graph handle.
///
/// Replaces [`GpuRegion`] with a generic `D: GpuData` parameter so callers
/// specify *what* they want without faking image rects.
pub struct GpuRequest<D: GpuData> {
    pub(crate) graph: Arc<std::sync::Mutex<Graph>>,
    pub(crate) cache: RegionCache,
    pub(crate) node_id: super::graph::NodeId,
    pub(crate) ctx: Arc<GpuContext>,
    pub(crate) lod: Lod,
    pub(crate) data: D,
    pub(crate) req: D::Request,
}

impl<D: GpuData> GpuRequest<D> {
    pub fn new(
        graph: Arc<std::sync::Mutex<Graph>>,
        cache: RegionCache,
        node_id: super::graph::NodeId,
        ctx: Arc<GpuContext>,
        lod: Lod,
        data: D,
        req: D::Request,
    ) -> Self {
        Self {
            graph,
            cache,
            node_id,
            ctx,
            lod,
            data,
            req,
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
            self.data.finish(&mat, &self.req, &self.ctx)
        } else {
            let mats = region.materialize_batch(&rects)?;
            self.data.finish(&mats[0], &self.req, &self.ctx)
        }
    }

    fn compute_rects(&self) -> Result<Vec<crate::geometry::Rect>, Error> {
        let plan = D::plan(&self.graph.lock().unwrap(), self.node_id, &self.req);
        Ok(plan.targets.iter().map(|t| t.rect).collect())
    }
}
