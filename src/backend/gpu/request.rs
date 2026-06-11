//! Typed materialization request — replaces the image-centric [`super::region::GpuRegion`].
//!
//! Use `GpuRequest<ImageType>` for image tiles, `GpuRequest<HistogramType>` for
//! histogram reads, etc.  No fake rects, no stringly-typed requests.

use std::sync::Arc;

use crate::error::Error;

use super::RegionCache;
use super::context::GpuContext;
use super::datatype::TypedData;
use super::graph::Graph;
use super::region::GpuRegion;
use super::work_unit::{AnyWorkUnit, Region};

/// A pending materialisation request from a typed graph handle.
///
/// Generic over `D: TypedData` — callers provide the DataType and its `WorkUnit`
/// to specify *what* they want without faking image rects or encoding
/// metadata in a stringly-typed request. `D::WorkUnit` carries the LOD
/// (`Region::lod`, defaulting to `Lod::FULL` for `Range`/`Atomic`) — there is
/// no separate `lod` field.
pub struct GpuRequest<D: TypedData> {
    pub(crate) graph: Arc<std::sync::Mutex<Graph>>,
    pub(crate) cache: RegionCache,
    pub(crate) node_id: super::graph::NodeId,
    pub(crate) ctx: Arc<GpuContext>,
    pub(crate) data: D,
    pub(crate) wu: D::WorkUnit,
}

impl<D: TypedData> GpuRequest<D> {
    pub fn new(
        graph: Arc<std::sync::Mutex<Graph>>,
        cache: RegionCache,
        node_id: super::graph::NodeId,
        ctx: Arc<GpuContext>,
        data: D,
        wu: D::WorkUnit,
    ) -> Self {
        Self {
            graph,
            cache,
            node_id,
            ctx,
            data,
            wu,
        }
    }

    /// Compile the fused DAG, dispatch on the GPU, and return the typed value.
    pub fn materialize(&self) -> Result<D::Value, Error> {
        let _sw = crate::utils::Stopwatch::new("gpu.materialize_typed");
        let lod = self.wu.to_work_unit().lod();
        let rects = self.compute_rects(lod)?;

        let region = GpuRegion::new(
            self.graph.clone(),
            self.cache.clone(),
            self.node_id,
            self.ctx.clone(),
        );

        if rects.len() == 1 {
            region.prepare(Region::new(rects[0], lod));
            let mat = region.materialize()?;
            self.data.finish(&mat, &self.wu, &self.ctx)
        } else {
            let wus: Vec<Region> = rects.iter().map(|&r| Region::new(r, lod)).collect();
            let mats = region.materialize_batch(&wus)?;
            self.data.finish(&mats[0], &self.wu, &self.ctx)
        }
    }

    fn compute_rects(&self, lod: super::Lod) -> Result<Vec<crate::geometry::Rect>, Error> {
        let wu = self.wu.to_work_unit();
        let plan = self
            .graph
            .lock()
            .unwrap()
            .materialize(&[(self.node_id, wu)], lod);
        Ok(plan.targets.iter().map(|t| t.rect).collect())
    }
}
