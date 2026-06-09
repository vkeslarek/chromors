use crate::error::Error;
use crate::geometry::Rect;

use super::super::context::GpuContext;
use super::super::graph::{Graph, NodeId};
use super::super::materialize::MaterializePlan;
use super::super::value::{GraphValue, ValueKind};
use super::super::Lod;
use super::GpuData;

// ── Mask1D ────────────────────────────────────────────────────────────────────

pub type Mask1dRequest = (Lod, u32);

pub struct Mask1dData;

impl GpuData for Mask1dData {
    type Value = Vec<f32>;
    type Request = Mask1dRequest;

    fn value_kind(req: &Self::Request) -> ValueKind {
        ValueKind::Mask1D { length: req.1 }
    }

    fn plan(graph: &Graph, root: NodeId, req: &Self::Request) -> MaterializePlan {
        let (lod, length) = *req;
        graph.materialize(&[(root, Rect::new(0, 0, length as i32, 1))], lod)
    }

    fn finish(
        &self,
        value: &GraphValue,
        req: &Self::Request,
        _ctx: &GpuContext,
    ) -> Result<Self::Value, Error> {
        let (_lod, expected_len) = *req;
        match value {
            GraphValue::Raw { bytes, kind, .. } => {
                if matches!(kind, ValueKind::Mask1D { .. }) {
                    let floats: &[f32] = bytemuck::cast_slice(bytes);
                    Ok(floats.iter().take(expected_len as usize).copied().collect())
                } else {
                    Err(Error::Gpu("Mask1dData::finish: unexpected value kind".into()))
                }
            }
            GraphValue::Image { .. } => Err(Error::Gpu("Mask1dData::finish: expected Raw".into())),
        }
    }
}

// ── Mask2D ────────────────────────────────────────────────────────────────────

pub type Mask2dRequest = (Lod, u32, u32);

pub struct Mask2dData;

impl GpuData for Mask2dData {
    type Value = Vec<f32>;
    type Request = Mask2dRequest;

    fn value_kind(req: &Self::Request) -> ValueKind {
        ValueKind::Mask2D {
            width: req.1,
            height: req.2,
        }
    }

    fn plan(graph: &Graph, root: NodeId, req: &Self::Request) -> MaterializePlan {
        let (lod, width, height) = *req;
        graph.materialize(&[(root, Rect::new(0, 0, width as i32, height as i32))], lod)
    }

    fn finish(
        &self,
        value: &GraphValue,
        req: &Self::Request,
        _ctx: &GpuContext,
    ) -> Result<Self::Value, Error> {
        let (_lod, expected_w, expected_h) = *req;
        match value {
            GraphValue::Raw { bytes, kind, .. } => {
                if matches!(kind, ValueKind::Mask2D { .. }) {
                    let floats: &[f32] = bytemuck::cast_slice(bytes);
                    Ok(floats.iter().take((expected_w * expected_h) as usize).copied().collect())
                } else {
                    Err(Error::Gpu("Mask2dData::finish: unexpected value kind".into()))
                }
            }
            GraphValue::Image { .. } => Err(Error::Gpu("Mask2dData::finish: expected Raw".into())),
        }
    }
}
