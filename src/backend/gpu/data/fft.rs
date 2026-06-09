use crate::error::Error;
use crate::geometry::Rect;

use super::super::context::GpuContext;
use super::super::graph::{Graph, NodeId};
use super::super::materialize::MaterializePlan;
use super::super::value::{GraphValue, ValueKind};
use super::super::Lod;
use super::GpuData;

// ── Fft1D ────────────────────────────────────────────────────────────────────

pub type Fft1dRequest = (Lod, u32);

pub struct Fft1dData;

impl GpuData for Fft1dData {
    // 8 bytes per complex element (f32 real, f32 imag)
    type Value = Vec<[f32; 2]>;
    type Request = Fft1dRequest;

    fn value_kind(req: &Self::Request) -> ValueKind {
        ValueKind::Fft1D { length: req.1 }
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
                if matches!(kind, ValueKind::Fft1D { .. }) {
                    let complex: &[[f32; 2]] = bytemuck::cast_slice(bytes);
                    Ok(complex.iter().take(expected_len as usize).copied().collect())
                } else {
                    Err(Error::Gpu("Fft1dData::finish: unexpected value kind".into()))
                }
            }
            GraphValue::Image { .. } => Err(Error::Gpu("Fft1dData::finish: expected Raw".into())),
        }
    }
}

// ── Fft2D ────────────────────────────────────────────────────────────────────

pub type Fft2dRequest = (Lod, u32, u32);

pub struct Fft2dData;

impl GpuData for Fft2dData {
    // 8 bytes per complex element (f32 real, f32 imag)
    type Value = Vec<[f32; 2]>;
    type Request = Fft2dRequest;

    fn value_kind(_req: &Self::Request) -> ValueKind {
        ValueKind::Fft2D
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
                if matches!(kind, ValueKind::Fft2D { .. }) {
                    let complex: &[[f32; 2]] = bytemuck::cast_slice(bytes);
                    Ok(complex.iter().take((expected_w * expected_h) as usize).copied().collect())
                } else {
                    Err(Error::Gpu("Fft2dData::finish: unexpected value kind".into()))
                }
            }
            GraphValue::Image { .. } => Err(Error::Gpu("Fft2dData::finish: expected Raw".into())),
        }
    }
}
