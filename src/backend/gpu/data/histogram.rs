use crate::geometry::Rect;
use crate::error::Error;

use super::super::context::GpuContext;
use super::super::graph::{Graph, NodeId};
use super::super::materialize::MaterializePlan;
use super::super::value::{GraphValue, ValueKind};
use super::super::Lod;
use super::GpuData;

/// Request type for histogram data. `(lod, bins)` — no fake rect.
pub type HistogramRequest = (Lod, u32);

/// [`GpuData`] impl for histogram accumulators.
pub struct HistogramData;

impl GpuData for HistogramData {
    type Value = Vec<u32>;
    type Request = HistogramRequest;

    fn value_kind(req: &Self::Request) -> ValueKind {
        ValueKind::Histogram { bins: req.1 }
    }

    fn plan(graph: &Graph, root: NodeId, req: &Self::Request) -> MaterializePlan {
        let (lod, bins) = *req;
        graph.materialize(&[(root, Rect::new(0, 0, bins as i32, 1))], lod)
    }

    fn finish(
        &self,
        value: &GraphValue,
        req: &Self::Request,
        _ctx: &GpuContext,
    ) -> Result<Self::Value, Error> {
        let (_lod, expected_bins) = *req;
        match value {
            GraphValue::Raw { bytes, kind, .. } => {
                if matches!(kind, ValueKind::Histogram { .. }) {
                    Ok(bytes
                        .chunks_exact(4)
                        .take(expected_bins as usize)
                        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect())
                } else {
                    Err(Error::Gpu(
                        "HistogramData::finish: unexpected value kind".into(),
                    ))
                }
            }
            GraphValue::Image { .. } => Err(Error::Gpu(
                "HistogramData::finish: expected Raw, got Image".into(),
            )),
        }
    }
}
