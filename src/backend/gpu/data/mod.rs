pub mod image;
pub mod histogram;
pub mod mask;
pub mod fft;

pub use self::image::*;
pub use histogram::*;
pub use mask::*;
pub use fft::*;

use super::context::GpuContext;
use super::graph::NodeId;
use super::materialize::MaterializePlan;
use super::value::{GraphValue, ValueKind};
use crate::error::Error;

/// Trait implemented per data type — describes what to request, how to plan,
/// and how to wrap the materialised result.
pub trait GpuData: Send + Sync + 'static {
    /// The materialized payload that callers receive.
    type Value: Clone + Send + Sync;

    /// A cache-key-able, hashable description of *what* is being requested.
    /// For images: `(Lod, Rect)`. For histograms: `(Lod, bins)`.
    type Request: Clone + Eq + std::hash::Hash + Send + Sync;

    /// The [`ValueKind`] tag for this typed request.
    fn value_kind(req: &Self::Request) -> ValueKind;

    /// Walk the graph and produce the typed plan for this request.
    fn plan(graph: &super::graph::Graph, root: NodeId, req: &Self::Request) -> MaterializePlan;

    /// Turn a materialised [`GraphValue`] into the typed value.
    fn finish(
        &self,
        value: &GraphValue,
        req: &Self::Request,
        ctx: &GpuContext,
    ) -> Result<Self::Value, Error>;
}
