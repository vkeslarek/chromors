//! JIT-fused GPU execution.
//!
//! Each `Graph` accumulates nodes emitted by `GpuOperation::emit()`. When
//! compiled, the graph is topo-sorted and a single fused SPIR-V entry point
//! is generated per node.

pub mod arena;
pub mod buffer;
pub mod builder;
pub mod cache;
pub mod compile;
pub mod context;
pub mod datatype;
pub mod emit;
pub mod graph;
pub mod handle;
pub mod materialize;
pub mod op;
pub mod param;
pub mod pass;
pub mod region;
pub mod request;
pub mod slang;
// FIXME: re-enable when libslang name collision is resolved
// pub mod slang_ffi;
pub mod source;
pub mod target;
pub mod typed;
pub mod value;
pub mod work_unit;

pub use crate::geometry::Rect;
pub use buffer::{GpuBuffer, ImageBuffer};
pub use compile::{Compiled, CompiledShader, DispatchPass};
pub use context::GpuContext;
pub use context::RegionCache;
pub use datatype::{
    DataType, Executable, FeaturesType, Fft1dType, Fft2dType, HistogramType, ImageType, Mask1dType,
    Mask2dType, PointListType, ScalarType, Sourceable, Targetable, TypedData,
};
pub use graph::NodeEval;
pub use graph::{Graph, GraphNode, KernelSpec, NodeId, SourceNode};
pub use handle::{GraphNodeHandle, Lod};
pub use op::{
    DispatchGrid, GpuOperation, InputEncoder, OutputCodec, OutputDecoder, TypedOperation,
};
pub use source::{AnyGpuSource, GpuSource};
pub use typed::HistogramBuffer;
pub use value::{MaterializedValue, Storage};
pub use work_unit::{AnyWorkUnit, Atomic, Range, Region, WorkUnit};

use std::sync::Arc;

// ── GpuBackend ───────────────────────────────────────────────────────────────

use crate::backend::Backend;

pub struct GpuBackend;

impl Backend for GpuBackend {
    type Handle = GraphNodeHandle;
    type Buffer = Arc<crate::backend::gpu::buffer::GpuBuffer>;
}
