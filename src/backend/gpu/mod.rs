//! JIT-fused GPU execution.
//!
//! Each `Graph` accumulates nodes emitted by `GpuOperation::emit()`. When
//! compiled, the graph is topo-sorted and a single fused SPIR-V entry point
//! is generated per node.

pub mod arena;
pub mod buffer;
pub mod cache;
pub mod compile;
pub mod context;
pub mod data;
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
pub mod value;

pub use crate::geometry::Rect;
pub use buffer::{GpuBuffer, ImageBuffer};
pub use compile::{Compiled, CompiledShader, DispatchPass};
pub use context::GpuContext;
pub use context::RegionCache;
pub use graph::NodeEval;
pub use graph::{Graph, GraphNode, KernelSpec, NodeId, SourceNode};
pub use handle::{GpuImageHandle, GraphNodeHandle, Lod};
pub use op::{Decoder, DispatchGrid, Encoder, GpuOperation, OutputSpec, WorkUnit};
pub use source::{AnyGpuSource, GpuSource};
pub use target::GpuTarget;
pub use value::GraphValue;
pub use value::ValueKind;

use std::sync::Arc;

// ── GpuBackend ───────────────────────────────────────────────────────────────

use crate::backend::Backend;

pub struct GpuBackend;

impl Backend for GpuBackend {
    type Handle = GpuImageHandle;
    type Buffer = Arc<crate::backend::gpu::buffer::GpuBuffer>;
}
