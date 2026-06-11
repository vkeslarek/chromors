pub mod gpu;
pub mod vips;

use std::sync::Arc;
use crate::error::Error;
use crate::work_unit::WorkUnit;
use crate::node::Node;
use crate::buffer::Buffer;

/// A Backend is the engine that runs a DAG.
/// It owns its context, its payload (VRAM vs CPU region), and how nodes are lowered.
pub trait Backend: Sized + Send + Sync + 'static {
    type Ctx: Send + Sync;
    type Payload: Send + Sync;
    type Builder;

    /// Walk the agnostic DAG, lower each node into a `Builder`, run it, return
    /// the result. GPU: emit one fused Slang module + dispatch. Vips: build a
    /// libvips demand-driven pipeline + sink the region. The output Kind is
    /// carried by the root `Node` (and the returned `Buffer`'s `spec`), so it
    /// is not a type parameter here.
    ///
    /// `ctx` is an `&Arc` (not `&Self::Ctx`) so a backend can clone it into its
    /// builder — a GPU source's `lower` fetches+uploads with it, keeping the
    /// materializer free of any `Node::Source` special-case.
    fn materialize(ctx: &Arc<Self::Ctx>, root: &Arc<Node<Self>>, wu: &WorkUnit)
        -> Result<Buffer<Self>, Error>;
}
