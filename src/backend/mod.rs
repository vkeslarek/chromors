pub mod gpu;
pub mod vips;
pub mod raw;
pub mod vello;

use std::sync::Arc;
use crate::error::Error;
use crate::work_unit::WorkUnit;
use crate::kind::AnyKind;
use crate::node::NodeId;
use crate::buffer::Buffer;

/// A Backend is the engine that runs a DAG.
/// It owns its context, its payload (VRAM vs CPU region), and how nodes are lowered.
pub trait Backend: Sized + Send + Sync + 'static {
    type Ctx: Send + Sync;
    type Payload: Send + Sync;
    type Builder: Builder<Self>;

    /// Evaluate the requested WorkUnit into a backend-resident Buffer.
    /// Defaults to the standard demand→lower→finish walk (`node::materialize`).
    /// GPU overrides this to inject pass splitting when a fused pass would
    /// exceed device limits (binding count, buffer size). Other backends
    /// inherit the default.
    fn materialize(
        ctx: &Arc<Self::Ctx>,
        root: &Arc<crate::node::Node<Self>>,
        wu: &WorkUnit,
    ) -> Result<Buffer<Self>, Error> {
        crate::node::materialize(ctx, root, wu)
    }
}

/// What a backend accumulates during the lower walk and how it finishes.
/// The core (`node::materialize`) owns the walk; a backend only says what
/// happens *per node* (`enter`, then `node.lower(&mut builder)`) and *at the
/// end* (`finish`).
pub trait Builder<B: Backend>: Sized {
    fn new(ctx: Arc<B::Ctx>) -> Self;
    /// Announce the node about to lower: its id, its input ids, its resolved unit.
    fn enter(&mut self, node: NodeId, inputs: &[NodeId], wu: &WorkUnit);
    /// Run the pass and produce the root's buffer.
    fn finish(self, root: NodeId, spec: Arc<dyn AnyKind>, root_wu: &WorkUnit) -> Result<Buffer<B>, Error>;
}
