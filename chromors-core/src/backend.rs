use crate::buffer::Buffer;
use crate::error::Error;
use crate::kind::AnyKind;
use crate::node::NodeId;
use crate::work_unit::WorkUnit;
use std::sync::Arc;

pub trait Backend: Sized + Send + Sync + 'static {
    type Ctx: Send + Sync;
    type Payload: Send + Sync;
    type Builder: Builder<Self>;

    fn materialize(
        ctx: &Arc<Self::Ctx>,
        root: &Arc<crate::node::Node<Self>>,
        wu: &WorkUnit,
    ) -> Result<Buffer<Self>, Error> {
        crate::node::materialize(ctx, root, wu)
    }
}

pub trait Builder<B: Backend>: Sized {
    fn new(ctx: Arc<B::Ctx>) -> Self;
    fn enter(&mut self, node: NodeId, inputs: &[NodeId], wu: &WorkUnit);
    fn finish(
        self,
        root: NodeId,
        spec: Arc<dyn AnyKind>,
        root_wu: &WorkUnit,
    ) -> Result<Buffer<B>, Error>;
}
