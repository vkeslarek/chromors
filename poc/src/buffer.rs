use std::sync::Arc;
use crate::backend::Backend;
use crate::kind::AnyKind;

/// Backend-resident payload + the Kind that tags it.
/// Data is always resident on the backend inside the model.
pub struct Buffer<B: Backend> {
    pub payload: Arc<B::Payload>,
    pub spec: Arc<dyn AnyKind>,
}
