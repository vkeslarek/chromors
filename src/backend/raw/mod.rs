pub mod decode;
pub mod handle;
pub mod params;

use crate::backend::{Backend, Builder};
use crate::buffer::Buffer;
use crate::error::Error;
use crate::kind::AnyKind;
use crate::node::{Node, NodeId};
use crate::work_unit::WorkUnit;
use std::sync::Arc;

pub use handle::{GpsInfo, LensInfo, RawFrame, RawHandle, RawMeta};
pub use params::{
    CameraMatrixMode, HighlightMode, ImageFormat, InterpolationQuality, IntoRawEnum,
    OutputColorSpace, ProcessWarnings, RawDecodeParams, ThumbnailFormat, WhiteBalanceSource,
    output_flags,
};

/// Marker struct for the LibRaw camera RAW decoding backend.
///
/// This backend decodes RAW files into RGB(A) pixel buffers using LibRaw.
/// It is a leaf-only backend — operations on raw images require converting
/// to a pixel-processing backend (Vips or GPU) first.
pub struct RawBackend;

/// Lowering accumulator for the RAW backend.
///
/// Node-keyed handle map — each lowered node registers its produced `RawHandle`;
/// a consumer looks its inputs up by edge identity.
pub struct RawBuilder {
    outputs: std::collections::HashMap<NodeId, Arc<RawHandle>>,
    current: Option<NodeId>,
}

impl Default for RawBuilder {
    fn default() -> Self {
        Self {
            outputs: std::collections::HashMap::new(),
            current: None,
        }
    }
}

impl RawBuilder {
    /// Look up an already-lowered upstream input's raw handle.
    /// Post-order lowering guarantees it is present.
    pub fn input(&self, src: &Arc<Node<RawBackend>>) -> Arc<RawHandle> {
        self.outputs
            .get(&NodeId::of(src))
            .expect("input lowered before its consumer")
            .clone()
    }

    /// Register the raw handle this node produced.
    pub fn emit(&mut self, handle: Arc<RawHandle>) {
        let k = self.current.expect("emit() called outside a lower()");
        self.outputs.insert(k, handle);
    }

    fn take(&mut self, node: NodeId) -> Option<Arc<RawHandle>> {
        self.outputs.remove(&node)
    }
}

impl Backend for RawBackend {
    type Ctx = ();
    type Payload = RawHandle;
    type Builder = RawBuilder;
}

impl Builder<RawBackend> for RawBuilder {
    fn new(_ctx: Arc<()>) -> Self {
        Self::default()
    }

    fn enter(&mut self, node: NodeId, _inputs: &[NodeId], _wu: &WorkUnit) {
        self.current = Some(node);
    }

    fn finish(
        mut self,
        root: NodeId,
        spec: Arc<dyn AnyKind>,
        _root_wu: &WorkUnit,
    ) -> Result<Buffer<RawBackend>, Error> {
        let handle = self
            .take(root)
            .ok_or_else(|| Error::Raw("root node produced no handle".into()))?;

        Ok(Buffer {
            payload: handle,
            spec,
        })
    }
}

// Re-export backend operations and decode implementation.
