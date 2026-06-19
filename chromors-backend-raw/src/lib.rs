pub use chromors_core::color::*;
pub use chromors_core::pixel::*;
pub use chromors_core::*;

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    warnings,
    clippy::all,
    unnecessary_transmutes,
    unsafe_op_in_unsafe_fn
)]
pub mod libraw_ffi {
    include!(concat!(env!("OUT_DIR"), "/libraw_ffi.rs"));
}

pub mod decode;
pub mod handle;
pub mod params;

pub use handle::{GpsInfo, LensInfo, RawFrame, RawHandle, RawMeta};
pub use params::{
    CameraMatrixMode, HighlightMode, ImageFormat, InterpolationQuality, IntoRawEnum,
    OutputColorSpace, ProcessWarnings, RawDecodeParams, ThumbnailFormat, WhiteBalanceSource,
    output_flags,
};

use self::handle::RawHandle as RawHandleAlias;
use std::sync::Arc;

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
    outputs: std::collections::HashMap<NodeId, Arc<RawHandleAlias>>,
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
    pub fn input(&self, src: &Arc<Node<RawBackend>>) -> Arc<RawHandleAlias> {
        self.outputs
            .get(&NodeId::of(src))
            .expect("input lowered before its consumer")
            .clone()
    }

    /// Register the raw handle this node produced.
    pub fn emit(&mut self, handle: Arc<RawHandleAlias>) {
        let k = self.current.expect("emit() called outside a lower()");
        self.outputs.insert(k, handle);
    }

    fn take(&mut self, node: NodeId) -> Option<Arc<RawHandleAlias>> {
        self.outputs.remove(&node)
    }
}

impl Backend for RawBackend {
    type Ctx = ();
    type Payload = RawHandleAlias;
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
