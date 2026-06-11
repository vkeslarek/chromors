pub mod decode;
pub mod params;
pub mod handle;

use std::sync::Arc;
use crate::backend::Backend;
use crate::error::Error;
use crate::pixel::PixelFormat;
use crate::buffer::Buffer;
use crate::work_unit::WorkUnit;
use crate::node::Node;

pub use handle::{GpsInfo, LensInfo, RawFrame, RawHandle, RawMeta};
pub use params::{
    CameraMatrixMode, HighlightMode, ImageFormat, InterpolationQuality, IntoRawEnum,
    OutputColorSpace, ProcessWarnings, RawDecodeParams, ThumbnailFormat, WhiteBalanceSource,
    output_flags,
};

// ── Backend marker ─────────────────────────────────────────────────────────────

pub struct RawBackend;

pub struct RawBuilder {
    outputs: std::collections::HashMap<usize, Arc<RawHandle>>,
    current: Option<usize>,
}

impl Default for RawBuilder {
    fn default() -> Self {
        Self { outputs: std::collections::HashMap::new(), current: None }
    }
}

impl RawBuilder {
    pub fn enter(&mut self, node: usize) {
        self.current = Some(node);
    }
    pub fn input(&self, src: &Arc<Node<RawBackend>>) -> Arc<RawHandle> {
        let k = Arc::as_ptr(src) as *const () as usize;
        self.outputs.get(&k).expect("input lowered before its consumer").clone()
    }
    pub fn emit(&mut self, handle: Arc<RawHandle>) {
        let k = self.current.expect("emit() called outside a lower()");
        self.outputs.insert(k, handle);
    }
    fn take(&mut self, node: usize) -> Option<Arc<RawHandle>> {
        self.outputs.remove(&node)
    }
}

impl Backend for RawBackend {
    type Ctx = ();
    type Payload = RawHandle;
    type Builder = RawBuilder;

    fn materialize(
        _ctx: &Arc<Self::Ctx>,
        root: &Arc<Node<Self>>,
        wu: &WorkUnit,
    ) -> Result<Buffer<Self>, Error> {
        let mut walk = crate::node::GraphWalk::new(root);
        walk.demand(wu);

        let mut builder = RawBuilder::default();
        walk.lower(|node, _n_wu| {
            let k = Arc::as_ptr(node) as *const () as usize;
            builder.enter(k);
            node.lower(&mut builder);
        });

        let k = Arc::as_ptr(root) as *const () as usize;
        let handle = builder
            .take(k)
            .ok_or_else(|| Error::Raw("root node produced no handle".into()))?;

        let spec = root.output_kind();

        Ok(Buffer {
            payload: handle,
            spec,
        })
    }
}

// Re-export backend operations and decode implementation.

