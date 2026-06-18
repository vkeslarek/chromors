pub use chromors_core::*;

pub mod handle;
pub use handle::{VelloHandle, VelloScene};

use std::sync::Arc;
use self::handle::VelloHandle as VelloHandleAlias;

/// Marker struct for the Vello vector graphics rasterization backend.
///
/// This backend converts vector scenes (SVG, etc.) into rasterized pixel
/// buffers. It is a leaf-only backend — operations on vector graphics
/// require converting to a pixel-processing backend (Vips or GPU) first.
pub struct VelloBackend;

/// Lowering accumulator for the Vello backend.
///
/// Node-keyed handle map — each lowered node registers its produced `VelloHandle`;
/// a consumer looks its inputs up by edge identity.
pub struct VelloBuilder {
    outputs: std::collections::HashMap<NodeId, Arc<VelloHandleAlias>>,
    current: Option<NodeId>,
}

impl Default for VelloBuilder {
    fn default() -> Self {
        Self {
            outputs: std::collections::HashMap::new(),
            current: None,
        }
    }
}

impl VelloBuilder {
    /// Look up an already-lowered upstream input's vello handle.
    /// Post-order lowering guarantees it is present.
    pub fn input(&self, src: &Arc<Node<VelloBackend>>) -> Arc<VelloHandleAlias> {
        self.outputs
            .get(&NodeId::of(src))
            .expect("input lowered before its consumer")
            .clone()
    }

    /// Register the vello handle this node produced.
    pub fn emit(&mut self, handle: Arc<VelloHandleAlias>) {
        let k = self.current.expect("emit() called outside a lower()");
        self.outputs.insert(k, handle);
    }

    fn take(&mut self, node: NodeId) -> Option<Arc<VelloHandleAlias>> {
        self.outputs.remove(&node)
    }
}

impl Backend for VelloBackend {
    type Ctx = ();
    type Payload = VelloHandleAlias;
    type Builder = VelloBuilder;
}

impl Builder<VelloBackend> for VelloBuilder {
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
    ) -> Result<Buffer<VelloBackend>, Error> {
        let handle = self
            .take(root)
            .ok_or_else(|| Error::Backend("root node produced no handle".into()))?;

        Ok(Buffer {
            payload: handle,
            spec,
        })
    }
}
