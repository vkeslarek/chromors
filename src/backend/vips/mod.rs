pub mod custom;
pub mod gobject;
pub mod working;

use std::collections::HashMap;
use std::ffi::CStr;
use std::ptr;
use std::sync::Arc;

use crate::backend::{Backend, Builder};
use crate::buffer::Buffer;
use crate::error::Error;
use crate::ffi;
use crate::kind::AnyKind;
use crate::node::Node;
use crate::work_unit::WorkUnit;

pub(crate) fn null() -> *const std::ffi::c_void {
    ptr::null()
}

/// Initialise libvips exactly once, before any other vips call.
///
/// libvips registers its GObject type system on first use; that lazy
/// registration is **not** thread-safe, so two threads racing their first vips
/// call corrupt the type system ("cannot retrieve class for invalid type" →
/// SIGSEGV). `vips_init` is otherwise safe to run once up front, and `Once`
/// gives every other thread the happens-before it needs before touching vips.
/// Every vips entry point (source construction, `VipsGObject::new`) calls this.
pub(crate) fn ensure_init() {
    static VIPS_INIT: std::sync::Once = std::sync::Once::new();
    VIPS_INIT.call_once(|| {
        let name = std::ffi::CString::new("chromors").unwrap();
        let rc = unsafe { crate::ffi::vips_init(name.as_ptr()) };
        if rc != 0 {
            panic!("vips_init failed: {}", vips_error());
        }
    });
}

pub(crate) fn vips_error() -> String {
    unsafe {
        let buf = crate::ffi::vips_error_buffer();
        let s = if buf.is_null() {
            String::from("unknown error")
        } else {
            CStr::from_ptr(buf).to_string_lossy().into_owned()
        };
        crate::ffi::vips_error_clear();
        s
    }
}

pub trait IntoVipsEnum {
    /// Convert this enum variant to its libvips integer representation.
    fn into_vips(self) -> i32;
}

pub trait IntoVipsName {
    /// Convert to the libvips string name (e.g. interpolation method nickname).
    fn into_vips_name(self) -> &'static str;
}

pub trait IntoVipsOption {
    /// Serialize into a libvips option string (key=value pairs).
    fn to_vips_options(&self) -> String;
}

pub trait IntoVipsBandFormat {
    /// Map this type to a `VipsBandFormat` enum integer.
    fn into_vips_band_format(self) -> i32;
}

pub trait FromVipsBandFormat: Sized {
    /// Reconstruct from a `VipsBandFormat` integer and band count.
    fn from_vips_band_format(raw: i32, bands: i32) -> Self;
}

/// Plain marker struct for the libvips backend.
pub struct VipsBackend;

/// Opaque handle wrapping a `VipsImage` GObject pointer.
pub struct VipsHandle {
    pub(crate) ptr: *mut ffi::VipsImage,
}

unsafe impl Send for VipsHandle {}
unsafe impl Sync for VipsHandle {}

impl Clone for VipsHandle {
    fn clone(&self) -> Self {
        unsafe { ffi::g_object_ref(self.ptr as ffi::gpointer) };
        VipsHandle { ptr: self.ptr }
    }
}

impl Drop for VipsHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::g_object_unref(self.ptr as ffi::gpointer) };
        }
    }
}

/// Per-backend Kind capability for libvips — symmetric to the GPU's `GpuView`.
/// A Kind that can live on the vips backend maps to a libvips band format; a
/// Kind that can't (e.g. a GPU-only point list) simply doesn't implement it, so
/// `Data<ThatKind, VipsBackend>` won't type-check. Keeps `AnyKind` agnostic.
pub trait VipsBand: crate::kind::Kind {
    /// The `VipsBandFormat` enum value this Kind decodes to.
    fn band_format(&self) -> i32;
}

type NodeKey = crate::node::NodeId;

fn node_key(node: &Arc<Node<VipsBackend>>) -> NodeKey {
    crate::node::NodeId::of(node)
}

/// Lowering accumulator for vips — node-keyed handle map (symmetric to the GPU
/// materializer), not a fragile stack. Each lowered node registers its produced
/// `VipsHandle`; a consumer looks its inputs up by edge identity. libvips fuses
/// the resulting operation chain itself (demand-driven), so there is no manual
/// fusion/params/view here — that's GPU vocabulary.
pub struct VipsBuilder {
    outputs: HashMap<NodeKey, VipsHandle>,
    current: Option<NodeKey>,
    current_wu: Option<WorkUnit>,
}

impl Default for VipsBuilder {
    fn default() -> Self {
        Self {
            outputs: HashMap::new(),
            current: None,
            current_wu: None,
        }
    }
}

impl VipsBuilder {
    /// Look up an already-lowered upstream input's handle. Post-order
    /// lowering guarantees it is present.
    pub fn input(&self, src: &Arc<Node<VipsBackend>>) -> VipsHandle {
        self.outputs
            .get(&node_key(src))
            .expect("input lowered before its consumer")
            .clone()
    }
    /// Register the handle this node produced.
    pub fn emit(&mut self, handle: VipsHandle) {
        let k = self.current.expect("emit() called outside a lower()");
        self.outputs.insert(k, handle);
    }
    fn take(&mut self, node: NodeKey) -> Option<VipsHandle> {
        self.outputs.remove(&node)
    }
    /// The resolved WorkUnit of the node being lowered.
    pub fn wu(&self) -> &WorkUnit {
        self.current_wu
            .as_ref()
            .expect("VipsBuilder::wu called outside a lower()")
    }
}

impl Backend for VipsBackend {
    type Ctx = ();
    type Payload = VipsHandle;
    type Builder = VipsBuilder;
}

impl Builder<VipsBackend> for VipsBuilder {
    fn new(_ctx: Arc<()>) -> Self {
        Self::default()
    }

    fn enter(&mut self, node: NodeKey, _inputs: &[NodeKey], wu: &WorkUnit) {
        self.current = Some(node);
        self.current_wu = Some(wu.clone());
    }

    fn finish(
        mut self,
        root: NodeKey,
        spec: Arc<dyn AnyKind>,
        _root_wu: &WorkUnit,
    ) -> Result<Buffer<VipsBackend>, Error> {
        let handle = self
            .take(root)
            .ok_or_else(|| Error::Vips("root node produced no handle".into()))?;

        Ok(Buffer {
            payload: Arc::new(handle),
            spec,
        })
    }
}
