use std::collections::HashMap;
use std::ffi::CStr;
use std::ptr;
use std::sync::Arc;

use crate::AnyKind;
use crate::Backend;
use crate::Buffer;
use crate::Builder;
use crate::Error;
use crate::Node;
use crate::WorkUnit;
use crate::ffi;

pub fn null() -> *const std::ffi::c_void {
    ptr::null()
}

pub fn ensure_init() {
    static VIPS_INIT: std::sync::Once = std::sync::Once::new();
    VIPS_INIT.call_once(|| {
        let name = std::ffi::CString::new("chromors").unwrap();
        let rc = unsafe { crate::ffi::vips_init(name.as_ptr()) };
        if rc != 0 {
            panic!("vips_init failed: {}", vips_error());
        }
    });
}

pub fn vips_error() -> String {
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
    fn into_vips(self) -> i32;
}

pub trait IntoVipsName {
    fn into_vips_name(self) -> &'static str;
}

pub trait IntoVipsOption {
    fn to_vips_options(&self) -> String;
}

pub trait IntoVipsBandFormat {
    fn into_vips_band_format(self) -> i32;
}

pub trait FromVipsBandFormat: Sized {
    fn from_vips_band_format(raw: i32, bands: i32) -> Self;
}

/// Plain marker struct for the libvips backend.
pub struct VipsBackend;

/// Opaque handle wrapping a `VipsImage` GObject pointer.
pub struct VipsHandle {
    pub ptr: *mut ffi::VipsImage,
}

impl VipsHandle {
    pub fn ptr(&self) -> *mut ffi::VipsImage {
        self.ptr
    }
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
pub trait VipsBand: crate::Kind {
    fn band_format(&self) -> i32;
}

type NodeKey = crate::NodeId;

fn node_key(node: &Arc<Node<VipsBackend>>) -> NodeKey {
    crate::NodeId::of(node)
}

/// Lowering accumulator for vips — node-keyed handle map.
pub struct VipsBuilder {
    outputs: HashMap<NodeKey, VipsHandle>,
    current: Option<NodeKey>,
    current_wu: Option<WorkUnit>,
    error: Option<Error>,
}

impl Default for VipsBuilder {
    fn default() -> Self {
        Self {
            outputs: HashMap::new(),
            current: None,
            current_wu: None,
            error: None,
        }
    }
}

impl VipsBuilder {
    pub fn input(&self, src: &Arc<Node<VipsBackend>>) -> VipsHandle {
        self.outputs
            .get(&node_key(src))
            .expect("input lowered before its consumer")
            .clone()
    }
    pub fn emit(&mut self, handle: VipsHandle) {
        let k = self.current.expect("emit() called outside a lower()");
        self.outputs.insert(k, handle);
    }
    pub fn fail(&mut self, e: Error) {
        if self.error.is_none() {
            self.error = Some(e);
        }
    }
    pub fn take_error(&mut self) -> Option<Error> {
        self.error.take()
    }
    fn take(&mut self, node: NodeKey) -> Option<VipsHandle> {
        self.outputs.remove(&node)
    }
    pub fn wu(&self) -> &WorkUnit {
        self.current_wu
            .as_ref()
            .expect("VipsBuilder::wu called outside a lower()")
    }
}

pub fn image_from_memory(
    bytes: &[u8],
    w: i32,
    h: i32,
    layout: chromors_core::pixel::PixelLayout,
) -> Result<Arc<VipsHandle>, Error> {
    ensure_init();
    let vips_format = layout.storage.into_vips_band_format();
    let bands = layout.channel_count() as i32;
    let ptr = unsafe {
        ffi::vips_image_new_from_memory_copy(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len(),
            w,
            h,
            bands,
            vips_format,
        )
    };
    if ptr.is_null() {
        return Err(Error::Vips(vips_error()));
    }
    Ok(Arc::new(VipsHandle { ptr }))
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
        if let Some(e) = self.take_error() {
            return Err(e);
        }
        let handle = self
            .take(root)
            .ok_or_else(|| Error::Vips("root node produced no handle".into()))?;

        Ok(Buffer {
            payload: Arc::new(handle),
            spec,
        })
    }
}
