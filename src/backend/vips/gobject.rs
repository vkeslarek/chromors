use crate::error::Error;
use crate::ffi;
use std::ffi::{CStr, CString};

thread_local! {
    /// Per-thread libvips cleanup guard.
    ///
    /// libvips stashes per-thread state (via `g_private`) the first time a
    /// thread runs an operation. A host thread that exits **without** calling
    /// `vips_thread_shutdown()` leaves that state dangling and segfaults at
    /// teardown — flaky, and only for ops that spin up the vips threadpool
    /// (reductions, generators, array combines); header-only lazy ops never
    /// trigger it. Touching this thread-local in [`VipsGObject::build`]
    /// registers the guard, whose `Drop` runs `vips_thread_shutdown()` when the
    /// thread exits.
    static VIPS_THREAD: VipsThreadGuard = const { VipsThreadGuard };
}

struct VipsThreadGuard;

impl Drop for VipsThreadGuard {
    fn drop(&mut self) {
        unsafe { ffi::vips_thread_shutdown() };
    }
}

const G_TYPE_DOUBLE: ffi::GType = (15 << 2) as ffi::GType;
const G_TYPE_INT: ffi::GType = (6 << 2) as ffi::GType;
const G_TYPE_STRING: ffi::GType = (16 << 2) as ffi::GType;
const G_TYPE_BOOL: ffi::GType = (5 << 2) as ffi::GType;

/// Extracts the typed result from a built `VipsGObject`.
pub trait Runner: Sized {
    fn run(op: VipsGObject) -> Result<Self, Error>;
}

use crate::backend::vips::VipsHandle;

impl Runner for VipsHandle {
    fn run(op: VipsGObject) -> Result<VipsHandle, Error> {
        op.run()
    }
}

impl Runner for f64 {
    fn run(op: VipsGObject) -> Result<f64, Error> {
        op.run_scalar()
    }
}

impl Runner for () {
    fn run(op: VipsGObject) -> Result<(), Error> {
        op.run_no_output()
    }
}

pub struct VipsGObject {
    pub(crate) ptr: *mut ffi::VipsOperation,
}

impl Clone for VipsGObject {
    fn clone(&self) -> Self {
        unsafe {
            ffi::g_object_ref(self.ptr as ffi::gpointer);
        }
        VipsGObject { ptr: self.ptr }
    }
}

impl Drop for VipsGObject {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::g_object_unref(self.ptr as ffi::gpointer);
            }
        }
    }
}

impl VipsGObject {
    pub(crate) fn new(name: &[u8]) -> Result<Self, Error> {
        crate::backend::vips::ensure_init();
        let c_name =
            CStr::from_bytes_with_nul(name).map_err(|_| Error::Vips("bad op name".into()))?;
        let ptr = unsafe { ffi::vips_operation_new(c_name.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(VipsGObject { ptr })
    }

    pub(crate) fn set_image(&mut self, name: &str, image: *mut ffi::VipsImage) {
        set_gvalue(self, name, |val| unsafe {
            ffi::g_value_init(val, ffi::vips_image_get_type());
            ffi::g_value_set_object(val, image as ffi::gpointer);
        });
    }

    pub(crate) fn set_double(&mut self, name: &str, value: f64) {
        set_gvalue(self, name, |val| unsafe {
            ffi::g_value_init(val, G_TYPE_DOUBLE);
            ffi::g_value_set_double(val, value);
        });
    }

    pub(crate) fn set_int(&mut self, name: &str, value: i32) {
        set_gvalue(self, name, |val| unsafe {
            ffi::g_value_init(val, G_TYPE_INT);
            ffi::g_value_set_int(val, value);
        });
    }

    pub(crate) fn set_string(&mut self, name: &str, value: &str) {
        let c_val = CString::new(value).unwrap();
        set_gvalue(self, name, |val| unsafe {
            ffi::g_value_init(val, G_TYPE_STRING);
            ffi::g_value_set_string(val, c_val.as_ptr());
        });
    }

    pub(crate) fn set_bool(&mut self, name: &str, value: bool) {
        set_gvalue(self, name, |val| unsafe {
            ffi::g_value_init(val, G_TYPE_BOOL);
            ffi::g_value_set_boolean(val, value as ffi::gboolean);
        });
    }

    pub(crate) fn set_array_double(&mut self, name: &str, data: &[f64]) {
        set_gvalue(self, name, |val| unsafe {
            ffi::g_value_init(val, ffi::vips_array_double_get_type());
            ffi::vips_value_set_array_double(val, data.as_ptr(), data.len() as _);
        });
    }

    pub(crate) fn set_array_int(&mut self, name: &str, data: &[i32]) {
        set_gvalue(self, name, |val| unsafe {
            ffi::g_value_init(val, ffi::vips_array_int_get_type());
            ffi::vips_value_set_array_int(val, data.as_ptr(), data.len() as _);
        });
    }

    pub(crate) fn set_blob(&mut self, name: &str, data: &[u8]) {
        set_gvalue(self, name, |val| unsafe {
            ffi::g_value_init(val, ffi::vips_blob_get_type());
            let blob = ffi::vips_blob_new(None, data.as_ptr() as *const _, data.len());
            ffi::g_value_set_boxed(val, blob as ffi::gpointer);
            ffi::vips_area_unref(blob as *mut ffi::VipsArea);
        });
    }

    pub(crate) fn set_object(&mut self, name: &str, obj: ffi::gpointer, gtype: ffi::GType) {
        set_gvalue(self, name, |val| unsafe {
            ffi::g_value_init(val, gtype);
            ffi::g_value_set_object(val, obj);
        });
    }

    pub(crate) fn set_array_image(&mut self, name: &str, images: &[*mut ffi::VipsImage]) {
        set_gvalue(self, name, |val| unsafe {
            ffi::g_value_init(val, ffi::vips_array_image_get_type());
            let arr = ffi::vips_array_image_new(images.as_ptr() as *mut _, images.len() as i32);
            ffi::g_value_set_boxed(val, arr as ffi::gpointer);
            ffi::vips_area_unref(arr as *mut ffi::VipsArea);
        });
    }

    pub(crate) fn build(&mut self) -> Result<(), Error> {
        // Register the per-thread shutdown guard (see `VIPS_THREAD`). `build` is
        // the single choke point every op (and custom `Runner`s) goes through.
        VIPS_THREAD.with(|_| {});
        if unsafe { ffi::vips_cache_operation_buildp(&mut self.ptr) } != 0 {
            unsafe {
                ffi::vips_object_unref_outputs(self.ptr as *mut ffi::VipsObject);
            }
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(())
    }

    fn run_body(self) -> Result<VipsHandle, Error> {
        unsafe {
            let mut op = self;
            op.build()?;
            let out_ptr = get_output_image(op.ptr);
            ffi::vips_object_unref_outputs(op.ptr as *mut ffi::VipsObject);
            if out_ptr.is_null() {
                return Err(Error::Vips(crate::backend::vips::vips_error()));
            }
            Ok(VipsHandle { ptr: out_ptr })
        }
    }

    pub fn run(self) -> Result<VipsHandle, Error> {
        self.run_body()
    }

    /// Run a generator/sink op. Same exclusive lock as [`run`] — kept as a
    /// distinct name so the generator call sites read clearly.
    pub(crate) fn run_generator(self) -> Result<VipsHandle, Error> {
        self.run_body()
    }

    pub(crate) fn run_scalar(mut self) -> Result<f64, Error> {
        unsafe {
            self.build()?;
            let mut val = std::mem::MaybeUninit::<ffi::GValue>::zeroed();
            ffi::g_value_init(val.as_mut_ptr(), G_TYPE_DOUBLE);
            let out_prop = c"out";
            ffi::g_object_get_property(
                self.ptr as *mut ffi::GObject,
                out_prop.as_ptr(),
                val.as_mut_ptr(),
            );
            let result = ffi::g_value_get_double(val.as_ptr());
            ffi::g_value_unset(val.as_mut_ptr());
            ffi::vips_object_unref_outputs(self.ptr as *mut ffi::VipsObject);
            Ok(result)
        }
    }

    pub(crate) fn run_no_output(mut self) -> Result<(), Error> {
        self.build()?;
        unsafe {
            ffi::vips_object_unref_outputs(self.ptr as *mut ffi::VipsObject);
        }
        Ok(())
    }

    /// # Safety
    /// `self` must already have been built successfully.
    pub(crate) unsafe fn output_int(&self, prop: &str) -> i32 {
        unsafe { get_int_prop(self.ptr, prop) }
    }

    /// # Safety
    /// `self` must already have been built successfully.
    pub(crate) unsafe fn output_double(&self, prop: &str) -> f64 {
        unsafe {
            let mut val = std::mem::MaybeUninit::<ffi::GValue>::zeroed();
            ffi::g_value_init(val.as_mut_ptr(), G_TYPE_DOUBLE);
            let c = CString::new(prop).unwrap();
            ffi::g_object_get_property(self.ptr as *mut ffi::GObject, c.as_ptr(), val.as_mut_ptr());
            let r = ffi::g_value_get_double(val.as_ptr());
            ffi::g_value_unset(val.as_mut_ptr());
            r
        }
    }

    /// # Safety
    /// `self` must already have been built successfully.
    pub(crate) unsafe fn output_bool(&self, prop: &str) -> bool {
        unsafe {
            let mut val = std::mem::MaybeUninit::<ffi::GValue>::zeroed();
            ffi::g_value_init(val.as_mut_ptr(), G_TYPE_BOOL);
            let c = CString::new(prop).unwrap();
            ffi::g_object_get_property(self.ptr as *mut ffi::GObject, c.as_ptr(), val.as_mut_ptr());
            let r = ffi::g_value_get_boolean(val.as_ptr()) != 0;
            ffi::g_value_unset(val.as_mut_ptr());
            r
        }
    }

    /// Reads a named output image property, returning an owned `Image2D<VipsBackend>`.
    ///
    /// # Safety
    /// `self` must already have been built successfully.
    pub(crate) unsafe fn output_image(&self, prop: &str) -> Result<VipsHandle, Error> {
        unsafe {
            let mut val = std::mem::MaybeUninit::<ffi::GValue>::zeroed();
            ffi::g_value_init(val.as_mut_ptr(), ffi::vips_image_get_type());
            let c = CString::new(prop).unwrap();
            ffi::g_object_get_property(self.ptr as *mut ffi::GObject, c.as_ptr(), val.as_mut_ptr());
            let ptr = ffi::g_value_dup_object(val.as_ptr()) as *mut ffi::VipsImage;
            ffi::g_value_unset(val.as_mut_ptr());
            if ptr.is_null() {
                return Err(Error::Vips(crate::backend::vips::vips_error()));
            }
            Ok(VipsHandle { ptr })
        }
    }

    /// # Safety
    /// `self` must already have been built successfully.
    pub(crate) unsafe fn output_array_double(&self, prop: &str) -> Vec<f64> {
        unsafe {
            let mut val = std::mem::MaybeUninit::<ffi::GValue>::zeroed();
            ffi::g_value_init(val.as_mut_ptr(), ffi::vips_array_double_get_type());
            let c = CString::new(prop).unwrap();
            ffi::g_object_get_property(self.ptr as *mut ffi::GObject, c.as_ptr(), val.as_mut_ptr());
            let mut n: std::os::raw::c_int = 0;
            let data = ffi::vips_value_get_array_double(val.as_ptr(), &mut n);
            let v = if data.is_null() || n <= 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(data, n as usize).to_vec()
            };
            ffi::g_value_unset(val.as_mut_ptr());
            v
        }
    }
}

fn set_gvalue(op: &mut VipsGObject, name: &str, init: impl FnOnce(*mut ffi::GValue)) {
    unsafe {
        let mut val = std::mem::MaybeUninit::<ffi::GValue>::zeroed();
        init(val.as_mut_ptr());
        let c_name = CString::new(name).unwrap();
        ffi::g_object_set_property(op.ptr as *mut ffi::GObject, c_name.as_ptr(), val.as_ptr());
        ffi::g_value_unset(val.as_mut_ptr());
    }
}

unsafe fn get_output_image(op: *mut ffi::VipsOperation) -> *mut ffi::VipsImage {
    unsafe {
        let mut val = std::mem::MaybeUninit::<ffi::GValue>::zeroed();
        ffi::g_value_init(val.as_mut_ptr(), ffi::vips_image_get_type());
        let prop = c"out";
        ffi::g_object_get_property(op as *mut ffi::GObject, prop.as_ptr(), val.as_mut_ptr());
        let ptr = ffi::g_value_dup_object(val.as_ptr()) as *mut ffi::VipsImage;
        ffi::g_value_unset(val.as_mut_ptr());
        ptr
    }
}

unsafe fn get_int_prop(op: *mut ffi::VipsOperation, prop: &str) -> i32 {
    unsafe {
        let mut val = std::mem::MaybeUninit::<ffi::GValue>::zeroed();
        ffi::g_value_init(val.as_mut_ptr(), G_TYPE_INT);
        let c_prop = CString::new(prop).unwrap();
        ffi::g_object_get_property(op as *mut ffi::GObject, c_prop.as_ptr(), val.as_mut_ptr());
        let result = ffi::g_value_get_int(val.as_ptr());
        ffi::g_value_unset(val.as_mut_ptr());
        result
    }
}
