use std::ffi::CString;

use crate::Error;
use crate::ffi;
use crate::vips_error;

/// A libvips output target (file or memory buffer).
///
/// Targets are the exit point for writing image data from Vips. They can be
/// created to write to a file path or to capture output in memory.
pub struct VipsTarget {
    pub(crate) ptr: *mut ffi::VipsTarget,
}

unsafe impl Send for VipsTarget {}
unsafe impl Sync for VipsTarget {}

impl Clone for VipsTarget {
    fn clone(&self) -> Self {
        unsafe {
            ffi::g_object_ref(self.ptr as ffi::gpointer);
        }
        VipsTarget { ptr: self.ptr }
    }
}

impl Drop for VipsTarget {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::g_object_unref(self.ptr as ffi::gpointer);
            }
        }
    }
}

impl VipsTarget {
    /// Creates a target that writes to a file.
    pub fn new_to_file(filename: &str) -> Result<VipsTarget, Error> {
        let c = CString::new(filename).map_err(|_| Error::Vips("invalid filename".into()))?;
        let ptr = unsafe { ffi::vips_target_new_to_file(c.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::Vips(vips_error()));
        }
        Ok(VipsTarget { ptr })
    }

    /// Creates a target that captures output to an in-memory buffer.
    /// The result can be retrieved via `vips_target_blob()` or `vips_target_steal()`.
    pub fn new_to_memory() -> VipsTarget {
        let ptr = unsafe { ffi::vips_target_new_to_memory() };
        VipsTarget { ptr }
    }
}
