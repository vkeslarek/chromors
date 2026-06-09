use std::ffi::CString;

use crate::backend::vips::vips_error;
use crate::error::Error;
use crate::libvips_ffi as ffi;

pub struct Target {
    pub(crate) ptr: *mut ffi::VipsTarget,
}

unsafe impl Send for Target {}
unsafe impl Sync for Target {}

impl Clone for Target {
    fn clone(&self) -> Self {
        unsafe {
            ffi::g_object_ref(self.ptr as ffi::gpointer);
        }
        Target { ptr: self.ptr }
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::g_object_unref(self.ptr as ffi::gpointer);
            }
        }
    }
}

impl Target {
    pub fn new_to_file(filename: &str) -> Result<Target, Error> {
        let c = CString::new(filename).map_err(|_| Error::Vips("invalid filename".into()))?;
        let ptr = unsafe { ffi::vips_target_new_to_file(c.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::Vips(vips_error()));
        }
        Ok(Target { ptr })
    }

    pub fn new_to_memory() -> Target {
        let ptr = unsafe { ffi::vips_target_new_to_memory() };
        Target { ptr }
    }
}
