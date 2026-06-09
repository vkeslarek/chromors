use std::ffi::CString;

use crate::backend::vips::vips_error;
use crate::error::Error;
use crate::libvips_ffi as ffi;

pub struct Source {
    pub(crate) ptr: *mut ffi::VipsSource,
}

unsafe impl Send for Source {}
unsafe impl Sync for Source {}

impl Clone for Source {
    fn clone(&self) -> Self {
        unsafe {
            ffi::g_object_ref(self.ptr as ffi::gpointer);
        }
        Source { ptr: self.ptr }
    }
}

impl Drop for Source {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::g_object_unref(self.ptr as ffi::gpointer);
            }
        }
    }
}

impl Source {
    pub fn new_from_file(filename: &str) -> Result<Source, Error> {
        let c = CString::new(filename).map_err(|_| Error::Vips("invalid filename".into()))?;
        let ptr = unsafe { ffi::vips_source_new_from_file(c.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::Vips(vips_error()));
        }
        Ok(Source { ptr })
    }

    pub fn new_from_memory(data: &[u8]) -> Result<Source, Error> {
        // `vips_source_new_from_memory` would reference `data` without copying,
        // so the source (and anything built on it) would dangle once `data`
        // drops. `vips_blob_copy` makes a vips-owned copy whose lifetime is tied
        // to the source's refcount, so the result is sound regardless.
        unsafe {
            let blob = ffi::vips_blob_copy(data.as_ptr() as *const _, data.len());
            if blob.is_null() {
                return Err(Error::Vips(vips_error()));
            }
            let ptr = ffi::vips_source_new_from_blob(blob);
            ffi::vips_area_unref(blob as *mut ffi::VipsArea);
            if ptr.is_null() {
                return Err(Error::Vips(vips_error()));
            }
            Ok(Source { ptr })
        }
    }
}
