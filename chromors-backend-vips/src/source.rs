use std::ffi::CString;

use crate::vips_error;
use crate::Error;
use crate::ffi as ffi;

/// A libvips input source (file or memory buffer).
///
/// Sources are the entry point for loading image data into Vips. They can be
/// created from a file path or an in-memory byte buffer.
pub struct VipsSource {
    pub(crate) ptr: *mut ffi::VipsSource,
}

unsafe impl Send for VipsSource {}
unsafe impl Sync for VipsSource {}

impl Clone for VipsSource {
    fn clone(&self) -> Self {
        unsafe {
            ffi::g_object_ref(self.ptr as ffi::gpointer);
        }
        VipsSource { ptr: self.ptr }
    }
}

impl Drop for VipsSource {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::g_object_unref(self.ptr as ffi::gpointer);
            }
        }
    }
}

impl VipsSource {
    /// Opens a file as a Vips source.
    pub fn new_from_file(filename: &str) -> Result<VipsSource, Error> {
        let c = CString::new(filename).map_err(|_| Error::Vips("invalid filename".into()))?;
        let ptr = unsafe { ffi::vips_source_new_from_file(c.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::Vips(vips_error()));
        }
        Ok(VipsSource { ptr })
    }

    /// Creates a Vips source from an in-memory byte buffer.
    ///
    /// The data is **copied** into a Vips-owned blob, so the source remains
    /// valid after the original `data` buffer is dropped.
    pub fn new_from_memory(data: &[u8]) -> Result<VipsSource, Error> {
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
            Ok(VipsSource { ptr })
        }
    }
}
