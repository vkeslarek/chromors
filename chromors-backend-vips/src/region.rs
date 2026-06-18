use crate::VipsBackend;
use crate::Error;
use crate::ffi as ffi;

/// A libvips region — a demand-driven viewport into a `VipsImage`.
///
/// Regions are the core lazy-evaluation primitive in libvips: `prepare` demands
/// a rectangle from the pipeline, and `fetch` reads the pixel data.
pub struct Region {
    pub(crate) ptr: *mut ffi::VipsRegion,
}

unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Clone for Region {
    fn clone(&self) -> Self {
        unsafe {
            ffi::g_object_ref(self.ptr as ffi::gpointer);
        }
        Region { ptr: self.ptr }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::g_object_unref(self.ptr as ffi::gpointer);
            }
        }
    }
}

impl Region {
    /// Creates a new region for the given VipsImage pointer.
    pub(crate) fn new(image_ptr: *mut ffi::VipsImage) -> Result<Region, Error> {
        let ptr = unsafe { ffi::vips_region_new(image_ptr) };
        if ptr.is_null() {
            return Err(Error::Vips(crate::vips_error()));
        }
        Ok(Region { ptr })
    }

    /// Demands a rectangular area `(left, top, width, height)` from the pipeline.
    /// After a successful prepare, the region's data pointer is valid for reading.
    pub fn prepare(&self, left: i32, top: i32, width: i32, height: i32) -> Result<(), Error> {
        let rect = ffi::VipsRect {
            left,
            top,
            width,
            height,
        };
        if unsafe { ffi::vips_region_prepare(self.ptr, &rect) } != 0 {
            return Err(Error::Vips(crate::vips_error()));
        }
        Ok(())
    }

    /// Convenience: prepares and fetches the entire image in one call.
    pub fn materialize(&self) -> Result<Vec<u8>, Error> {
        self.fetch(0, 0, self.width(), self.height())
    }

    /// Fetches pixel data for `(left, top, width, height)` into a `Vec<u8>`.
    /// The region must already be prepared over this rectangle.
    pub fn fetch(&self, left: i32, top: i32, width: i32, height: i32) -> Result<Vec<u8>, Error> {
        let mut len: usize = 0;
        let ptr = unsafe { ffi::vips_region_fetch(self.ptr, left, top, width, height, &mut len) };
        if ptr.is_null() || len == 0 {
            return Err(Error::Vips("vips_region_fetch returned null".into()));
        }
        let out = unsafe { std::slice::from_raw_parts(ptr as *const u8, len).to_vec() };
        unsafe { ffi::g_free(ptr as *mut std::ffi::c_void) };
        Ok(out)
    }

    /// Returns the image width in pixels.
    pub fn width(&self) -> i32 {
        unsafe { ffi::vips_region_width(self.ptr) }
    }

    /// Returns the image height in pixels.
    pub fn height(&self) -> i32 {
        unsafe { ffi::vips_region_height(self.ptr) }
    }
}
