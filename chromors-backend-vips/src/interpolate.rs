use std::ffi::CString;

use super::IntoVipsName;
use crate::Error;
use crate::ffi as ffi;

/// Vips interpolation methods for geometric transforms (resize, rotate, affine, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolationMethod {
    /// Nearest-neighbor (fastest, lowest quality).
    Nearest,
    /// Bilinear interpolation.
    Bilinear,
    /// Bicubic interpolation (Mitchell filter).
    Bicubic,
    /// Locally Bounded Bicubic (moderate quality, sharp).
    Lbb,
    /// Nohalo (edge-directed, high quality).
    Nohalo,
    /// Vsqbs (variable-size quadratic B-spline).
    Vsqbs,
}

impl IntoVipsName for InterpolationMethod {
    fn into_vips_name(self) -> &'static str {
        match self {
            InterpolationMethod::Nearest => "nearest",
            InterpolationMethod::Bilinear => "bilinear",
            InterpolationMethod::Bicubic => "bicubic",
            InterpolationMethod::Lbb => "lbb",
            InterpolationMethod::Nohalo => "nohalo",
            InterpolationMethod::Vsqbs => "vsqbs",
        }
    }
}

/// An opaque libvips interpolator object, created from an [`InterpolationMethod`].
pub struct Interpolate {
    pub(crate) ptr: *mut ffi::VipsInterpolate,
}

unsafe impl Send for Interpolate {}
unsafe impl Sync for Interpolate {}

impl Clone for Interpolate {
    fn clone(&self) -> Self {
        unsafe {
            ffi::g_object_ref(self.ptr as ffi::gpointer);
        }
        Interpolate { ptr: self.ptr }
    }
}

impl Drop for Interpolate {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::g_object_unref(self.ptr as ffi::gpointer);
            }
        }
    }
}

impl Interpolate {
    /// Creates a new libvips interpolator from the given method.
    pub fn new(method: impl IntoVipsName) -> Result<Interpolate, Error> {
        let c = CString::new(method.into_vips_name())
            .map_err(|_| Error::Vips("invalid nickname".into()))?;
        let ptr = unsafe { ffi::vips_interpolate_new(c.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::Vips(crate::vips_error()));
        }
        Ok(Interpolate { ptr })
    }
}
