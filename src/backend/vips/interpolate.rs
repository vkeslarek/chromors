use std::ffi::CString;

use super::IntoVipsName;
use crate::error::Error;
use crate::ffi as ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterpolationMethod {
    Nearest,
    Bilinear,
    Bicubic,
    Lbb,
    Nohalo,
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
    pub fn new(method: impl IntoVipsName) -> Result<Interpolate, Error> {
        let c = CString::new(method.into_vips_name())
            .map_err(|_| Error::Vips("invalid nickname".into()))?;
        let ptr = unsafe { ffi::vips_interpolate_new(c.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(Interpolate { ptr })
    }
}
