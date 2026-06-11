use crate::error::Error;
use crate::ffi as ffi;

use super::Source;

pub struct Sbuf {
    pub(crate) ptr: *mut ffi::VipsSbuf,
}

impl Drop for Sbuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::vips_sbuf_unbuffer(self.ptr);
            }
        }
    }
}

impl Sbuf {
    pub fn new(source: &Source) -> Result<Sbuf, Error> {
        let ptr = unsafe { ffi::vips_sbuf_new_from_source(source.ptr) };
        if ptr.is_null() {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(Sbuf { ptr })
    }

    pub fn getc(&mut self) -> Option<u8> {
        let c = unsafe { ffi::vips_sbuf_getc(self.ptr) };
        if c == -1 {
            return None;
        }
        Some(c as u8)
    }
}
