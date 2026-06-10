use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

pub struct ForwardFftOperation;
impl VipsOperation for ForwardFftOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"fwfft\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct InverseFftOperation;
impl VipsOperation for InverseFftOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"invfft\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct SpectrumOperation;
impl VipsOperation for SpectrumOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"spectrum\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}
