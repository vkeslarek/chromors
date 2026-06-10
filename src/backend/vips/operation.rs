use crate::backend::Operation;
use crate::backend::vips::VipsBackend;
use crate::data::image::Image2D;
use crate::error::Error;
use crate::libvips_ffi as ffi;

use super::gobject::{Runner, VipsGObject};

pub trait VipsOperation {
    type Output: Runner;
    fn name() -> &'static [u8];
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage);
}

impl<T: VipsOperation> Operation<Image2D<VipsBackend>> for T {
    type Output = T::Output;

    fn execute(&self, image: &Image2D<VipsBackend>) -> Result<Self::Output, Error> {
        let mut op = VipsGObject::new(T::name())?;
        self.build(&mut op, image.vips_ptr());
        T::Output::run(op)
    }
}
