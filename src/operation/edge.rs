use crate::backend::gpu::datatype::ImageType;
use crate::backend::gpu::graph::{Graph, NodeId};
use crate::backend::gpu::op::emit_image;
use crate::backend::gpu::op::{GpuOperation, TypedOperation};
use std::sync::Arc;

use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

pub struct SobelOperation;
impl VipsOperation for SobelOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"sobel\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

#[derive(Clone, Debug)]
pub struct InvertOperation;
impl VipsOperation for InvertOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"invert\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct SignOperation;
impl VipsOperation for SignOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"sign\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct AbsOperation;
impl VipsOperation for AbsOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"abs\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct PrewittOperation;
impl VipsOperation for PrewittOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"prewitt\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct ScharrOperation;
impl VipsOperation for ScharrOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"scharr\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

impl TypedOperation for InvertOperation {
    type Output = ImageType;
}

impl GpuOperation for InvertOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        emit_image(
            graph,
            input,
            self_arc,
            "ops.invert",
            "invert_kernel",
            vec![],
        )
    }
}
