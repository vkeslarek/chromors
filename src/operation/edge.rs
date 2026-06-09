use crate::backend::gpu::graph::{Graph, NodeId};
use crate::backend::gpu::op::GpuOperation;
use crate::backend::gpu::op::emit_image;
use std::sync::Arc;

use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

pub struct SobelOperation;
impl VipsOperation for SobelOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
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
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"invert\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct SignOperation;
impl VipsOperation for SignOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"sign\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct AbsOperation;
impl VipsOperation for AbsOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"abs\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct PrewittOperation;
impl VipsOperation for PrewittOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"prewitt\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct ScharrOperation;
impl VipsOperation for ScharrOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"scharr\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

impl GpuOperation for InvertOperation {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
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
