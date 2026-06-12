use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::vips::{VipsBackend, VipsBuilder};
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

// ── Edge Operations (with halo) ───────────────────────────────────────────────

pub struct Sobel<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Sobel<B> where Sobel<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.expanded(1)))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Sobel<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"sobel\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Prewitt<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Prewitt<B> where Prewitt<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.expanded(1)))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Prewitt<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"prewitt\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Scharr<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Scharr<B> where Scharr<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.expanded(1)))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Scharr<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"scharr\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Pointwise Operations (no halo) ────────────────────────────────────────────

pub struct Invert<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Invert<B> where Invert<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Invert<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"invert\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<GpuBackend> for Invert<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("invert_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Sign<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("sign_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Abs<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("abs_kernel");
        cx.output(self.output_spec().output());
    }
}

pub struct Sign<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Sign<B> where Sign<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Sign<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"sign\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Abs<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Abs<B> where Abs<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Abs<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"abs\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}


impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Sobel<B>: crate::operation::Lower<B>,
{
    pub fn sobel(&self) -> Self {
        self.push(Sobel { input: self.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Prewitt<B>: crate::operation::Lower<B>,
{
    pub fn prewitt(&self) -> Self {
        self.push(Prewitt { input: self.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Scharr<B>: crate::operation::Lower<B>,
{
    pub fn scharr(&self) -> Self {
        self.push(Scharr { input: self.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Invert<B>: crate::operation::Lower<B>,
{
    pub fn invert(&self) -> Self {
        self.push(Invert { input: self.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Sign<B>: crate::operation::Lower<B>,
{
    pub fn sign(&self) -> Self {
        self.push(Sign { input: self.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Abs<B>: crate::operation::Lower<B>,
{
    pub fn abs(&self) -> Self {
        self.push(Abs { input: self.as_input() })
    }
}
