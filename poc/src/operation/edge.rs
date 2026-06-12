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

pub struct Hypot<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Hypot<B> where Hypot<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())), Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn std::hash::Hasher) {}
}

impl Lower<GpuBackend> for Hypot<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("hypot_kernel");
        cx.output(self.output_spec().output());
    }
}

impl crate::data::image::Image2D<GpuBackend> {
    pub fn sobel(&self) -> Self {
        use crate::operation::convolution::Convolution;
        let ctx = std::sync::Arc::clone(self.ctx());
        let mask_gx = crate::data::image::Image2D::from_constant_f32(
            std::sync::Arc::clone(&ctx), 3, 3,
            &[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]
        );
        let mask_gy = crate::data::image::Image2D::from_constant_f32(
            ctx, 3, 3,
            &[-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0]
        );
        let gx = self.push(Convolution {
            input: self.as_input(), mask: mask_gx.as_input(),
            precision: None, layers: None, cluster: None
        });
        let gy = self.push(Convolution {
            input: self.as_input(), mask: mask_gy.as_input(),
            precision: None, layers: None, cluster: None
        });
        gx.push(Hypot { left: gx.as_input(), right: gy.as_input() })
    }

    pub fn prewitt(&self) -> Self {
        use crate::operation::convolution::Convolution;
        let ctx = std::sync::Arc::clone(self.ctx());
        let mask_gx = crate::data::image::Image2D::from_constant_f32(
            std::sync::Arc::clone(&ctx), 3, 3,
            &[-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
        );
        let mask_gy = crate::data::image::Image2D::from_constant_f32(
            ctx, 3, 3,
            &[-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        );
        let gx = self.push(Convolution {
            input: self.as_input(), mask: mask_gx.as_input(),
            precision: None, layers: None, cluster: None
        });
        let gy = self.push(Convolution {
            input: self.as_input(), mask: mask_gy.as_input(),
            precision: None, layers: None, cluster: None
        });
        gx.push(Hypot { left: gx.as_input(), right: gy.as_input() })
    }

    pub fn scharr(&self) -> Self {
        use crate::operation::convolution::Convolution;
        let ctx = std::sync::Arc::clone(self.ctx());
        let mask_gx = crate::data::image::Image2D::from_constant_f32(
            std::sync::Arc::clone(&ctx), 3, 3,
            &[-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0]
        );
        let mask_gy = crate::data::image::Image2D::from_constant_f32(
            ctx, 3, 3,
            &[-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0]
        );
        let gx = self.push(Convolution {
            input: self.as_input(), mask: mask_gx.as_input(),
            precision: None, layers: None, cluster: None
        });
        let gy = self.push(Convolution {
            input: self.as_input(), mask: mask_gy.as_input(),
            precision: None, layers: None, cluster: None
        });
        gx.push(Hypot { left: gx.as_input(), right: gy.as_input() })
    }
}
