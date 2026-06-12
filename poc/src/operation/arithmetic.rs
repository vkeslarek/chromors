use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{VipsBackend, VipsBuilder, IntoVipsEnum};
use crate::data::image::ImageKind;
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::gpu::view::ParamBlock;
use crate::operation::{
    AnyInput, Input, Lower, Operation, OperationComplex2, OperationMath, OperationMath2,
    OperationRound,
};
use crate::work_unit::{Region, WorkUnit};

// ── Binary operations ─────────────────────────────────────────────────────────

pub struct Add<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Add<B> where Add<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Add<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"add\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Subtract<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Subtract<B> where Subtract<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Subtract<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"subtract\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Multiply<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Multiply<B> where Multiply<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Multiply<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"multiply\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Divide<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Divide<B> where Divide<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Divide<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"divide\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct MaxPair<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for MaxPair<B> where MaxPair<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for MaxPair<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"maxpair\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct MinPair<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for MinPair<B> where MinPair<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for MinPair<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"minpair\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Remainder<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Remainder<B> where Remainder<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Remainder<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"remainder\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Complexform<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Complexform<B> where Complexform<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Complexform<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"complexform\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Operations with enums ─────────────────────────────────────────────────────

pub struct Complex2<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
    pub cmplx: OperationComplex2,
}

impl<B: Backend> Operation<B> for Complex2<B>
where
    Complex2<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.cmplx.into_vips());
    }
}

impl Lower<VipsBackend> for Complex2<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"complex2\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        op.set_int("cmplx", self.cmplx.into_vips());
        let out_handle = op.run().expect("vips complex2 failed");
        cx.emit(out_handle);
    }
}

pub struct Math<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub math: OperationMath,
}

impl<B: Backend> Operation<B> for Math<B>
where
    Math<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.math.into_vips());
    }
}

impl Lower<VipsBackend> for Math<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"math\0")
            .expect("failed to create vips math op");
        op.set_image("in", input_handle.ptr);
        op.set_int("math", self.math.into_vips());
        let out_handle = op.run().expect("vips math failed");
        cx.emit(out_handle);
    }
}

pub struct Round<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub round: OperationRound,
}

impl<B: Backend> Operation<B> for Round<B>
where
    Round<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.round.into_vips());
    }
}

impl Lower<VipsBackend> for Round<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"round\0")
            .expect("failed to create vips round op");
        op.set_image("in", input_handle.ptr);
        op.set_int("round", self.round.into_vips());
        let out_handle = op.run().expect("vips round failed");
        cx.emit(out_handle);
    }
}

pub struct Math2<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
    pub math2: OperationMath2,
}

impl<B: Backend> Operation<B> for Math2<B>
where
    Math2<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.left, &self.right] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.left.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.math2.into_vips());
    }
}

impl Lower<VipsBackend> for Math2<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"math2\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        op.set_int("math2", self.math2.into_vips());
        let out_handle = op.run().expect("vips math2 failed");
        cx.emit(out_handle);
    }
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────

impl Lower<GpuBackend> for Add<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { cx.kernel("add_kernel"); cx.output(self.output_spec().output()); }
}
impl Lower<GpuBackend> for Subtract<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { cx.kernel("subtract_kernel"); cx.output(self.output_spec().output()); }
}
impl Lower<GpuBackend> for Multiply<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { cx.kernel("multiply_kernel"); cx.output(self.output_spec().output()); }
}
impl Lower<GpuBackend> for Divide<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { cx.kernel("divide_kernel"); cx.output(self.output_spec().output()); }
}
impl Lower<GpuBackend> for MaxPair<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { cx.kernel("max_kernel"); cx.output(self.output_spec().output()); }
}
impl Lower<GpuBackend> for MinPair<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { cx.kernel("min_kernel"); cx.output(self.output_spec().output()); }
}
impl Lower<GpuBackend> for Remainder<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { cx.kernel("remainder_kernel"); cx.output(self.output_spec().output()); }
}
impl Lower<GpuBackend> for Complexform<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { cx.kernel("complexform_kernel"); cx.output(self.output_spec().output()); }
}
impl Lower<GpuBackend> for Complex2<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { 
        cx.param_block(ParamBlock::scalar("op", "uint", self.cmplx.into_vips() as u32));
        cx.kernel("complex2_kernel"); 
        cx.output(self.output_spec().output()); 
    }
}
impl Lower<GpuBackend> for Math<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { 
        cx.param_block(ParamBlock::scalar("op", "uint", self.math.into_vips() as u32));
        cx.kernel("math_kernel"); 
        cx.output(self.output_spec().output()); 
    }
}
impl Lower<GpuBackend> for Round<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { 
        cx.param_block(ParamBlock::scalar("op", "uint", self.round.into_vips() as u32));
        cx.kernel("round_kernel"); 
        cx.output(self.output_spec().output()); 
    }
}
impl Lower<GpuBackend> for Math2<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) { 
        cx.param_block(ParamBlock::scalar("op", "uint", self.math2.into_vips() as u32));
        cx.kernel("math2_kernel"); 
        cx.output(self.output_spec().output()); 
    }
}


impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Add<B>: crate::operation::Lower<B>,
{
    pub fn add(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Add { left: self.as_input(), right: right.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Subtract<B>: crate::operation::Lower<B>,
{
    pub fn subtract(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Subtract { left: self.as_input(), right: right.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Multiply<B>: crate::operation::Lower<B>,
{
    pub fn multiply(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Multiply { left: self.as_input(), right: right.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Divide<B>: crate::operation::Lower<B>,
{
    pub fn divide(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Divide { left: self.as_input(), right: right.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    MaxPair<B>: crate::operation::Lower<B>,
{
    pub fn max_pair(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(MaxPair { left: self.as_input(), right: right.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    MinPair<B>: crate::operation::Lower<B>,
{
    pub fn min_pair(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(MinPair { left: self.as_input(), right: right.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Remainder<B>: crate::operation::Lower<B>,
{
    pub fn remainder(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Remainder { left: self.as_input(), right: right.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Complexform<B>: crate::operation::Lower<B>,
{
    pub fn complexform(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Complexform { left: self.as_input(), right: right.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Complex2<B>: crate::operation::Lower<B>,
{
    pub fn complex2(&self, right: &crate::data::image::Image2D<B>, cmplx: OperationComplex2) -> Self {
        self.push(Complex2 { left: self.as_input(), right: right.as_input(), cmplx })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Math<B>: crate::operation::Lower<B>,
{
    pub fn math(&self, math: OperationMath) -> Self {
        self.push(Math { input: self.as_input(), math })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Round<B>: crate::operation::Lower<B>,
{
    pub fn round(&self, round: OperationRound) -> Self {
        self.push(Round { input: self.as_input(), round })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Math2<B>: crate::operation::Lower<B>,
{
    pub fn math2(&self, right: &crate::data::image::Image2D<B>, math2: OperationMath2) -> Self {
        self.push(Math2 { left: self.as_input(), right: right.as_input(), math2 })
    }
}
