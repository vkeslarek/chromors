use crate::operation::IntoVipsEnum;
use std::hash::Hasher;

use crate::backend::Backend;
use crate::data::image::ImageKind;
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

impl<B: Backend> Operation<B> for Add<B>
where
    Add<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct Subtract<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Subtract<B>
where
    Subtract<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct Multiply<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Multiply<B>
where
    Multiply<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct Divide<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Divide<B>
where
    Divide<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct MaxPair<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for MaxPair<B>
where
    MaxPair<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct MinPair<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for MinPair<B>
where
    MinPair<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct Remainder<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Remainder<B>
where
    Remainder<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct Complexform<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Complexform<B>
where
    Complexform<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
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
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.cmplx.into_vips());
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
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.math.into_vips());
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
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.round.into_vips());
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
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.math2.into_vips());
    }
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Add<B>: crate::operation::Lower<B>,
{
    pub fn add(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Add {
            left: self.as_input(),
            right: right.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Subtract<B>: crate::operation::Lower<B>,
{
    pub fn subtract(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Subtract {
            left: self.as_input(),
            right: right.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Multiply<B>: crate::operation::Lower<B>,
{
    pub fn multiply(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Multiply {
            left: self.as_input(),
            right: right.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Divide<B>: crate::operation::Lower<B>,
{
    pub fn divide(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Divide {
            left: self.as_input(),
            right: right.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    MaxPair<B>: crate::operation::Lower<B>,
{
    pub fn max_pair(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(MaxPair {
            left: self.as_input(),
            right: right.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    MinPair<B>: crate::operation::Lower<B>,
{
    pub fn min_pair(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(MinPair {
            left: self.as_input(),
            right: right.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Remainder<B>: crate::operation::Lower<B>,
{
    pub fn remainder(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Remainder {
            left: self.as_input(),
            right: right.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Complexform<B>: crate::operation::Lower<B>,
{
    pub fn complexform(&self, right: &crate::data::image::Image2D<B>) -> Self {
        self.push(Complexform {
            left: self.as_input(),
            right: right.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Complex2<B>: crate::operation::Lower<B>,
{
    pub fn complex2(
        &self,
        right: &crate::data::image::Image2D<B>,
        cmplx: OperationComplex2,
    ) -> Self {
        self.push(Complex2 {
            left: self.as_input(),
            right: right.as_input(),
            cmplx,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Math<B>: crate::operation::Lower<B>,
{
    pub fn math(&self, math: OperationMath) -> Self {
        self.push(Math {
            input: self.as_input(),
            math,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Round<B>: crate::operation::Lower<B>,
{
    pub fn round(&self, round: OperationRound) -> Self {
        self.push(Round {
            input: self.as_input(),
            round,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Math2<B>: crate::operation::Lower<B>,
{
    pub fn math2(&self, right: &crate::data::image::Image2D<B>, math2: OperationMath2) -> Self {
        self.push(Math2 {
            left: self.as_input(),
            right: right.as_input(),
            math2,
        })
    }
}

// ── Constant operations ────────────────────────────────────────────────────────

pub struct Linear<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub a: Vec<f64>,
    pub b: Vec<f64>,
}
impl<B: Backend> Operation<B> for Linear<B>
where
    Linear<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        for &v in &self.a {
            state.write(&v.to_le_bytes());
        }
        for &v in &self.b {
            state.write(&v.to_le_bytes());
        }
    }
}

pub struct Math2Const<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub math2: OperationMath2,
    pub c: Vec<f64>,
}
impl<B: Backend> Operation<B> for Math2Const<B>
where
    Math2Const<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.math2.into_vips());
        for &v in &self.c {
            state.write(&v.to_le_bytes());
        }
    }
}

pub struct RemainderConst<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub c: Vec<f64>,
}
impl<B: Backend> Operation<B> for RemainderConst<B>
where
    RemainderConst<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        for &v in &self.c {
            state.write(&v.to_le_bytes());
        }
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Linear<B>: crate::operation::Lower<B>,
{
    pub fn linear(&self, a: Vec<f64>, b: Vec<f64>) -> Self {
        self.push(Linear {
            input: self.as_input(),
            a,
            b,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Math2Const<B>: crate::operation::Lower<B>,
{
    pub fn math2_const(&self, math2: OperationMath2, c: Vec<f64>) -> Self {
        self.push(Math2Const {
            input: self.as_input(),
            math2,
            c,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    RemainderConst<B>: crate::operation::Lower<B>,
{
    pub fn remainder_const(&self, c: Vec<f64>) -> Self {
        self.push(RemainderConst {
            input: self.as_input(),
            c,
        })
    }
}
