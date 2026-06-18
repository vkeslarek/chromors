use std::hash::Hasher;

use crate::backend::Backend;
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

// ── Edge Operations (with halo) ───────────────────────────────────────────────

pub struct Sobel<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Sobel<B>
where
    Sobel<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.expanded(1)))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.layout = spec.layout.to_f32();
        spec
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct Prewitt<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Prewitt<B>
where
    Prewitt<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.expanded(1)))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.layout = spec.layout.to_f32();
        spec
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct Scharr<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Scharr<B>
where
    Scharr<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.expanded(1)))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.layout = spec.layout.to_f32();
        spec
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── Pointwise Operations (no halo) ────────────────────────────────────────────

pub struct Invert<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Invert<B>
where
    Invert<B>: Lower<B>,
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
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct Sign<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Sign<B>
where
    Sign<B>: Lower<B>,
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
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct Abs<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Abs<B>
where
    Abs<B>: Lower<B>,
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
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Sobel<B>: crate::operation::Lower<B>,
{
    pub fn sobel(&self) -> Self {
        self.push(Sobel {
            input: self.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Prewitt<B>: crate::operation::Lower<B>,
{
    pub fn prewitt(&self) -> Self {
        self.push(Prewitt {
            input: self.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Scharr<B>: crate::operation::Lower<B>,
{
    pub fn scharr(&self) -> Self {
        self.push(Scharr {
            input: self.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Invert<B>: crate::operation::Lower<B>,
{
    pub fn invert(&self) -> Self {
        self.push(Invert {
            input: self.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Sign<B>: crate::operation::Lower<B>,
{
    pub fn sign(&self) -> Self {
        self.push(Sign {
            input: self.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Abs<B>: crate::operation::Lower<B>,
{
    pub fn abs(&self) -> Self {
        self.push(Abs {
            input: self.as_input(),
        })
    }
}

pub struct Hypot<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Hypot<B>
where
    Hypot<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Region(out.clone())),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn std::hash::Hasher) {}
}

