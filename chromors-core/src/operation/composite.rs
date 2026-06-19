use crate::operation::IntoVipsEnum;
use std::hash::Hasher;

use crate::backend::Backend;
use crate::data::image::ImageKind;
use crate::operation::geometry::Direction;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Align {
    Low,
    Centre,
    High,
}
impl IntoVipsEnum for Align {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    Clear,
    Source,
    Over,
    In,
    Out,
    Atop,
    Dest,
    DestOver,
    DestIn,
    DestOut,
    DestAtop,
    Xor,
    Add,
    Saturate,
}
impl IntoVipsEnum for BlendMode {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

// ── Composite2 ────────────────────────────────────────────────────────────────

pub struct Composite2<B: Backend> {
    pub base: Input<ImageKind, B>,
    pub overlay: Input<ImageKind, B>,
    pub mode: BlendMode,
    pub x: Option<i32>,
    pub y: Option<i32>,
    pub premultiplied: Option<bool>,
}

impl<B: Backend> Operation<B> for Composite2<B>
where
    Composite2<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.base, &self.overlay]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.base.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.mode.into_vips());
        state.write_i32(self.x.unwrap_or(0));
        state.write_i32(self.y.unwrap_or(0));
    }
}

// ── Join ──────────────────────────────────────────────────────────────────────

pub struct Join<B: Backend> {
    pub in1: Input<ImageKind, B>,
    pub in2: Input<ImageKind, B>,
    pub direction: Direction,
    pub expand: Option<bool>,
    pub shim: Option<i32>,
    pub align: Option<Align>,
}

impl<B: Backend> Operation<B> for Join<B>
where
    Join<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.in1, &self.in2]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.in1.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.direction.into_vips());
        state.write_i32(self.shim.unwrap_or(0));
        if let Some(a) = self.align {
            state.write_i32(a.into_vips());
        }
    }
}

// ── Insert ────────────────────────────────────────────────────────────────────

pub struct Insert<B: Backend> {
    pub main: Input<ImageKind, B>,
    pub sub: Input<ImageKind, B>,
    pub x: i32,
    pub y: i32,
    pub expand: Option<bool>,
}

impl<B: Backend> Operation<B> for Insert<B>
where
    Insert<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.main, &self.sub]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.main.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.x);
        state.write_i32(self.y);
    }
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────
