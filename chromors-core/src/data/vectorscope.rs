//! The vectorscope datatype — a 2-D Cb/Cr density grid.
//! `Atomic`-shaped (the whole grid is one reduction), GPU-only by construction.

use std::any::Any;
use std::hash::Hasher;

use crate::backend::Backend;
use crate::data::image::ImageKind;
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::operation::{AnyInput, Input, Operation};
use crate::work_unit::{Atomic, Lod, Region, WorkUnit};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// A `grid × grid` Cb/Cr density grid of atomic counters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorscopeKind {
    pub grid: u32,
}

impl AnyKind for VectorscopeKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, _wu: &WorkUnit) -> u64 {
        (self.grid as u64 * self.grid as u64 * 4).max(16)
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.grid);
    }
}

impl Kind for VectorscopeKind {
    type WorkUnit = Atomic;
}

/// What the user holds. Per-backend.
pub type Vectorscope<B> = Data<VectorscopeKind, B>;

// ── Operation: Image → Vectorscope ────────────────────────────────────────────

pub struct VectorscopeOp<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub grid: u32,
}

impl<B: Backend> Operation<B> for VectorscopeOp<B> 
where 
    Self: crate::operation::Lower<B>,
{
    type Output = VectorscopeKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, _out: &Atomic) -> Vec<Option<WorkUnit>> {
        let (w, h) = self.input.spec.dims();
        vec![Some(WorkUnit::Region(Region::full((w, h), Lod(0))))]
    }
    fn output_spec(&self) -> VectorscopeKind {
        VectorscopeKind { grid: self.grid }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.grid);
    }
}

impl<B: Backend> crate::data::image::Image2D<B>
where
    VectorscopeOp<B>: crate::operation::Lower<B>,
{
    pub fn vectorscope(&self, grid: u32) -> Vectorscope<B> {
        self.push(VectorscopeOp {
            input: self.as_input(),
            grid,
        })
    }
}
