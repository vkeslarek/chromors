//! The vector graphics datatype — used to hold Vello scenes or SVG data.

use std::any::Any;
use std::hash::Hasher;

use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::work_unit::{Atomic, WorkUnit};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// VectorGraphics metadata. Usually resolution-independent, but may have a nominal bounding box.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorGraphicsKind {
    pub width: f32,
    pub height: f32,
}

impl VectorGraphicsKind {
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

impl AnyKind for VectorGraphicsKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, _wu: &WorkUnit) -> u64 {
        0
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.width.to_bits());
        state.write_u32(self.height.to_bits());
    }
}

impl Kind for VectorGraphicsKind {
    type WorkUnit = Atomic;
}

/// What the user holds. Aliased over the generic core; per-backend.
pub type VectorGraphics<B> = Data<VectorGraphicsKind, B>;

// ── Interop: Vello -> Image2D ──────────────────────────────────────────────────

use crate::buffer::Buffer;
use crate::color::model::ColorModel;
use crate::color::space::ColorSpace;
use crate::data::image::ImageKind;
use crate::error::Error;
use crate::io::{Source, Target};
use crate::pixel::{AlphaState, PixelLayout, Storage};
use crate::work_unit::Region;
use std::sync::Arc;
