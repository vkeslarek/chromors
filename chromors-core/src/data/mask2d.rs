//! The 2-D mask datatype — small raw-f32 weight grids: convolution masks,
//! morphology elements, band-recombination matrices. NOT colorimetric: no
//! `PixelFormat`, no `ColorSpace`, no codec sandwich. Weights are bound and
//! read as plain `f32`, broadcast to `float4(v, v, v, 1)` for `IRegion`
//! consumers (same trick as the Gray codecs).

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use crate::buffer::Buffer;
use crate::error::Error;
use crate::io::Source;
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::work_unit::{Region, WorkUnit, WorkUnitFor};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// Mask metadata: just an extent. A raw `f32` grid, no pixel format, no color
/// space — there is nothing left to lie about.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask2DKind {
    pub width: i32,
    pub height: i32,
}

impl Mask2DKind {
    pub fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }
    pub fn dims(&self) -> (i32, i32) {
        (self.width, self.height)
    }
}

impl AnyKind for Mask2DKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        match wu {
            WorkUnit::Region(r) => (r.w.max(0) as u64) * (r.h.max(0) as u64) * 4,
            _ => 0,
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height);
    }
}

impl Kind for Mask2DKind {
    type WorkUnit = Region;
}

/// What the user holds. Aliased over the generic core; per-backend.
pub type Mask2D<B> = Data<Mask2DKind, B>;

// ── GPU constant source ──────────────────────────────────────────────────────

/// A GPU leaf holding a constant `f32` grid — replaces
/// `Image2D::from_constant_f32` for non-colorimetric weight data.
pub struct GpuConstantMaskSource {
    pub spec: Arc<Mask2DKind>,
    pub data: Vec<f32>,
}

// ── Vips constant source ────────────────────────────────────────────────────

/// A Vips leaf holding a constant `f64` weight grid — `conv`/`morph`-family
/// ops read mask images in vips' native `VIPS_FORMAT_DOUBLE`.
pub struct VipsConstantMaskSource {
    pub spec: Arc<Mask2DKind>,
    pub data: Vec<f64>,
    pub scale: f64,
    pub offset: f64,
}

impl<B: crate::backend::Backend> Mask2D<B> {
    pub fn width(&self) -> i32 {
        self.spec.width
    }
    pub fn height(&self) -> i32 {
        self.spec.height
    }
}

// ── Targets ──────────────────────────────────────────────────────────────────

use crate::backend::Backend;
use crate::io::Target;

/// Extracts the mask's raw `f32` weight grid into host RAM.
///
/// This is the sanctioned exit for mask data — analogous to `RamImageTarget`
/// for images.
pub struct RamMaskTarget;
