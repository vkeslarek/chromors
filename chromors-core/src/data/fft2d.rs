//! The 2-D FFT spectrum datatype — a complex-valued frequency plane,
//! `width x height x bands`, 2 x `f32` per sample. Vips-only for now
//! (`VipsBand` -> `VIPS_FORMAT_DPCOMPLEX`); a GPU `IRegion` view lands if/when
//! a GPU FFT exists.

use std::any::Any;
use std::hash::Hasher;

use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::work_unit::{Region, WorkUnit};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// Spectrum metadata: extent + band count. Each sample is a complex number
/// (2 x `f32`).
#[derive(Clone, Debug, PartialEq)]
pub struct Fft2DKind {
    pub width: i32,
    pub height: i32,
    pub bands: u32,
}

impl Fft2DKind {
    pub fn new(width: i32, height: i32, bands: u32) -> Self {
        Self {
            width,
            height,
            bands,
        }
    }
    pub fn dims(&self) -> (i32, i32) {
        (self.width, self.height)
    }
}

impl AnyKind for Fft2DKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        match wu {
            WorkUnit::Region(r) => {
                (r.w.max(0) as u64) * (r.h.max(0) as u64) * self.bands.max(1) as u64 * 8
            }
            _ => 0,
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height);
        state.write_u32(self.bands);
    }
}

impl Kind for Fft2DKind {
    type WorkUnit = Region;
}

/// What the user holds. Aliased over the generic core; per-backend.
pub type Fft2D<B> = Data<Fft2DKind, B>;
