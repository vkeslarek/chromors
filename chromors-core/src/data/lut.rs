//! The lookup-table datatype — a 1-D raw-f32 table, `entries` samples x
//! `bands` channels. The first `Range`-shaped Kind: a LUT is genuinely
//! 1-D, unlike `Mask2DKind`'s small 2-D grids.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use crate::buffer::Buffer;
use crate::error::Error;
use crate::io::{Source, Target};
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::work_unit::{Range, Region, WorkUnit};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// LUT metadata: `entries` samples x `bands` channels, raw `f32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LutKind {
    pub entries: u32,
    pub bands: u32,
}

impl LutKind {
    pub fn new(entries: u32, bands: u32) -> Self {
        Self { entries, bands }
    }
}

impl AnyKind for LutKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        let entries = match wu {
            WorkUnit::Range(r) => (r.end - r.start).max(0) as u64,
            _ => self.entries as u64,
        };
        // Storage is always packed `float4` per entry (`bands <= 4`, zero-padded),
        // regardless of `bands`.
        (entries * 16).max(16)
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.entries);
        state.write_u32(self.bands);
    }
}

impl Kind for LutKind {
    type WorkUnit = Range;
}

/// What the user holds. Aliased over the generic core; per-backend.
pub type Lut<B> = Data<LutKind, B>;

// ── GPU constant source ────────────────────────────────────────────────────

/// A GPU leaf holding a constant LUT — `entries` packed `float4`s
/// (row-major, `bands` <= 4 channels, zero-padded).

// ── Vips constant source ────────────────────────────────────────────────────

/// A Vips leaf holding a constant `f64` LUT/matrix — `entries` rows x
/// `bands` columns, row-major, as a 1-band `VIPS_FORMAT_DOUBLE` image of
/// `Xsize = bands`, `Ysize = entries` (vips' "matrix image" convention).

// ── Raw targets ─────────────────────────────────────────────────────────────

/// Reads a LUT buffer back to host RAM as raw bytes — GPU side gives packed
/// `float4` (`entries * 16` bytes), Vips side gives row-major `f64` (`entries
/// * bands * 8` bytes, 1-band `Xsize x Ysize` matrix image).
pub struct RawLutTarget;

