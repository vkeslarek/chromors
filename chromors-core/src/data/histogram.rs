use std::any::Any;
use std::hash::Hasher;

use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::work_unit::{Atomic, WorkUnit};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// A 1-D histogram of `bins` atomic counters, `bands` of them side by side
/// (e.g. `hist_find` on an RGB image → `bands == 3`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistogramKind {
    pub bins: u32,
    pub bands: u32,
}

impl AnyKind for HistogramKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, _wu: &WorkUnit) -> u64 {
        (self.bins as u64 * self.bands.max(1) as u64 * 4).max(16)
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.bins);
        state.write_u32(self.bands);
    }
}

impl Kind for HistogramKind {
    type WorkUnit = Atomic;
}

/// What the user holds.
pub type Histogram<B> = Data<HistogramKind, B>;
