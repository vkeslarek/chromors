//! Work-unit vocabulary for the GPU graph — typed and erased forms.
//!
//! Each DataType declares its natural division strategy via
//! [`super::datatype::DataType::work_unit_kind`].  The concrete structs
//! ([`Region`], [`Range`], [`Atomic`]) are the typed handles that
//! [`super::datatype::TypedData`] impls receive and return.
//! [`WorkUnit`] is the type-erased enum used wherever the graph must hold
//! or pass demands across heterogeneous node boundaries.
//!
//! The WU type — not the DataType — is what drives the shader interface:
//! `Region` → rect-addressed storage (WorkingDecodeRegion / RWRegion);
//! `Range`  → 1-D range storage;
//! `Atomic` → indivisible accumulator (HistogramOut / atomic u32 array).

use crate::geometry::Rect;

use super::handle::Lod;

// ── AnyWorkUnit ───────────────────────────────────────────────────────────────

/// Marker trait for typed work-unit structs.
///
/// Implementors can convert to and from the erased [`WorkUnit`] enum, enabling
/// typed [`DemandMap`] impls to interop with the heterogeneous graph walk.
pub trait AnyWorkUnit:
    Clone + std::fmt::Debug + PartialEq + Eq + std::hash::Hash + Send + Sync + 'static
{
    fn to_work_unit(&self) -> WorkUnit;
    fn from_work_unit(wu: &WorkUnit) -> Option<Self>;
}

// ── Typed work-unit structs ───────────────────────────────────────────────────

/// 2-D sub-rectangle at a given LOD — the natural WU of Image2D, Mask2D, Fft2D, FeatureMap.
///
/// Carries both the spatial extent (`rect`) and the resolution level (`lod`),
/// so a tile request is fully self-describing.  Drives the rect-addressed
/// shader interface: `WorkingDecodeRegion<CodecRegion<…>>` on reads,
/// `RWRegion` / `from_working` on writes.  Dispatch grid covers
/// `rect.width × rect.height` threads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Region {
    pub rect: Rect,
    pub lod: Lod,
}

impl Region {
    pub fn new(rect: Rect, lod: Lod) -> Self {
        Self { rect, lod }
    }

    pub fn full_res(rect: Rect) -> Self {
        Self {
            rect,
            lod: Lod::FULL,
        }
    }
}

/// 1-D sub-range `[start, end)` — the natural WU of Mask1D, Fft1D.
///
/// Drives 1-D range-addressed shader interface.  Dispatch grid is linear:
/// `end - start` threads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Range {
    pub start: u32,
    pub end: u32,
}

/// Indivisible unit — the natural WU of Histogram, Scalar, PointList.
///
/// There is no meaningful sub-piece: the only demand is the whole result.
/// Drives the atomic-accumulator shader interface (`HistogramOut`, raw u32
/// counter array).  Dispatch grid is determined by the *input* being scanned,
/// not this output's shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atomic;

// ── AnyWorkUnit impls ─────────────────────────────────────────────────────────

impl AnyWorkUnit for Region {
    fn to_work_unit(&self) -> WorkUnit {
        WorkUnit::Region {
            rect: self.rect,
            lod: self.lod,
        }
    }
    fn from_work_unit(wu: &WorkUnit) -> Option<Self> {
        match wu {
            WorkUnit::Region { rect, lod } => Some(Region {
                rect: *rect,
                lod: *lod,
            }),
            _ => None,
        }
    }
}

impl AnyWorkUnit for Range {
    fn to_work_unit(&self) -> WorkUnit {
        WorkUnit::Range {
            start: self.start,
            end: self.end,
        }
    }
    fn from_work_unit(wu: &WorkUnit) -> Option<Self> {
        match wu {
            WorkUnit::Range { start, end } => Some(Range {
                start: *start,
                end: *end,
            }),
            _ => None,
        }
    }
}

impl AnyWorkUnit for Atomic {
    fn to_work_unit(&self) -> WorkUnit {
        WorkUnit::Atomic
    }
    fn from_work_unit(wu: &WorkUnit) -> Option<Self> {
        match wu {
            WorkUnit::Atomic => Some(Atomic),
            _ => None,
        }
    }
}

// ── WorkUnitKind ──────────────────────────────────────────────────────────────

/// The *shape* of a datatype's natural division strategy, with no payload.
///
/// [`super::datatype::DataType::work_unit_kind`] returns this — it answers
/// "Region, Range, or Atomic?" without needing the node's resolved
/// dimensions, which the caller (readback fork in `materialize.rs`) discards
/// anyway in favor of its own resolved `rect`/`lod`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WorkUnitKind {
    Region,
    Range,
    Atomic,
}

// ── WorkUnit (erased) ─────────────────────────────────────────────────────────

/// Type-erased work-unit — the wire format for the heterogeneous graph walk.
///
/// Nodes are stored as `Arc<dyn GpuOperation>` so demands must cross DataType
/// boundaries dynamically.  [`WorkUnit`] is the shared vocabulary: both sides
/// of an edge convert through it.  Typed [`DemandMap`] impls downcast
/// from this via [`AnyWorkUnit::from_work_unit`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WorkUnit {
    /// 2-D sub-rectangle at a given LOD — erased form of [`Region`].
    Region { rect: Rect, lod: Lod },
    /// 1-D sub-range — erased form of [`Range`].
    Range { start: u32, end: u32 },
    /// Indivisible unit — erased form of [`Atomic`].
    Atomic,
}

impl WorkUnit {
    /// Resolve to a bounding `Rect` for buffer allocation and kernel dispatch.
    ///
    /// `Atomic` and `Range` collapse to the node's full output rect — their
    /// storage is not spatially sub-divided.  `Region` passes through unchanged.
    pub fn resolve(&self, w: u32, h: u32) -> Rect {
        match self {
            WorkUnit::Region { rect, .. } => *rect,
            WorkUnit::Atomic | WorkUnit::Range { .. } => Rect::new(0, 0, w as i32, h as i32),
        }
    }

    /// Extract the LOD from a Region seed, defaulting to full-res for non-Region units.
    pub fn lod(&self) -> Lod {
        match self {
            WorkUnit::Region { lod, .. } => *lod,
            _ => Lod::FULL,
        }
    }
}
