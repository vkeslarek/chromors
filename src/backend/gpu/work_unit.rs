//! Work-unit vocabulary for the GPU graph — typed and erased forms.
//!
//! Each DataType declares its natural division strategy as an associated type
//! on [`GpuData`].  The concrete structs ([`Region`], [`Range`], [`Atomic`])
//! are the typed handles that [`MaterializePlan`] receives and returns.
//! [`WorkUnit`] is the type-erased enum used wherever the graph must hold
//! or pass demands across heterogeneous node boundaries.
//!
//! The WU type — not the DataType — is what drives the shader interface:
//! `Region` → rect-addressed storage (WorkingDecodeRegion / RWRegion);
//! `Range`  → 1-D range storage;
//! `Atomic` → indivisible accumulator (HistogramOut / atomic u32 array).

use crate::geometry::Rect;

// ── AnyWorkUnit ───────────────────────────────────────────────────────────────

/// Marker trait for typed work-unit structs.
///
/// Implementors can convert to and from the erased [`WorkUnit`] enum, enabling
/// typed [`MaterializePlan`] impls to interop with the heterogeneous graph walk.
pub trait AnyWorkUnit: Clone + std::fmt::Debug + Send + Sync + 'static {
    fn to_work_unit(&self) -> WorkUnit;
    fn from_work_unit(wu: &WorkUnit) -> Option<Self>;
}

// ── Typed work-unit structs ───────────────────────────────────────────────────

/// 2-D sub-rectangle — the natural WU of Image, Mask2D, Fft2D, FeatureMap.
///
/// Drives the rect-addressed shader interface:
/// `WorkingDecodeRegion<CodecRegion<…>>` on reads, `RWRegion` / `from_working`
/// on writes.  Dispatch grid covers `rect.width × rect.height` threads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region(pub Rect);

/// 1-D sub-range `[start, end)` — the natural WU of Mask1D, Fft1D.
///
/// Drives 1-D range-addressed shader interface.  Dispatch grid is linear:
/// `end - start` threads.
#[derive(Clone, Debug, PartialEq, Eq)]
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Atomic;

// ── AnyWorkUnit impls ─────────────────────────────────────────────────────────

impl AnyWorkUnit for Region {
    fn to_work_unit(&self) -> WorkUnit {
        WorkUnit::Region(self.0)
    }
    fn from_work_unit(wu: &WorkUnit) -> Option<Self> {
        match wu {
            WorkUnit::Region(r) => Some(Region(*r)),
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

// ── WorkUnit (erased) ─────────────────────────────────────────────────────────

/// Type-erased work-unit — the wire format for the heterogeneous graph walk.
///
/// Nodes are stored as `Arc<dyn GpuOperation>` so demands must cross DataType
/// boundaries dynamically.  [`WorkUnit`] is the shared vocabulary: both sides
/// of an edge convert through it.  Typed [`MaterializePlan`] impls downcast
/// from this via [`AnyWorkUnit::from_work_unit`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkUnit {
    /// 2-D sub-rectangle — erased form of [`Region`].
    Region(Rect),
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
            WorkUnit::Region(r) => *r,
            WorkUnit::Atomic | WorkUnit::Range { .. } => {
                Rect::new(0, 0, w as i32, h as i32)
            }
        }
    }
}
