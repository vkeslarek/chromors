//! [`GpuData`] — associates each DataType with its natural [`AnyWorkUnit`].
//!
//! The work-unit type is what drives the shader interface, not the DataType
//! directly.  `GpuData` is the bridge: a DataType marker struct implements
//! `GpuData`, declaring `type WorkUnit = Region | Range | Atomic`, and the
//! emitter/planner use `WorkUnit` to pick the correct shader wrapping.

use super::op::OutputSpec;
use super::work_unit::{Atomic, Range, Region};

// ── GpuData ───────────────────────────────────────────────────────────────────

/// A kind of data flowing through the GPU graph.
///
/// Marker structs (e.g. [`ImageData`], [`HistogramData`]) implement this trait
/// to declare their natural work-unit type.  [`MaterializePlan<Out>`] is then
/// generic over `Out: GpuData`, giving each operation a typed request/response
/// surface keyed to its actual output kind.
pub trait GpuData: 'static {
    /// The work-unit type this DataType uses to express demands.
    /// Determines the shader interface (rect-addressed, range, or atomic).
    type WorkUnit: super::work_unit::AnyWorkUnit;

    /// Construct the "full output" work-unit from an [`OutputSpec`].
    /// Used as the seed demand when materializing a root node.
    fn full_work_unit(spec: &OutputSpec) -> Self::WorkUnit;
}

// ── DataType marker structs ───────────────────────────────────────────────────

/// 2-D pixel image. WorkUnit = [`Region`].
pub struct ImageData;

/// Fixed-size histogram accumulator. WorkUnit = [`Atomic`].
pub struct HistogramData;

/// Atomic-append coordinate list. WorkUnit = [`Atomic`].
pub struct PointListData;

/// Single float scalar. WorkUnit = [`Atomic`].
pub struct ScalarData;

/// Multi-channel feature map (spatial). WorkUnit = [`Region`].
pub struct FeatureMapData;

/// 1-D mask (separable convolution kernel). WorkUnit = [`Range`].
pub struct Mask1DData;

/// 2-D mask (morphology, compass). WorkUnit = [`Region`].
pub struct Mask2DData;

/// 1-D FFT result (frequency domain). WorkUnit = [`Range`].
pub struct Fft1DData;

/// 2-D FFT result (frequency domain image). WorkUnit = [`Region`].
pub struct Fft2DData;

// ── GpuData impls ─────────────────────────────────────────────────────────────

use crate::geometry::Rect;

impl GpuData for ImageData {
    type WorkUnit = Region;
    fn full_work_unit(spec: &OutputSpec) -> Region {
        let (w, h) = spec.image_dims().unwrap_or((0, 0));
        Region(Rect::new(0, 0, w as i32, h as i32))
    }
}

impl GpuData for HistogramData {
    type WorkUnit = Atomic;
    fn full_work_unit(_: &OutputSpec) -> Atomic {
        Atomic
    }
}

impl GpuData for PointListData {
    type WorkUnit = Atomic;
    fn full_work_unit(_: &OutputSpec) -> Atomic {
        Atomic
    }
}

impl GpuData for ScalarData {
    type WorkUnit = Atomic;
    fn full_work_unit(_: &OutputSpec) -> Atomic {
        Atomic
    }
}

impl GpuData for FeatureMapData {
    type WorkUnit = Region;
    fn full_work_unit(spec: &OutputSpec) -> Region {
        let (w, h) = spec.image_dims().unwrap_or((0, 0));
        Region(Rect::new(0, 0, w as i32, h as i32))
    }
}

impl GpuData for Mask1DData {
    type WorkUnit = Range;
    fn full_work_unit(spec: &OutputSpec) -> Range {
        let len = match spec {
            OutputSpec::Image { width, .. } => *width,
            _ => 0,
        };
        Range { start: 0, end: len }
    }
}

impl GpuData for Mask2DData {
    type WorkUnit = Region;
    fn full_work_unit(spec: &OutputSpec) -> Region {
        let (w, h) = spec.image_dims().unwrap_or((0, 0));
        Region(Rect::new(0, 0, w as i32, h as i32))
    }
}

impl GpuData for Fft1DData {
    type WorkUnit = Range;
    fn full_work_unit(spec: &OutputSpec) -> Range {
        let len = match spec {
            OutputSpec::Image { width, .. } => *width,
            _ => 0,
        };
        Range { start: 0, end: len }
    }
}

impl GpuData for Fft2DData {
    type WorkUnit = Region;
    fn full_work_unit(spec: &OutputSpec) -> Region {
        let (w, h) = spec.image_dims().unwrap_or((0, 0));
        Region(Rect::new(0, 0, w as i32, h as i32))
    }
}
