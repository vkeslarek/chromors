pub mod image;
pub mod histogram;
pub mod mask;
pub mod fft;

pub use self::image::*;
pub use histogram::*;
pub use mask::*;
pub use fft::*;

use super::context::GpuContext;
use super::value::{GraphValue, ValueKind};
use super::work_unit::AnyWorkUnit;
use crate::error::Error;

pub use super::op::{InputEncoder, OutputCodec, OutputDecoder};

/// Trait implemented per DataType — typed request surface for the GPU graph.
///
/// Each DataType declares its natural [`WorkUnit`] (`Region`, `Range`, or
/// `Atomic`).  That WorkUnit — not the DataType — drives the shader interface:
/// rect-addressed storage, 1-D range, or indivisible accumulator.
///
/// Metadata that varies per DataType (bin count, mask length, etc.) belongs on
/// the struct fields, not in the WorkUnit.  The WorkUnit encodes *shape*; the
/// struct encodes *configuration*.
///
/// [`WorkUnit`]: super::work_unit::WorkUnit
pub trait GpuData: Send + Sync + 'static {
    /// The materialized payload callers receive.
    type Value: Clone + Send + Sync;

    /// The natural division strategy for this DataType.
    ///
    /// `Image` / `Mask2D` / `Fft2D` / `FeatureMap` → [`Region`] (2-D rects).
    /// `Histogram` / `Scalar` / `PointList`         → [`Atomic`] (indivisible).
    /// `Mask1D` / `Fft1D`                           → [`Range`] (1-D extents).
    ///
    /// [`Region`]: super::work_unit::Region
    /// [`Atomic`]: super::work_unit::Atomic
    /// [`Range`]: super::work_unit::Range
    type WorkUnit: AnyWorkUnit;

    /// Declare what kind of graph value this DataType produces (for buffer sizing).
    fn value_kind(&self) -> ValueKind;

    /// Slang wrapper the emitter generates around each input slot for kernels
    /// that produce this DataType.  Index 0 = primary, 1+ = extras.
    ///
    /// Default: `WorkingDecodeRegion` for image/histogram DataTypes; `MaskRegion`
    /// for mask types; `ComplexRegion` for FFT types.
    fn input_encoders(&self, num_inputs: usize) -> Vec<InputEncoder> {
        let enc = match self.value_kind() {
            ValueKind::Mask1D { .. } | ValueKind::Mask2D { .. } => InputEncoder::MaskRegion,
            ValueKind::Fft1D { .. } | ValueKind::Fft2D => InputEncoder::ComplexRegion,
            _ => InputEncoder::WorkingDecodeRegion,
        };
        vec![enc; num_inputs]
    }

    /// Slang wrapper the emitter generates around writing the kernel's output
    /// for this DataType.
    ///
    /// Default: `WorkingEncodeRegion { codec: None }` for image output;
    /// `HistogramOut` for histogram; `RWMaskRegion` / `RWComplexRegion` for
    /// mask / FFT.  `ImageData` overrides this to carry explicit codec params.
    fn output_decoder(&self) -> OutputDecoder {
        match self.value_kind() {
            ValueKind::Histogram { .. } => OutputDecoder::HistogramOut,
            ValueKind::Mask1D { .. } | ValueKind::Mask2D { .. } => OutputDecoder::RWMaskRegion,
            ValueKind::Fft1D { .. } | ValueKind::Fft2D => OutputDecoder::RWComplexRegion,
            _ => OutputDecoder::WorkingEncodeRegion { codec: None },
        }
    }

    /// Convert a materialised [`GraphValue`] to the typed payload.
    fn finish(
        &self,
        value: &GraphValue,
        lod: super::Lod,
        wu: &Self::WorkUnit,
        ctx: &GpuContext,
    ) -> Result<Self::Value, Error>;
}
