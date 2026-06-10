//! Datatype vocabulary for the GPU computation graph.
//!
//! [`DataType`] is the single source of truth for "what kind of value does
//! this graph node produce" — it replaces the old closed `ValueKind` enum.
//! Every [`super::graph::GraphNode`] carries an `Arc<dyn DataType>`. New
//! datatypes are added by writing a struct in its own submodule and
//! implementing this trait — no central enum to edit.
//!
//! [`TypedData`] is the static, request-side counterpart used by
//! [`super::request::GpuRequest`] to decode a [`super::value::MaterializedValue`]
//! into a typed payload (`Arc<ImageBuffer>`, `Arc<HistogramBuffer>`, `Vec<f32>`, …).
//!
//! Concrete datatype structs live one-kind-per-file: [`image`] (`ImageType`),
//! [`histogram`] (`HistogramType`), [`mask`] (`Mask1dType`/`Mask2dType`),
//! [`fft`] (`Fft1dType`/`Fft2dType`), and [`reduction`]
//! (`ScalarType`/`PointListType`/`FeaturesType`). Each struct, plus its
//! `DataType`/`TypedData`/`Sourceable` impls, lives together in that file.
//! The cross-cutting capability traits (`DataType`, `TypedData`,
//! `Sourceable`, `Targetable`, `Executable`) live here in `mod.rs`.

use std::fmt::Debug;
use std::sync::Arc;

use crate::geometry::Rect;
use crate::pixel::PixelFormat;

use super::context::GpuContext;
use super::handle::Lod;
use super::value::{MaterializedValue, Storage, WriteMode};
use super::work_unit::{AnyWorkUnit, WorkUnitKind};
use crate::error::Error;

pub mod fft;
pub mod histogram;
pub mod image;
pub mod mask;
pub mod reduction;

pub use fft::{Fft1dType, Fft2dType};
pub use histogram::HistogramType;
pub use image::ImageType;
pub use mask::{Mask1dType, Mask2dType};
pub use reduction::{FeaturesType, PointListType, ScalarType};

// ── DataType ──────────────────────────────────────────────────────────────────

/// Object-safe datatype descriptor — lives on every [`super::graph::GraphNode`]
/// as `Arc<dyn DataType>`.
///
/// Carries only the structural information needed to allocate buffers, emit
/// shader code, and seed the inverse-region walk — no runtime payload.
///
/// Slang wrapper choice (`InputEncoder`/`OutputDecoder`) is decided per
/// *operation* (`GpuOperation::input_encoders`/`output_decoder`), not here —
/// see `op.rs`. `DataType` only describes the produced value's shape/size.
pub trait DataType: Send + Sync + Debug + 'static {
    /// Downcast support — used by capability impls (e.g. histogram pull) that
    /// need to recover a concrete datatype's configuration (`bins`, `length`, …)
    /// from a node's `Arc<dyn DataType>`.
    ///
    /// No default body: a default `{ self }` is only valid for `Self: Sized`,
    /// which would exclude this method from the vtable. Each concrete
    /// datatype implements it as a one-line `{ self }`.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Returns `true` if this node kind needs a float4 `RWRegion` intermediate
    /// temp buffer in the fused shader. Non-image outputs (histograms, masks,
    /// FFTs, scalars, …) write directly to their target and get no temp.
    fn needs_fused_temp(&self) -> bool {
        false
    }

    /// How the emitter wraps writes to a buffer of this datatype.
    fn write_mode(&self) -> WriteMode {
        WriteMode::Positional
    }

    /// Byte size of the GPU output buffer for a node of this datatype.
    ///
    /// `w`/`h` are the resolved output-rect dimensions (pixels) for spatially
    /// divisible datatypes (Image2D, Features, Fft2D); `image_format` is the
    /// resolved pixel format for image outputs (decided by the layout pass,
    /// not embedded in `ImageType` — a node's declared `ImageType` describes
    /// its *working-space* shape, the final target format may differ).
    /// Self-describing datatypes (Histogram, masks, Fft1D, Scalar, PointList,
    /// Mask2D) ignore both `w`/`h`/`image_format` and use their own fixed extent.
    fn byte_size(&self, w: u32, h: u32, image_format: PixelFormat) -> u64;

    /// This datatype's natural division strategy — `Region` (2-D rects),
    /// `Range` (1-D extents), or `Atomic` (indivisible).
    ///
    /// Used by the readback fork to pick the [`super::work_unit::WorkUnit`]
    /// variant for a resolved node, so it never hardcodes `Region` for
    /// non-image roots.
    fn work_unit_kind(&self) -> WorkUnitKind;
}

/// Typed decode surface used by [`super::request::GpuRequest`]. Static,
/// request-side. Replaces the old `GpuData::{Value, finish}`.
pub trait TypedData: DataType + Sized {
    /// The materialized payload callers receive.
    type Value: Clone + Send + Sync;

    /// The natural division strategy for this datatype — `Region` (2-D
    /// rects), `Range` (1-D extents), or `Atomic` (indivisible).
    type WorkUnit: AnyWorkUnit;

    /// Convert a materialised [`MaterializedValue`] to the typed payload.
    fn finish(
        &self,
        value: &MaterializedValue,
        lod: Lod,
        wu: &Self::WorkUnit,
        ctx: &GpuContext,
    ) -> Result<Self::Value, Error>;
}

// ── Sourceable ────────────────────────────────────────────────────────────────

/// Datatypes that can be a graph leaf — provide pixels into the fused graph
/// from a [`super::source::GpuSource`].
///
/// Today only [`ImageType`] is sourceable: [`super::source::GpuSource`] /
/// [`super::source::AnyGpuSource`] are themselves image-only. This is the
/// datatype-side counterpart to [`Targetable::pull`] — generic glue
/// (`Storage` out), no rect/buffer-layout reconstruction (that stays in
/// `materialize.rs`).
pub trait Sourceable: DataType {
    fn fetch_region(
        &self,
        src: &super::source::GpuSource,
        rect: Rect,
        lod: Lod,
        ctx: &Arc<GpuContext>,
    ) -> Result<Storage, Error>;
}

// ── Targetable ────────────────────────────────────────────────────────────────

/// Datatypes that can terminate a pull from the graph.
///
/// Blanket-implemented for every [`TypedData`]: materialize via
/// [`super::request::GpuRequest`] and decode through [`TypedData::finish`] is
/// the same sequence regardless of datatype — only `Self::Value` differs.
/// Collapses the per-datatype `GpuRequest::new(...).materialize()` boilerplate
/// that target.rs previously hand-rolled for histograms.
pub trait Targetable: TypedData + Clone {
    fn pull(
        &self,
        node: &super::handle::GraphNodeHandle,
        lod: Lod,
        wu: &Self::WorkUnit,
    ) -> Result<Self::Value, Error> {
        let request = super::request::GpuRequest::new(
            node.graph.clone(),
            node.ctx.cache.clone(),
            node.root_id,
            node.ctx.clone(),
            lod,
            self.clone(),
            wu.clone(),
        );
        request.materialize()
    }
}

impl<D: TypedData + Clone> Targetable for D {}

// ── Executable ────────────────────────────────────────────────────────────────

/// Datatypes that can be produced by executing a [`super::op::TypedOperation`].
///
/// Blanket-implemented for every [`DataType`]: emitting a node and wrapping
/// the result in a fresh [`super::handle::GraphNodeHandle`] is identical
/// regardless of what the op produces — `O::Output` (this type) only pins
/// down *which* `GraphNodeHandle` flavor (`Image2D`, `Histogram`, …) the caller
/// is allowed to wrap the result as.
pub trait Executable: DataType + Sized {
    fn execute<O>(op: &O, node: &super::handle::GraphNodeHandle) -> super::handle::GraphNodeHandle
    where
        O: super::op::TypedOperation<Output = Self> + Clone + 'static,
    {
        super::builder::GraphBuilder::build(op, &[node])
    }
}

impl<D: DataType + Sized> Executable for D {}
