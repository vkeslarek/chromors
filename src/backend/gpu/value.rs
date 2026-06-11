//! Runtime materialization payload for the GPU computation graph.
//!
//! [`MaterializedValue`] is the result of materializing any graph node —
//! image, histogram, mask, FFT, scalar, … alike. It is deliberately generic:
//! *every* [`super::datatype::DataType`] is materialized the same way (compile
//! the fused DAG, dispatch, land the result in [`Storage`]). Image2D-specific
//! interpretation (buffer-local coordinates, `MaterializedImage`, …) lives in
//! the image datatype's own code (`data/image.rs`, `target.rs`), not here.

use std::sync::Arc;

use super::buffer::GpuBuffer;
use super::datatype::DataType;
use super::work_unit::WorkUnit;

// ── WriteMode ─────────────────────────────────────────────────────────────────

/// How a kernel writes its result into a target buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteMode {
    /// One write per dispatched thread, addressed by `region_index` —
    /// `target.write(idx, value)` / `from_working` + `codec::encode`.
    /// The default for spatial outputs (Image2D, FeatureMap, masks, FFTs).
    Positional,
    /// Scattered atomic increments into a fixed-size counter buffer —
    /// `HistogramOut { target, bin_count }`. No per-thread positional write,
    /// no `region_target` descriptor; `count` sizes the `bin_count` param.
    AtomicAccumulate { count: u32 },
}

// ── Storage ───────────────────────────────────────────────────────────────────

/// Where a materialized value's bytes live.
#[derive(Clone)]
pub enum Storage {
    /// Resident in VRAM (today: image data only).
    Vram(Arc<GpuBuffer>),
    /// Read back to host bytes (histograms, masks, FFTs, scalars, …, or an
    /// image spilled for cheap re-upload).
    Host(Vec<u8>),
}

// ── MaterializedValue ─────────────────────────────────────────────────────────

/// Runtime payload produced when a graph node is materialised.
///
/// Generic across every [`DataType`] — `datatype` + `extent` together fully
/// describe what `storage` holds; [`super::datatype::TypedData::finish`]
/// interprets the bytes through `self` (the typed datatype), so there is no
/// embedded shape tag to re-validate.
#[derive(Clone)]
pub struct MaterializedValue {
    pub storage: Storage,
    pub datatype: Arc<dyn DataType>,
    /// The work unit this materialization covers — `Region { rect, lod }` for
    /// spatially-divisible datatypes, `Range`/`Atomic` otherwise.
    pub extent: WorkUnit,
}

impl MaterializedValue {
    pub fn vram(buffer: Arc<GpuBuffer>, datatype: Arc<dyn DataType>, extent: WorkUnit) -> Self {
        MaterializedValue {
            storage: Storage::Vram(buffer),
            datatype,
            extent,
        }
    }

    pub fn host(bytes: Vec<u8>, datatype: Arc<dyn DataType>, extent: WorkUnit) -> Self {
        MaterializedValue {
            storage: Storage::Host(bytes),
            datatype,
            extent,
        }
    }

    pub fn byte_size(&self) -> u64 {
        match &self.storage {
            Storage::Vram(buffer) => buffer.byte_len,
            Storage::Host(bytes) => bytes.len() as u64,
        }
    }

    pub fn read_bytes(
        &self,
        ctx: &crate::backend::gpu::context::GpuContext,
    ) -> Result<Vec<u8>, crate::error::Error> {
        match &self.storage {
            Storage::Vram(buffer) => buffer.read_to_cpu(ctx),
            Storage::Host(bytes) => Ok(bytes.clone()),
        }
    }
}
