//! [`HistogramType`] — fixed-size atomic-accumulator histogram datatype.

use std::sync::Arc;

use crate::error::Error;
use crate::pixel::PixelFormat;

use super::super::context::GpuContext;
use super::super::handle::Lod;
use super::super::typed::histogram::HistogramBuffer;
use super::super::value::{MaterializedValue, Storage, WriteMode};
use super::super::work_unit::{Atomic, WorkUnitKind};
use super::{DataType, TypedData};

/// Fixed-size histogram accumulator. `bins` × u32 atomic counters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistogramType {
    pub bins: u32,
}

impl DataType for HistogramType {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn write_mode(&self) -> WriteMode {
        WriteMode::AtomicAccumulate { count: self.bins }
    }

    fn byte_size(&self, _w: u32, _h: u32, _image_format: PixelFormat) -> u64 {
        (self.bins as u64 * 4).max(64)
    }

    fn work_unit_kind(&self) -> WorkUnitKind {
        WorkUnitKind::Atomic
    }
}

impl TypedData for HistogramType {
    type Value = Arc<HistogramBuffer>;
    type WorkUnit = Atomic;

    fn finish(
        &self,
        value: &MaterializedValue,
        _lod: Lod,
        _wu: &Atomic,
        _ctx: &GpuContext,
    ) -> Result<Self::Value, Error> {
        match &value.storage {
            Storage::Host(bytes) => Ok(HistogramBuffer::from_bytes(bytes.clone(), self.bins)),
            Storage::Vram(_) => Err(Error::Gpu(
                "HistogramType::finish: expected Host, got Vram".into(),
            )),
        }
    }
}
