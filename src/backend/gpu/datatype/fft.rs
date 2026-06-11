//! [`Fft1dType`] / [`Fft2dType`] — frequency-domain datatypes.

use crate::error::Error;

use super::super::context::GpuContext;
use super::super::value::{MaterializedValue, Storage};
use super::super::work_unit::{Range, Region, WorkUnit, WorkUnitKind};
use super::{DataType, TypedData};

/// 1-D FFT result (frequency domain). Each element is a complex f32 pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fft1dType {
    pub length: u32,
}

impl DataType for Fft1dType {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn byte_size(&self, _wu: &WorkUnit) -> u64 {
        (self.length as u64 * 8).max(64) // complex f32 = 8 bytes
    }

    fn work_unit_kind(&self) -> WorkUnitKind {
        WorkUnitKind::Range
    }
}

impl TypedData for Fft1dType {
    type Value = Vec<[f32; 2]>;
    type WorkUnit = Range;

    fn finish(
        &self,
        value: &MaterializedValue,
        wu: &Range,
        _ctx: &GpuContext,
    ) -> Result<Self::Value, Error> {
        let count = (wu.end - wu.start) as usize;
        match &value.storage {
            Storage::Host(bytes) => {
                let complex: &[[f32; 2]] = bytemuck::cast_slice(bytes);
                Ok(complex.iter().take(count).copied().collect())
            }
            Storage::Vram(_) => Err(Error::Gpu(
                "Fft1dType::finish: expected Host, got Vram".into(),
            )),
        }
    }
}

/// 2-D FFT result (frequency domain image). Each element is a complex f32 pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fft2dType;

impl DataType for Fft2dType {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        let WorkUnit::Region { rect, .. } = wu else {
            unreachable!("Fft2dType::work_unit_kind is Region")
        };
        (rect.width as u64 * rect.height as u64 * 8).max(64) // complex f32 = 8 bytes/pixel
    }

    fn work_unit_kind(&self) -> WorkUnitKind {
        WorkUnitKind::Region
    }
}

impl TypedData for Fft2dType {
    type Value = Vec<[f32; 2]>;
    type WorkUnit = Region;

    fn finish(
        &self,
        value: &MaterializedValue,
        wu: &Region,
        _ctx: &GpuContext,
    ) -> Result<Self::Value, Error> {
        let count = (wu.rect.width * wu.rect.height) as usize;
        match &value.storage {
            Storage::Host(bytes) => {
                let complex: &[[f32; 2]] = bytemuck::cast_slice(bytes);
                Ok(complex.iter().take(count).copied().collect())
            }
            Storage::Vram(_) => Err(Error::Gpu(
                "Fft2dType::finish: expected Host, got Vram".into(),
            )),
        }
    }
}
