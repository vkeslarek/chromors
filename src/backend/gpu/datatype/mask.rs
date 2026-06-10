//! [`Mask1dType`] / [`Mask2dType`] — separable and 2-D convolution mask datatypes.

use crate::error::Error;
use crate::pixel::PixelFormat;

use super::super::context::GpuContext;
use super::super::handle::Lod;
use super::super::value::{MaterializedValue, Storage};
use super::super::work_unit::{Range, Region, WorkUnitKind};
use super::{DataType, TypedData};

/// 1-D Mask (e.g. for separable convolution).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mask1dType {
    pub length: u32,
}

impl DataType for Mask1dType {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn byte_size(&self, _w: u32, _h: u32, _image_format: PixelFormat) -> u64 {
        (self.length as u64 * 4).max(64) // f32 masks
    }

    fn work_unit_kind(&self) -> WorkUnitKind {
        WorkUnitKind::Range
    }
}

impl TypedData for Mask1dType {
    type Value = Vec<f32>;
    type WorkUnit = Range;

    fn finish(
        &self,
        value: &MaterializedValue,
        _lod: Lod,
        wu: &Range,
        _ctx: &GpuContext,
    ) -> Result<Self::Value, Error> {
        let count = (wu.end - wu.start) as usize;
        match &value.storage {
            Storage::Host(bytes) => {
                let floats: &[f32] = bytemuck::cast_slice(bytes);
                Ok(floats.iter().take(count).copied().collect())
            }
            Storage::Vram(_) => Err(Error::Gpu(
                "Mask1dType::finish: expected Host, got Vram".into(),
            )),
        }
    }
}

/// 2-D Mask (e.g. for morphology/compass kernels).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mask2dType {
    pub width: u32,
    pub height: u32,
}

impl DataType for Mask2dType {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn byte_size(&self, _w: u32, _h: u32, _image_format: PixelFormat) -> u64 {
        (self.width as u64 * self.height as u64 * 4).max(64)
    }

    fn work_unit_kind(&self) -> WorkUnitKind {
        WorkUnitKind::Region
    }
}

impl TypedData for Mask2dType {
    type Value = Vec<f32>;
    type WorkUnit = Region;

    fn finish(
        &self,
        value: &MaterializedValue,
        _lod: Lod,
        wu: &Region,
        _ctx: &GpuContext,
    ) -> Result<Self::Value, Error> {
        let count = (wu.rect.width * wu.rect.height) as usize;
        match &value.storage {
            Storage::Host(bytes) => {
                let floats: &[f32] = bytemuck::cast_slice(bytes);
                Ok(floats.iter().take(count).copied().collect())
            }
            Storage::Vram(_) => Err(Error::Gpu(
                "Mask2dType::finish: expected Host, got Vram".into(),
            )),
        }
    }
}
