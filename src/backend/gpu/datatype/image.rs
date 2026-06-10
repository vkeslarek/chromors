//! [`ImageType`] — the 2-D pixel image datatype.

use std::sync::Arc;

use crate::color::space::ColorSpace;
use crate::error::Error;
use crate::geometry::Rect;
use crate::pixel::{AlphaPolicy, PixelFormat, PixelMeta};

use super::super::buffer::ImageBuffer;
use super::super::context::GpuContext;
use super::super::handle::Lod;
use super::super::source::{AnyGpuSource, GpuSource};
use super::super::value::{MaterializedValue, Storage};
use super::super::work_unit::{Region, WorkUnitKind};
use super::{DataType, Sourceable, TypedData};

/// A 2-D pixel image. Gray images are represented here with a Gray
/// [`PixelFormat`] — there is no separate Gray variant.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageType {
    pub color_space: ColorSpace,
    pub format: PixelFormat,
}

impl DataType for ImageType {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn needs_fused_temp(&self) -> bool {
        true
    }

    fn byte_size(&self, w: u32, h: u32, image_format: PixelFormat) -> u64 {
        let bpp = image_format.bytes_per_pixel() as u64;
        (w as u64 * h as u64 * bpp).max(64)
    }

    fn work_unit_kind(&self) -> WorkUnitKind {
        WorkUnitKind::Region
    }
}

impl Sourceable for ImageType {
    fn fetch_region(
        &self,
        src: &GpuSource,
        rect: Rect,
        lod: Lod,
        ctx: &Arc<GpuContext>,
    ) -> Result<Storage, Error> {
        let buf = src.fetch_region(rect, lod, ctx)?;
        Ok(Storage::Vram(buf.buffer.clone()))
    }
}

impl TypedData for ImageType {
    type Value = Arc<ImageBuffer>;
    type WorkUnit = Region;

    fn finish(
        &self,
        value: &MaterializedValue,
        _lod: Lod,
        wu: &Region,
        _ctx: &GpuContext,
    ) -> Result<Self::Value, Error> {
        let rect = wu.rect;
        let meta = PixelMeta::new(self.format, self.color_space, AlphaPolicy::Straight);
        match &value.storage {
            Storage::Vram(buffer) => Ok(ImageBuffer::from_raw(
                buffer.buffer.clone(),
                rect.width as u32,
                rect.height as u32,
                meta,
            )),
            Storage::Host(_) => Err(Error::Gpu(
                "ImageType::finish: expected Vram, got Host".into(),
            )),
        }
    }
}
