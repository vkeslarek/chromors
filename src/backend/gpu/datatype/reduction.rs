//! [`ScalarType`], [`PointListType`], [`FeaturesType`] — reduction and
//! feature-map output datatypes. None of these are [`super::TypedData`]
//! today (no decode path is wired up yet); they exist as `Arc<dyn DataType>`
//! tags so ops can declare them as outputs.

use crate::pixel::PixelFormat;

use super::super::work_unit::WorkUnitKind;
use super::DataType;

/// Single float scalar output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScalarType;

impl DataType for ScalarType {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn byte_size(&self, _w: u32, _h: u32, _image_format: PixelFormat) -> u64 {
        64
    }

    fn work_unit_kind(&self) -> WorkUnitKind {
        WorkUnitKind::Atomic
    }
}

/// Atomic-append coordinate list. Counter at offset 0, then (x, y) pairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointListType {
    pub capacity: u32,
}

impl DataType for PointListType {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn byte_size(&self, _w: u32, _h: u32, _image_format: PixelFormat) -> u64 {
        (4 + self.capacity as u64 * 8).max(64)
    }

    fn work_unit_kind(&self) -> WorkUnitKind {
        WorkUnitKind::Atomic
    }
}

/// Multi-channel feature map. Storage = `width × height × ceil(channels/4) × 16` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FeaturesType {
    pub channels: u32,
}

impl DataType for FeaturesType {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn byte_size(&self, w: u32, h: u32, _image_format: PixelFormat) -> u64 {
        (w as u64 * h as u64 * self.channels.div_ceil(4) as u64 * 16).max(64)
    }

    fn work_unit_kind(&self) -> WorkUnitKind {
        WorkUnitKind::Region
    }
}
