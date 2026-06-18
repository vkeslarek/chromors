use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use crate::backend::Backend;
use crate::color::space::ColorSpace;
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::pixel::{PixelLayout, layout_with_bands};
use crate::work_unit::{Region, WorkUnit};

#[derive(Clone, Debug, PartialEq)]
pub struct ImageKind {
    pub layout: PixelLayout,
    pub width: i32,
    pub height: i32,
}

impl ImageKind {
    pub fn new(layout: PixelLayout, width: i32, height: i32) -> Self {
        Self { layout, width, height }
    }
    pub fn dims(&self) -> (i32, i32) { (self.width, self.height) }
    pub fn color_space(&self) -> ColorSpace { self.layout.color_space }
    pub fn with_layout(&self, layout: PixelLayout) -> Self {
        Self { layout, width: self.width, height: self.height }
    }
    pub fn set_band_count(&mut self, count: i32) {
        self.layout = layout_with_bands(self.layout, count as usize);
    }
}

impl AnyKind for ImageKind {
    fn as_any(&self) -> &dyn Any { self }
    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        let bpp = self.layout.bytes_per_pixel() as u64;
        match wu { WorkUnit::Region(r) => (r.w.max(0) as u64) * (r.h.max(0) as u64) * bpp, _ => 0 }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write(format!("{:?}", self.layout).as_bytes());
        state.write_i32(self.width);
        state.write_i32(self.height);
    }
}

impl Kind for ImageKind { type WorkUnit = Region; }

pub type Image2D<B> = Data<ImageKind, B>;

impl<B: Backend> Image2D<B> {
    pub fn width(&self) -> i32 { self.spec.width }
    pub fn height(&self) -> i32 { self.spec.height }
    pub fn layout(&self) -> PixelLayout { self.spec.layout }
    pub fn color_space(&self) -> ColorSpace { self.spec.color_space() }
}

pub struct RamImageTarget;
pub struct GpuBufferTarget;
