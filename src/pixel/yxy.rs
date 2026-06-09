//! CIE xyY pixel type (chromaticity x, y + luminance Y).

use crate::pixel::{AlphaPolicy, Pixel};
use wide::f32x4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Yxy<T> {
    pub y_lum: T,
    pub x: T,
    pub y: T,
}

unsafe impl bytemuck::Pod for Yxy<f32> {}
unsafe impl bytemuck::Zeroable for Yxy<f32> {}

impl<T> Yxy<T> {
    pub const fn new(y_lum: T, x: T, y: T) -> Self {
        Self { y_lum, x, y }
    }
}

impl Pixel for Yxy<f32> {
    fn unpack(self) -> [f32; 4] {
        [self.y_lum, self.x, self.y, 1.0]
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Self {
            y_lum: rgba[0],
            x: rgba[1],
            y: rgba[2],
        }
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, _aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r = rr.to_array();
        let g = gg.to_array();
        let b = bb.to_array();
        for i in 0..4 {
            out[i] = Self {
                y_lum: r[i],
                x: g[i],
                y: b[i],
            };
        }
    }
}
