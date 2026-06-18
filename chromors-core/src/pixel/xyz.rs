//! CIE 1931 XYZ pixel type (storage-only; colorimetry math lives in `color::cie`).

use crate::pixel::{AlphaPolicy, Pixel};
use wide::f32x4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xyz<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

unsafe impl bytemuck::Pod for Xyz<f32> {}
unsafe impl bytemuck::Zeroable for Xyz<f32> {}

impl<T> Xyz<T> {
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl Pixel for Xyz<f32> {
    fn unpack(self) -> [f32; 4] {
        [self.x, self.y, self.z, 1.0]
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Self {
            x: rgba[0],
            y: rgba[1],
            z: rgba[2],
        }
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, _aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r = rr.to_array();
        let g = gg.to_array();
        let b = bb.to_array();
        for i in 0..4 {
            out[i] = Self {
                x: r[i],
                y: g[i],
                z: b[i],
            };
        }
    }
}
