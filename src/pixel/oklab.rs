//! Oklab perceptual color space pixel type.

use crate::pixel::{AlphaPolicy, Pixel};
use wide::f32x4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Oklab<T> {
    pub l: T,
    pub a: T,
    pub b: T,
}

unsafe impl bytemuck::Pod for Oklab<f32> {}
unsafe impl bytemuck::Zeroable for Oklab<f32> {}

impl<T> Oklab<T> {
    pub const fn new(l: T, a: T, b: T) -> Self {
        Self { l, a, b }
    }
}

impl Pixel for Oklab<f32> {
    fn unpack(self) -> [f32; 4] {
        [self.l, self.a, self.b, 1.0]
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Self {
            l: rgba[0],
            a: rgba[1],
            b: rgba[2],
        }
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, _aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r = rr.to_array();
        let g = gg.to_array();
        let b = bb.to_array();
        for i in 0..4 {
            out[i] = Self {
                l: r[i],
                a: g[i],
                b: b[i],
            };
        }
    }
}
