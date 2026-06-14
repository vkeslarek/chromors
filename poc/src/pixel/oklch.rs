//! Oklch (cylindrical Oklab) pixel type.

use crate::pixel::{AlphaPolicy, Pixel};
use wide::f32x4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OkLCh<T> {
    pub l: T,
    pub c: T,
    pub h: T,
}

unsafe impl bytemuck::Pod for OkLCh<f32> {}
unsafe impl bytemuck::Zeroable for OkLCh<f32> {}

impl<T> OkLCh<T> {
    pub const fn new(l: T, c: T, h: T) -> Self {
        Self { l, c, h }
    }
}

impl Pixel for OkLCh<f32> {
    fn unpack(self) -> [f32; 4] {
        [self.l, self.c, self.h, 1.0]
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Self {
            l: rgba[0],
            c: rgba[1],
            h: rgba[2],
        }
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, _aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r = rr.to_array();
        let g = gg.to_array();
        let b = bb.to_array();
        for i in 0..4 {
            out[i] = Self {
                l: r[i],
                c: g[i],
                h: b[i],
            };
        }
    }
}
