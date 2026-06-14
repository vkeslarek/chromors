//! scRGB pixel type — linear, unbounded HDR floating-point RGB(A).

use crate::pixel::{AlphaPolicy, Pixel};
use wide::f32x4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScRgb<T> {
    pub r: T,
    pub g: T,
    pub b: T,
    pub a: T,
}

unsafe impl bytemuck::Pod for ScRgb<f32> {}
unsafe impl bytemuck::Zeroable for ScRgb<f32> {}

impl<T> ScRgb<T> {
    pub const fn new(r: T, g: T, b: T, a: T) -> Self {
        Self { r, g, b, a }
    }
}

impl Pixel for ScRgb<f32> {
    fn unpack(self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Self {
            r: rgba[0],
            g: rgba[1],
            b: rgba[2],
            a: rgba[3],
        }
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r = rr.to_array();
        let g = gg.to_array();
        let b = bb.to_array();
        let a = aa.to_array();
        for i in 0..4 {
            out[i] = Self {
                r: r[i],
                g: g[i],
                b: b[i],
                a: a[i],
            };
        }
    }
}
