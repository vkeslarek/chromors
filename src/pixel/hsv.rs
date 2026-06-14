//! HSV (Hue/Saturation/Value) pixel type. u8 variant (0-255 per channel) and
//! f32 variant (h in [0,1] cyclical, s/v in [0,1]).

use crate::pixel::{AlphaPolicy, Pixel};
use wide::f32x4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hsv<T> {
    pub h: T,
    pub s: T,
    pub v: T,
}

unsafe impl bytemuck::Pod for Hsv<u8> {}
unsafe impl bytemuck::Zeroable for Hsv<u8> {}
unsafe impl bytemuck::Pod for Hsv<f32> {}
unsafe impl bytemuck::Zeroable for Hsv<f32> {}

impl<T> Hsv<T> {
    pub const fn new(h: T, s: T, v: T) -> Self {
        Self { h, s, v }
    }
}

impl Pixel for Hsv<u8> {
    fn unpack(self) -> [f32; 4] {
        [
            self.h as f32 / 255.0,
            self.s as f32 / 255.0,
            self.v as f32 / 255.0,
            1.0,
        ]
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Self {
            h: (rgba[0].rem_euclid(1.0) * 255.0 + 0.5) as u8,
            s: (rgba[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            v: (rgba[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        }
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, _aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r = rr.to_array();
        let g = gg.to_array();
        let b = bb.to_array();
        for i in 0..4 {
            out[i] = Self {
                h: (r[i].rem_euclid(1.0) * 255.0 + 0.5) as u8,
                s: (g[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                v: (b[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            };
        }
    }
}

impl Pixel for Hsv<f32> {
    fn unpack(self) -> [f32; 4] {
        [self.h, self.s, self.v, 1.0]
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Self {
            h: rgba[0],
            s: rgba[1],
            v: rgba[2],
        }
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, _aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r = rr.to_array();
        let g = gg.to_array();
        let b = bb.to_array();
        for i in 0..4 {
            out[i] = Self {
                h: r[i],
                s: g[i],
                v: b[i],
            };
        }
    }
}
