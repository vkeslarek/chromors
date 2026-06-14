use crate::pixel::{AlphaPolicy, Component, Pixel};
use half::f16;
use wide::f32x4;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgb<T: Component> {
    pub r: T,
    pub g: T,
    pub b: T,
}

impl<T: Component> Rgb<T> {
    pub const fn new(r: T, g: T, b: T) -> Self {
        Self { r, g, b }
    }
}

impl Rgb<f32> {
    pub fn min(self, other: Self) -> Self {
        Rgb {
            r: self.r.min(other.r),
            g: self.g.min(other.g),
            b: self.b.min(other.b),
        }
    }
}

impl std::ops::Add for Rgb<f32> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Rgb {
            r: self.r + rhs.r,
            g: self.g + rhs.g,
            b: self.b + rhs.b,
        }
    }
}

impl std::ops::Div for Rgb<f32> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Rgb {
            r: self.r / rhs.r,
            g: self.g / rhs.g,
            b: self.b / rhs.b,
        }
    }
}

impl std::ops::Mul<f32> for Rgb<f32> {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Rgb {
            r: self.r * rhs,
            g: self.g * rhs,
            b: self.b * rhs,
        }
    }
}

impl std::ops::Div<f32> for Rgb<f32> {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        let inv = 1.0 / rhs;
        self * inv
    }
}

impl std::ops::Add<f32> for Rgb<f32> {
    type Output = Self;
    fn add(self, rhs: f32) -> Self {
        Rgb {
            r: self.r + rhs,
            g: self.g + rhs,
            b: self.b + rhs,
        }
    }
}

impl Rgb<f32> {
    pub const ONE: Self = Rgb {
        r: 1.0,
        g: 1.0,
        b: 1.0,
    };
    pub const ZERO: Self = Rgb {
        r: 0.0,
        g: 0.0,
        b: 0.0,
    };
}

unsafe impl<T: Component> bytemuck::Pod for Rgb<T> {}
unsafe impl<T: Component> bytemuck::Zeroable for Rgb<T> {}

impl Pixel for Rgb<f16> {
    fn unpack(self) -> [f32; 4] {
        [self.r.to_f32(), self.g.to_f32(), self.b.to_f32(), 1.0]
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, _aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r: [f32; 4] = rr.into();
        let g: [f32; 4] = gg.into();
        let b: [f32; 4] = bb.into();
        for i in 0..4 {
            out[i] = Rgb {
                r: f16::from_f32(r[i]),
                g: f16::from_f32(g[i]),
                b: f16::from_f32(b[i]),
            };
        }
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Rgb {
            r: f16::from_f32(rgba[0]),
            g: f16::from_f32(rgba[1]),
            b: f16::from_f32(rgba[2]),
        }
    }
}

impl Pixel for Rgb<f32> {
    fn unpack(self) -> [f32; 4] {
        [self.r, self.g, self.b, 1.0]
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, _aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r: [f32; 4] = rr.into();
        let g: [f32; 4] = gg.into();
        let b: [f32; 4] = bb.into();
        for i in 0..4 {
            out[i] = Rgb {
                r: r[i],
                g: g[i],
                b: b[i],
            };
        }
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Rgb {
            r: rgba[0],
            g: rgba[1],
            b: rgba[2],
        }
    }
}

impl Pixel for Rgb<u8> {
    fn unpack(self) -> [f32; 4] {
        [
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
            1.0,
        ]
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Rgb {
            r: (rgba[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            g: (rgba[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            b: (rgba[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        }
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, _aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r = rr.to_array();
        let g = gg.to_array();
        let b = bb.to_array();
        for i in 0..4 {
            out[i] = Rgb {
                r: (r[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                g: (g[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
                b: (b[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            };
        }
    }
}

impl Pixel for Rgb<u16> {
    fn unpack(self) -> [f32; 4] {
        [
            self.r as f32 / 65535.0,
            self.g as f32 / 65535.0,
            self.b as f32 / 65535.0,
            1.0,
        ]
    }
    fn pack_one(rgba: [f32; 4], _mode: AlphaPolicy) -> Self {
        Rgb {
            r: (rgba[0].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
            g: (rgba[1].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
            b: (rgba[2].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
        }
    }
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, _aa: f32x4, _mode: AlphaPolicy, out: &mut [Self]) {
        let r = rr.to_array();
        let g = gg.to_array();
        let b = bb.to_array();
        for i in 0..4 {
            out[i] = Rgb {
                r: (r[i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
                g: (g[i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
                b: (b[i].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16,
            };
        }
    }
}
