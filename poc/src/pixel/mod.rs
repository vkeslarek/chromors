//! Pixel types, formats, and the [`Pixel`] conversion trait.
//!
//! The pixel module defines the central [`Pixel`] trait that bridges concrete
//! pixel types (RGB, RGBA, CMYK, etc.) to the engine's internal `[f32; 4]`
//! RGBA intermediate representation. All pixel formats go through this
//! representation during color space conversion.
//!
//! # Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | `component` | [`Component`] trait — numeric channel abstraction |
//! | `format` | [`PixelFormat`] enum — all supported pixel layouts |
//! | `meta` | [`PixelMeta`] — compressed pixel descriptor (format + space + alpha) |
//! | `rgb` | [`Rgb`] — three-channel RGB pixels |
//! | `rgba` | [`Rgba`] — four-channel RGBA pixels |
//! | `cmyk` | [`Cmyk`], [`CmykA`] — CMYK pixel types |
//! | `gray` | [`Gray`], [`GrayAlpha`] — grayscale pixel types |
//! | `lab` | [`Lab`] — CIE L*a*b* pixel type |
//! | `ycbcr` | [`YCbCr`] — YCbCr pixel type |
//! | `pack` | Internal pack/unpack helpers |

use bytemuck::Pod;
use serde::{Deserialize, Serialize};
use wide::f32x4;

// ---------------------------------------------------------------------------
// Component trait
// ---------------------------------------------------------------------------

mod component;
pub use component::Component;

// ---------------------------------------------------------------------------
// Alpha policy — runtime param controlling premultiplication on pack
// ---------------------------------------------------------------------------

/// Controls how alpha is handled during pixel packing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlphaPolicy {
    /// Multiply RGB channels by alpha before packing.
    PremultiplyOnPack,
    /// Keep RGB and alpha independent (straight alpha).
    Straight,
    /// Destination has no alpha channel; RGB is premultiplied, alpha discarded on pack.
    OpaqueDrop,
}

impl AlphaPolicy {
    /// Shader `AlphaPolicy` discriminant (Straight=0, PremultiplyOnPack=1,
    /// OpaqueDrop=2). The Rust enum order differs, so this maps explicitly —
    /// keep in sync with `lib/pixel.slang`.
    pub fn to_shader(self) -> u32 {
        match self {
            AlphaPolicy::Straight => 0,
            AlphaPolicy::PremultiplyOnPack => 1,
            AlphaPolicy::OpaqueDrop => 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel trait — unified pack/unpack between concrete type ↔ [f32;4]
// ---------------------------------------------------------------------------

/// Bidirectional conversion between a concrete pixel type and the `[f32;4]`
/// intermediate RGBA representation used by the conversion pipeline.
///
/// `unpack`: pixel → straight linear `[r, g, b, a]` (source side, unpremuls if needed).
/// `pack`: post-matrix+encode `[r, g, b, a]` → pixel (destination side).
pub trait Pixel: Copy + Pod {
    /// Unpacks a single pixel into straight `[r, g, b, a]` in `f32`.
    fn unpack(self) -> [f32; 4];

    /// Unpacks 4 consecutive pixels into four `f32x4` SIMD registers:
    /// `(r, g, b, a)` each holding 4 lanes.
    fn unpack_x4(s: &[Self]) -> (f32x4, f32x4, f32x4, f32x4) {
        let mut r = [0.0_f32; 4];
        let mut g = [0.0_f32; 4];
        let mut b = [0.0_f32; 4];
        let mut a = [0.0_f32; 4];
        for i in 0..4 {
            let [rr, gg, bb, aa] = s[i].unpack();
            r[i] = rr;
            g[i] = gg;
            b[i] = bb;
            a[i] = aa;
        }
        (
            f32x4::from(r),
            f32x4::from(g),
            f32x4::from(b),
            f32x4::from(a),
        )
    }

    /// Packs 4 pixels from SIMD registers into an output slice, applying the
    /// given alpha policy.
    fn pack_x4(rr: f32x4, gg: f32x4, bb: f32x4, aa: f32x4, mode: AlphaPolicy, out: &mut [Self]);

    /// Packs a single pixel from `[r, g, b, a]` in `f32`, applying the given
    /// alpha policy.
    fn pack_one(rgba: [f32; 4], mode: AlphaPolicy) -> Self;

    /// Splits into normalized `Rgb<f32>` + alpha using `scale`.
    fn split_norm(self, scale: f32) -> (Rgb<f32>, f32) {
        let c = self.unpack();
        let inv = 1.0 / scale;
        (
            Rgb {
                r: c[0] * inv,
                g: c[1] * inv,
                b: c[2] * inv,
            },
            c[3],
        )
    }

    /// Joins normalized `Rgb<f32>` + alpha back into `Self` via `pack_one`.
    fn join_norm(rgb: Rgb<f32>, alpha: f32, scale: f32, mode: AlphaPolicy) -> Self {
        Self::pack_one([rgb.r * scale, rgb.g * scale, rgb.b * scale, alpha], mode)
    }
}

// ---------------------------------------------------------------------------
// Sub-modules
// ---------------------------------------------------------------------------

pub mod cmyk;
pub mod format;
pub mod gray;
pub mod hsv;
pub mod lab;
pub mod lch;
pub mod meta;
pub mod oklab;
pub mod oklch;
mod pack;
pub mod rgb;
pub mod rgba;
pub mod scrgb;
pub mod xyz;
pub mod ycbcr;
pub mod yxy;

pub use cmyk::{Cmyk, CmykA};
pub use format::PixelFormat;
pub use gray::{Gray, GrayAlpha};
pub use hsv::Hsv;
pub use lab::Lab;
pub use lch::LCh;
pub use meta::PixelMeta;
pub use oklab::Oklab;
pub use oklch::OkLCh;
pub use rgb::Rgb;
pub use rgba::Rgba;
pub use scrgb::ScRgb;
pub use xyz::Xyz;
pub use ycbcr::YCbCr;
pub use yxy::Yxy;

pub use half::f16;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_policy_is_copy() {
        let p = AlphaPolicy::PremultiplyOnPack;
        let _q = p;
        assert_eq!(p, AlphaPolicy::PremultiplyOnPack);
    }

    #[test]
    fn pixel_roundtrip_u8() {
        let orig: [u8; 4] = [128, 64, 32, 255];
        let unpacked = orig.unpack();
        let repacked = <[u8; 4]>::pack_one(unpacked, AlphaPolicy::Straight);
        assert_eq!(orig, repacked);
    }
}
