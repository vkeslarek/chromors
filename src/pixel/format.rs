//! Pixel format enumeration.
//!
//! [`PixelFormat`] enumerates all supported pixel layouts (channel count +
//! sample bit depth) and provides metadata queries: bytes per pixel, channel
//! count, sample width, scalar type, and the associated color model transform.

use serde::{Deserialize, Serialize};

/// Enumerates all supported pixel formats.
///
/// Format names follow the pattern `{Model}{BitDepth}`:
/// - `{Model}`: Gray, GrayA, Rgb, Rgba, Cmyk, CmykA, YCbCr, Lab, Argb
/// - `{BitDepth}`: 8 (u8), 16 (u16), F16 (half), F32 (float)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PixelFormat {
    /// 8-bit grayscale (1 channel, u8).
    Gray8,
    /// 8-bit grayscale + alpha (2 channels, u8).
    GrayA8,
    /// 8-bit RGB (3 channels, u8).
    Rgb8,
    /// 8-bit RGBA (4 channels, u8).
    Rgba8,
    /// 8-bit CMYK (4 channels, u8).
    Cmyk8,
    /// 8-bit CMYK + alpha (5 channels, u8).
    CmykA8,
    /// 8-bit YCbCr (3 channels, u8).
    YCbCr8,
    /// 8-bit CIE L*a*b* (3 channels, u8).
    Lab8,
    /// 16-bit grayscale (1 channel, u16).
    Gray16,
    /// 16-bit grayscale + alpha (2 channels, u16).
    GrayA16,
    /// 16-bit RGB (3 channels, u16).
    Rgb16,
    /// 16-bit RGBA (4 channels, u16).
    Rgba16,
    /// 16-bit CMYK (4 channels, u16).
    Cmyk16,
    /// 16-bit CMYK + alpha (5 channels, u16).
    CmykA16,
    /// 16-bit CIE L*a*b* (3 channels, u16).
    Lab16,
    /// Half-float grayscale (1 channel, f16).
    GrayF16,
    /// Half-float grayscale + alpha (2 channels, f16).
    GrayAF16,
    /// Half-float RGB (3 channels, f16).
    RgbF16,
    /// Half-float RGBA (4 channels, f16).
    RgbaF16,
    /// Half-float CMYK (4 channels, f16).
    CmykF16,
    /// Half-float CMYK + alpha (5 channels, f16).
    CmykAF16,
    /// Half-float YCbCr (3 channels, f16).
    YCbCrF16,
    /// Float grayscale (1 channel, f32).
    GrayF32,
    /// Float grayscale + alpha (2 channels, f32).
    GrayAF32,
    /// Float RGB (3 channels, f32).
    RgbF32,
    /// Float RGBA (4 channels, f32).
    RgbaF32,
    /// Float CMYK (4 channels, f32).
    CmykF32,
    /// Float CMYK + alpha (5 channels, f32).
    CmykAF32,
    /// Float YCbCr (3 channels, f32).
    YCbCrF32,
    /// 32-bit ARGB packed into a u32 (4 channels, 1 byte each).
    Argb32,

    // ── Color-model floating-point formats (one f32 per channel) ──
    /// Float CIE L*a*b* (3 channels, f32). L in [0,100], a/b unbounded (typ. [-128,128]).
    LabF32,
    /// Float CIE 1931 XYZ tristimulus (3 channels, f32).
    XyzF32,
    /// Float CIE xyY (3 channels, f32 — Y, x, y per libvips convention).
    YxyF32,
    /// Float CIE LCh cylindrical Lab (3 channels, f32). L,C,h.
    LChF32,
    /// 8-bit HSV (3 channels, u8). H wraps in [0,255]; S,V in [0,255].
    HsvU8,
    /// Float HSV (3 channels, f32). H cyclic [0,1); S,V in [0,1].
    HsvF32,
    /// Float Oklab perceptual (3 channels, f32). L in ~[0,1], a/b small.
    OklabF32,
    /// Float Oklch (cylindrical Oklab) (3 channels, f32). L,C,h.
    OkLChF32,
    /// Float scRGB — linear, unbounded HDR RGBA (4 channels, f32).
    ScRgbF32,
}

impl PixelFormat {
    /// Returns the total number of bytes per pixel.
    pub fn bytes_per_pixel(self) -> usize {
        let bytes_per_sample = match self {
            // u8 samples
            PixelFormat::Gray8
            | PixelFormat::GrayA8
            | PixelFormat::Rgb8
            | PixelFormat::Rgba8
            | PixelFormat::Argb32
            | PixelFormat::Cmyk8
            | PixelFormat::CmykA8
            | PixelFormat::YCbCr8
            | PixelFormat::Lab8
            | PixelFormat::HsvU8 => 1,
            // u16 / f16 samples
            PixelFormat::Gray16
            | PixelFormat::GrayA16
            | PixelFormat::Rgb16
            | PixelFormat::Rgba16
            | PixelFormat::GrayF16
            | PixelFormat::GrayAF16
            | PixelFormat::RgbF16
            | PixelFormat::RgbaF16
            | PixelFormat::Cmyk16
            | PixelFormat::CmykA16
            | PixelFormat::CmykF16
            | PixelFormat::CmykAF16
            | PixelFormat::YCbCrF16
            | PixelFormat::Lab16 => 2,
            // f32 samples
            PixelFormat::GrayF32
            | PixelFormat::GrayAF32
            | PixelFormat::RgbF32
            | PixelFormat::RgbaF32
            | PixelFormat::CmykF32
            | PixelFormat::CmykAF32
            | PixelFormat::YCbCrF32
            | PixelFormat::LabF32
            | PixelFormat::XyzF32
            | PixelFormat::YxyF32
            | PixelFormat::LChF32
            | PixelFormat::HsvF32
            | PixelFormat::OklabF32
            | PixelFormat::OkLChF32
            | PixelFormat::ScRgbF32 => 4,
        };
        bytes_per_sample * self.channel_count()
    }

    /// Returns the color-model transform needed to bring this format to RGB.
    ///
    /// Drives the GPU/CPU model decode (and its inverse on encode). RGB-family
    /// and gray formats need no transform (`None`).
    pub fn model_transform(self) -> crate::color::model::ColorModelTransform {
        use crate::color::model::ColorModelTransform as M;
        match self {
            PixelFormat::Cmyk8
            | PixelFormat::Cmyk16
            | PixelFormat::CmykF16
            | PixelFormat::CmykF32 => M::CmykToRgb,
            PixelFormat::CmykA8
            | PixelFormat::CmykA16
            | PixelFormat::CmykAF16
            | PixelFormat::CmykAF32 => M::CmykAToRgb,
            PixelFormat::YCbCr8 | PixelFormat::YCbCrF16 | PixelFormat::YCbCrF32 => M::YCbCrToRgb,
            PixelFormat::Lab8 | PixelFormat::Lab16 | PixelFormat::LabF32 => M::LabToRgb,
            _ => M::None,
        }
    }

    /// Returns the number of channels in this format.
    pub fn channel_count(self) -> usize {
        match self {
            PixelFormat::Gray8
            | PixelFormat::Gray16
            | PixelFormat::GrayF16
            | PixelFormat::GrayF32 => 1,
            PixelFormat::GrayA8
            | PixelFormat::GrayA16
            | PixelFormat::GrayAF16
            | PixelFormat::GrayAF32 => 2,
            PixelFormat::Rgb8
            | PixelFormat::Rgb16
            | PixelFormat::RgbF16
            | PixelFormat::RgbF32
            | PixelFormat::YCbCr8
            | PixelFormat::YCbCrF16
            | PixelFormat::YCbCrF32
            | PixelFormat::Lab8
            | PixelFormat::Lab16 => 3,
            PixelFormat::Rgba8
            | PixelFormat::Rgba16
            | PixelFormat::RgbaF16
            | PixelFormat::RgbaF32
            | PixelFormat::Argb32
            | PixelFormat::Cmyk8
            | PixelFormat::Cmyk16
            | PixelFormat::CmykF16
            | PixelFormat::CmykF32 => 4,
            PixelFormat::CmykA8
            | PixelFormat::CmykA16
            | PixelFormat::CmykAF16
            | PixelFormat::CmykAF32 => 5,
            // ── color-model formats ──
            PixelFormat::LabF32
            | PixelFormat::XyzF32
            | PixelFormat::YxyF32
            | PixelFormat::LChF32
            | PixelFormat::HsvU8
            | PixelFormat::HsvF32
            | PixelFormat::OklabF32
            | PixelFormat::OkLChF32 => 3,
            PixelFormat::ScRgbF32 => 4,
        }
    }

    /// Returns the number of channels (alias for `channel_count`).
    pub fn channels(self) -> usize {
        self.channel_count()
    }

    /// Returns whether this format has an alpha channel.
    pub fn has_alpha(self) -> bool {
        matches!(
            self,
            PixelFormat::GrayA8
                | PixelFormat::Rgba8
                | PixelFormat::CmykA8
                | PixelFormat::GrayA16
                | PixelFormat::Rgba16
                | PixelFormat::CmykA16
                | PixelFormat::GrayAF16
                | PixelFormat::RgbaF16
                | PixelFormat::CmykAF16
                | PixelFormat::GrayAF32
                | PixelFormat::RgbaF32
                | PixelFormat::CmykAF32
                | PixelFormat::Argb32
                | PixelFormat::ScRgbF32
        )
    }

    /// Returns the equivalent format with an alpha channel.
    pub fn to_f32(self) -> Self {
        match self.channel_count() {
            1 => {
                if self.has_alpha() {
                    PixelFormat::GrayAF32
                } else {
                    PixelFormat::GrayF32
                }
            }
            2 => PixelFormat::GrayAF32,
            3 => PixelFormat::RgbF32,
            4 => PixelFormat::RgbaF32,
            _ => PixelFormat::RgbaF32,
        }
    }

    pub fn with_alpha(self) -> Self {
        match self {
            PixelFormat::Gray8 => PixelFormat::GrayA8,
            PixelFormat::Rgb8 => PixelFormat::Rgba8,
            PixelFormat::Cmyk8 => PixelFormat::CmykA8,
            PixelFormat::Gray16 => PixelFormat::GrayA16,
            PixelFormat::Rgb16 => PixelFormat::Rgba16,
            PixelFormat::Cmyk16 => PixelFormat::CmykA16,
            PixelFormat::GrayF16 => PixelFormat::GrayAF16,
            PixelFormat::RgbF16 => PixelFormat::RgbaF16,
            PixelFormat::CmykF16 => PixelFormat::CmykAF16,
            PixelFormat::GrayF32 => PixelFormat::GrayAF32,
            PixelFormat::RgbF32 => PixelFormat::RgbaF32,
            PixelFormat::CmykF32 => PixelFormat::CmykAF32,
            // Formats that already have alpha or don't have a direct alpha equivalent
            // return themselves
            _ => self,
        }
    }
    /// Maximum component value for normalization (255.0 for u8, 65535.0 for u16,
    /// 1.0 for float formats). Used to scale operations that work on raw Vips
    /// pixel values so results land in the [0, 1] float range.
    pub fn component_max_f64(self) -> f64 {
        match self {
            PixelFormat::Gray8
            | PixelFormat::GrayA8
            | PixelFormat::Rgb8
            | PixelFormat::Rgba8
            | PixelFormat::Argb32
            | PixelFormat::Cmyk8
            | PixelFormat::CmykA8
            | PixelFormat::YCbCr8
            | PixelFormat::Lab8
            | PixelFormat::HsvU8 => 255.0,
            PixelFormat::Gray16
            | PixelFormat::GrayA16
            | PixelFormat::Rgb16
            | PixelFormat::Rgba16
            | PixelFormat::Cmyk16
            | PixelFormat::CmykA16
            | PixelFormat::Lab16 => 65535.0,
            // float and half-float formats are already normalized [0, 1]
            _ => 1.0,
        }
    }
}

impl crate::backend::vips::IntoVipsBandFormat for PixelFormat {
    fn into_vips_band_format(self) -> i32 {
        match self {
            PixelFormat::Gray8
            | PixelFormat::GrayA8
            | PixelFormat::Rgb8
            | PixelFormat::Rgba8
            | PixelFormat::Cmyk8
            | PixelFormat::CmykA8
            | PixelFormat::Lab8
            | PixelFormat::YCbCr8
            | PixelFormat::Argb32
            | PixelFormat::HsvU8 => 0,
            PixelFormat::Gray16
            | PixelFormat::GrayA16
            | PixelFormat::Rgb16
            | PixelFormat::Rgba16
            | PixelFormat::Cmyk16
            | PixelFormat::CmykA16
            | PixelFormat::Lab16 => 2,
            PixelFormat::GrayF16
            | PixelFormat::GrayAF16
            | PixelFormat::RgbF16
            | PixelFormat::RgbaF16
            | PixelFormat::CmykF16
            | PixelFormat::CmykAF16
            | PixelFormat::YCbCrF16 => 6,
            PixelFormat::GrayF32
            | PixelFormat::GrayAF32
            | PixelFormat::RgbF32
            | PixelFormat::RgbaF32
            | PixelFormat::CmykF32
            | PixelFormat::CmykAF32
            | PixelFormat::LabF32
            | PixelFormat::XyzF32
            | PixelFormat::YxyF32
            | PixelFormat::LChF32
            | PixelFormat::HsvF32
            | PixelFormat::OklabF32
            | PixelFormat::OkLChF32
            | PixelFormat::ScRgbF32
            | PixelFormat::YCbCrF32 => 6,
        }
    }
}

impl crate::backend::vips::FromVipsBandFormat for PixelFormat {
    fn from_vips_band_format(raw: i32, bands: i32) -> Self {
        match (raw, bands) {
            (0, 1) => PixelFormat::Gray8,
            (0, 2) => PixelFormat::GrayA8,
            (0, 3) => PixelFormat::Rgb8,
            (0, 4) => PixelFormat::Rgba8,
            (0, 5) => PixelFormat::CmykA8,
            (2, 1) => PixelFormat::Gray16,
            (2, 2) => PixelFormat::GrayA16,
            (2, 3) => PixelFormat::Rgb16,
            (2, 4) => PixelFormat::Rgba16,
            (6, 1) => PixelFormat::GrayF32,
            (6, 2) => PixelFormat::GrayAF32,
            (6, 3) => PixelFormat::RgbF32,
            (6, 4) => PixelFormat::RgbaF32,
            (8, 1) => PixelFormat::GrayF32,
            (8, 3) => PixelFormat::RgbF32,
            (8, 4) => PixelFormat::RgbaF32,
            _ => PixelFormat::Rgba8,
        }
    }
}
