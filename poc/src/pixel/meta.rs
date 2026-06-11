//! Compressed pixel metadata descriptor.
//!
//! [`PixelMeta`] bundles the three essential properties of pixel data —
//! format, color space, and alpha policy — into a single `Copy`-able type
//! used throughout the pipeline to pass pixel metadata efficiently.

use crate::color::space::ColorSpace;
use crate::pixel::{AlphaPolicy, PixelFormat};

/// Compact descriptor for pixel data: format + color space + alpha policy.
///
/// This triplet fully defines how a pixel buffer should be interpreted:
/// - `format`: the channel layout and sample type (e.g. `Rgba8`, `GrayF32`).
/// - `color_space`: the primaries + white point + transfer function.
/// - `alpha_policy`: how alpha is stored (straight, premultiplied, or opaque).
#[derive(Debug, Clone, Copy)]
pub struct PixelMeta {
    /// The pixel format (channel layout + bit depth).
    pub format: PixelFormat,
    /// The color space of the pixel data.
    pub color_space: ColorSpace,
    /// How alpha is stored in the pixel data.
    pub alpha_policy: AlphaPolicy,
}

impl PixelMeta {
    /// Creates a new `PixelMeta` from individual components.
    pub fn new(format: PixelFormat, color_space: ColorSpace, alpha_policy: AlphaPolicy) -> Self {
        Self {
            format,
            color_space,
            alpha_policy,
        }
    }
}
