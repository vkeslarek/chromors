//! `PixelLayout` — orthogonal storage × model × alpha × color space descriptor
//! (`docs/native-color-management.md` §3.4).

use serde::{Deserialize, Serialize};

use crate::color::model::ColorModel;
use crate::color::space::ColorSpace;
use crate::pixel::{AlphaState, Storage};

/// The single source of truth carried by `ImageKind`: storage (sample
/// quantization), color model (channel meaning), alpha state, and color
/// space (primaries/white point/transfer), as four independent axes
/// (`docs/native-color-management.md` §3.4/§4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PixelLayout {
    /// How each channel sample is quantized in memory.
    pub storage: Storage,
    /// What the channels mean.
    pub model: ColorModel,
    /// Whether/how an alpha channel is present.
    pub alpha: AlphaState,
    /// Primaries + white point + transfer. Meaningful for RGB-family/Gray;
    /// for Lab/Xyz it pins the connection white point; ignored for
    /// `Multiband`.
    pub color_space: ColorSpace,
}

impl PixelLayout {
    /// Total channel count: color channels + alpha (if any).
    pub const fn channel_count(self) -> usize {
        self.model.color_channels() + self.alpha.extra_channels()
    }

    /// Total bytes per pixel: `channel_count * bytes_per_sample`.
    pub const fn bytes_per_pixel(self) -> usize {
        self.storage.bytes_per_sample() * self.channel_count()
    }

    /// Whether this layout has an alpha channel.
    pub fn has_alpha(self) -> bool {
        !matches!(self.alpha, AlphaState::None)
    }

    /// `storage.component_max()` as `f64`, for params that need double
    /// precision (e.g. boolean/relational constants).
    pub fn component_max_f64(self) -> f64 {
        self.storage.component_max() as f64
    }

    /// Returns a copy of this layout with `storage` replaced, all other axes
    /// unchanged.
    pub const fn with_storage(self, storage: Storage) -> Self {
        Self { storage, ..self }
    }

    /// Returns a copy of this layout with storage set to `F32`.
    pub const fn to_f32(self) -> Self {
        self.with_storage(Storage::F32)
    }
}

/// Derives a `PixelLayout` with `n` channels from `base`, preserving
/// `storage`/`color_space` (`docs/native-color-management.md` §6.3):
/// `n == 1` -> `Gray` (no alpha), `n == 3` -> `Rgb` (no alpha), `n == 4` ->
/// `Rgb` + `Straight` alpha (Rgba), otherwise -> `Multiband(n)` (no alpha).
pub fn layout_with_bands(base: PixelLayout, n: usize) -> PixelLayout {
    let (model, alpha) = match n {
        1 => (ColorModel::Gray, AlphaState::None),
        3 => (ColorModel::Rgb, AlphaState::None),
        4 => (ColorModel::Rgb, AlphaState::Straight),
        _ => (ColorModel::Multiband(n as u8), AlphaState::None),
    };
    PixelLayout {
        model,
        alpha,
        ..base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pixel::AlphaState;

    #[test]
    fn channel_count_and_bytes_per_pixel() {
        let rgba8 = PixelLayout {
            storage: Storage::U8,
            model: ColorModel::Rgb,
            alpha: AlphaState::Straight,
            color_space: ColorSpace::SRGB,
        };
        assert_eq!(rgba8.channel_count(), 4);
        assert_eq!(rgba8.bytes_per_pixel(), 4);
        assert!(rgba8.has_alpha());

        let gray_f32 = PixelLayout {
            storage: Storage::F32,
            model: ColorModel::Gray,
            alpha: AlphaState::None,
            color_space: ColorSpace::SRGB,
        };
        assert_eq!(gray_f32.channel_count(), 1);
        assert_eq!(gray_f32.bytes_per_pixel(), 4);
        assert!(!gray_f32.has_alpha());

        let cmyka16 = PixelLayout {
            storage: Storage::U16,
            model: ColorModel::Cmyk,
            alpha: AlphaState::Straight,
            color_space: ColorSpace::SRGB,
        };
        assert_eq!(cmyka16.channel_count(), 5);
        assert_eq!(cmyka16.bytes_per_pixel(), 10);
    }

    #[test]
    fn layout_with_bands_maps_known_counts() {
        let base = PixelLayout {
            storage: Storage::F32,
            model: ColorModel::Rgb,
            alpha: AlphaState::Straight,
            color_space: ColorSpace::SRGB,
        };

        let gray = layout_with_bands(base, 1);
        assert_eq!(gray.model, ColorModel::Gray);
        assert_eq!(gray.alpha, AlphaState::None);

        let rgb = layout_with_bands(base, 3);
        assert_eq!(rgb.model, ColorModel::Rgb);
        assert_eq!(rgb.alpha, AlphaState::None);

        let rgba = layout_with_bands(base, 4);
        assert_eq!(rgba.model, ColorModel::Rgb);
        assert_eq!(rgba.alpha, AlphaState::Straight);

        let multi = layout_with_bands(base, 7);
        assert_eq!(multi.model, ColorModel::Multiband(7));
        assert_eq!(multi.alpha, AlphaState::None);
        assert_eq!(multi.channel_count(), 7);

        // storage/color_space preserved
        assert_eq!(gray.storage, Storage::F32);
        assert_eq!(gray.color_space, ColorSpace::SRGB);
    }

    #[test]
    fn to_f32_and_with_storage() {
        let u8_rgb = PixelLayout {
            storage: Storage::U8,
            model: ColorModel::Rgb,
            alpha: AlphaState::None,
            color_space: ColorSpace::SRGB,
        };
        let f32_rgb = u8_rgb.to_f32();
        assert_eq!(f32_rgb.storage, Storage::F32);
        assert_eq!(f32_rgb.model, ColorModel::Rgb);

        let u16_rgb = u8_rgb.with_storage(Storage::U16);
        assert_eq!(u16_rgb.storage, Storage::U16);
    }
}
