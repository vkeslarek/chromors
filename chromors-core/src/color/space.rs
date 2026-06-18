//! Color space definition — primaries + white point + transfer function.
//!
//! A [`ColorSpace`] bundles the three components needed to fully characterize
//! an RGB color space: the RGB primaries (gamut), the white point (reference
//! illuminant), and the transfer function (opto-electronic transfer function,
//! or OETF). This is the core type used throughout the engine to tag pixel
//! data and compute conversion matrices.

use super::primaries::{RgbPrimaries, WhitePoint};
use super::transfer::TransferFn;
use serde::{Deserialize, Serialize};

/// A color space composed of RGB primaries, a white point, and a transfer function.
///
/// The three components are independent: you can mix any primaries with any
/// white point and any transfer function. Predefined combinations for common
/// standards are provided as associated constants (e.g. [`Self::SRGB`]).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColorSpace {
    primaries: RgbPrimaries,
    white_point: WhitePoint,
    transfer: TransferFn,
}

impl ColorSpace {
    /// Creates a color space from the given primaries, white point, and transfer function.
    pub const fn new(
        primaries: RgbPrimaries,
        white_point: WhitePoint,
        transfer: TransferFn,
    ) -> Self {
        Self {
            primaries,
            white_point,
            transfer,
        }
    }

    /// Convenience: creates a linear color space (no gamma) with the given primaries and white point.
    pub const fn linear(primaries: RgbPrimaries, white_point: WhitePoint) -> Self {
        Self::new(primaries, white_point, TransferFn::Linear)
    }

    /// Returns the RGB primaries of this color space.
    pub const fn primaries(self) -> RgbPrimaries {
        self.primaries
    }

    /// Returns the white point of this color space.
    pub const fn white_point(self) -> WhitePoint {
        self.white_point
    }

    /// Returns the transfer function of this color space.
    pub const fn transfer(self) -> TransferFn {
        self.transfer
    }

    /// Returns `true` if the transfer function is [`TransferFn::Linear`].
    pub const fn is_linear(self) -> bool {
        self.transfer.is_linear()
    }

    /// Returns a copy of this color space with the transfer function replaced by [`TransferFn::Linear`].
    pub const fn as_linear(self) -> Self {
        Self::new(self.primaries, self.white_point, TransferFn::Linear)
    }

    /// Returns a copy of this color space with the given transfer function.
    pub const fn with_transfer(self, tf: TransferFn) -> Self {
        Self::new(self.primaries, self.white_point, tf)
    }

    /// Returns a copy of this color space with the given primaries.
    pub const fn with_primaries(self, p: RgbPrimaries) -> Self {
        Self::new(p, self.white_point, self.transfer)
    }

    /// Returns a copy of this color space with the given white point.
    pub const fn with_white_point(self, wp: WhitePoint) -> Self {
        Self::new(self.primaries, wp, self.transfer)
    }

    /// Constructs a color space from optional parameters, falling back to sRGB-like defaults.
    ///
    /// Defaults: BT.709 primaries, D65 white point, sRGB gamma.
    pub fn with_optional_params(
        primaries: Option<RgbPrimaries>,
        white_point: Option<WhitePoint>,
        transfer: Option<TransferFn>,
    ) -> Self {
        let p = primaries.unwrap_or(RgbPrimaries::Bt709);
        let wp = white_point.unwrap_or(WhitePoint::D65);
        let t = transfer.unwrap_or(TransferFn::SrgbGamma);
        Self::new(p, wp, t)
    }

    // --- Predefined color spaces ---

    /// sRGB: BT.709 primaries, D65 white point, sRGB transfer function.
    pub const SRGB: Self = Self::new(RgbPrimaries::Bt709, WhitePoint::D65, TransferFn::SrgbGamma);

    /// Linear sRGB: BT.709 primaries, D65 white point, identity transfer.
    pub const LINEAR_SRGB: Self = Self::linear(RgbPrimaries::Bt709, WhitePoint::D65);

    /// Rec.709: BT.709 primaries, D65 white point, Rec.709 transfer function.
    pub const REC709: Self = Self::new(
        RgbPrimaries::Bt709,
        WhitePoint::D65,
        TransferFn::Rec709Gamma,
    );

    /// Rec.2020: BT.2020 primaries, D65 white point, Rec.709 transfer function.
    pub const REC2020: Self = Self::new(
        RgbPrimaries::Bt2020,
        WhitePoint::D65,
        TransferFn::Rec709Gamma,
    );

    /// Linear Rec.2020: BT.2020 primaries, D65 white point, identity transfer.
    pub const LINEAR_REC2020: Self = Self::linear(RgbPrimaries::Bt2020, WhitePoint::D65);

    /// Adobe RGB (1998): Adobe 1998 primaries, D65 white point, 2.2 gamma.
    pub const ADOBE_RGB: Self = Self::new(
        RgbPrimaries::Adobe1998,
        WhitePoint::D65,
        TransferFn::Gamma22,
    );

    /// Display-P3: P3 primaries, D65 white point, sRGB transfer function.
    pub const DISPLAY_P3: Self =
        Self::new(RgbPrimaries::P3, WhitePoint::D65, TransferFn::SrgbGamma);

    /// Linear Display-P3: P3 primaries, D65 white point, identity transfer.
    pub const LINEAR_DISPLAY_P3: Self = Self::linear(RgbPrimaries::P3, WhitePoint::D65);

    /// DCI-P3: P3 primaries, P3-DCI white point (≈6300K), 2.6 gamma.
    pub const DCI_P3: Self = Self::new(RgbPrimaries::P3, WhitePoint::P3Dci, TransferFn::Gamma26);

    /// ProPhoto RGB: ProPhoto primaries, D50 white point, ProPhoto transfer function.
    pub const PROPHOTO: Self = Self::new(
        RgbPrimaries::ProPhoto,
        WhitePoint::D50,
        TransferFn::ProPhotoGamma,
    );

    /// ACES2065-1: AP0 primaries, D60 white point, identity transfer (linear).
    pub const ACES2065_1: Self = Self::linear(RgbPrimaries::Ap0, WhitePoint::D60);

    /// ACEScg: AP1 primaries, D60 white point, identity transfer (linear).
    pub const ACES_CG: Self = Self::linear(RgbPrimaries::Ap1, WhitePoint::D60);

    /// Adobe Wide Gamut RGB: AdobeWide primaries, D50 white point, 2.2 gamma.
    pub const ADOBE_WIDE_GAMUT: Self = Self::new(
        RgbPrimaries::AdobeWide,
        WhitePoint::D50,
        TransferFn::Gamma22,
    );

    /// Apple RGB: Apple primaries, D65 white point, 2.2 gamma (approx. 1.8).
    pub const APPLE_RGB: Self =
        Self::new(RgbPrimaries::Apple, WhitePoint::D65, TransferFn::Gamma22);

    /// CIE RGB: CIE primaries, E white point, 2.2 gamma.
    pub const CIE_RGB: Self = Self::new(RgbPrimaries::CieRgb, WhitePoint::E, TransferFn::Gamma22);

    /// Rec.2100 PQ: BT.2020 primaries, D65 white point, PQ transfer (HDR).
    pub const REC2100_PQ: Self = Self::new(RgbPrimaries::Bt2020, WhitePoint::D65, TransferFn::Pq);

    /// Rec.2100 HLG: BT.2020 primaries, D65 white point, HLG transfer (HDR).
    pub const REC2100_HLG: Self = Self::new(RgbPrimaries::Bt2020, WhitePoint::D65, TransferFn::Hlg);
}

impl ColorSpace {
    /// Encode this color space as a compact integer for storage in VipsImage metadata.
    ///
    /// Returns 0 for unrecognised spaces. Only well-known standard spaces are encoded;
    /// use 0 as a sentinel meaning "fall back to Vips interpretation".
    pub fn to_pixors_id(self) -> i32 {
        match self {
            ColorSpace::SRGB => 1,
            ColorSpace::LINEAR_SRGB => 2,
            ColorSpace::REC709 => 3,
            ColorSpace::REC2020 => 4,
            ColorSpace::LINEAR_REC2020 => 5,
            ColorSpace::ADOBE_RGB => 6,
            ColorSpace::DISPLAY_P3 => 7,
            ColorSpace::LINEAR_DISPLAY_P3 => 8,
            ColorSpace::DCI_P3 => 9,
            ColorSpace::PROPHOTO => 13,
            ColorSpace::ACES2065_1 => 10,
            ColorSpace::ACES_CG => 11,
            ColorSpace::ADOBE_WIDE_GAMUT => 12,
            ColorSpace::REC2100_PQ => 14,
            ColorSpace::REC2100_HLG => 15,
            _ => 0,
        }
    }

    /// Decode a compact integer (from Vips metadata) back to a `ColorSpace`.
    ///
    /// Returns `None` for unrecognised IDs (0 or unknown).
    pub fn from_pixors_id(id: i32) -> Option<Self> {
        match id {
            1 => Some(ColorSpace::SRGB),
            2 => Some(ColorSpace::LINEAR_SRGB),
            3 => Some(ColorSpace::REC709),
            4 => Some(ColorSpace::REC2020),
            5 => Some(ColorSpace::LINEAR_REC2020),
            6 => Some(ColorSpace::ADOBE_RGB),
            7 => Some(ColorSpace::DISPLAY_P3),
            8 => Some(ColorSpace::LINEAR_DISPLAY_P3),
            9 => Some(ColorSpace::DCI_P3),
            10 => Some(ColorSpace::ACES2065_1),
            11 => Some(ColorSpace::ACES_CG),
            12 => Some(ColorSpace::ADOBE_WIDE_GAMUT),
            13 => Some(ColorSpace::PROPHOTO),
            14 => Some(ColorSpace::REC2100_PQ),
            15 => Some(ColorSpace::REC2100_HLG),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_space_constants() {
        assert_eq!(ColorSpace::SRGB.primaries(), RgbPrimaries::Bt709);
        assert_eq!(ColorSpace::SRGB.white_point(), WhitePoint::D65);
        assert_eq!(ColorSpace::SRGB.transfer(), TransferFn::SrgbGamma);
        assert!(ColorSpace::ACES_CG.is_linear());
    }
}
