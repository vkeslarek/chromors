//! RGB primaries and white points.
//!
//! [`RgbPrimaries`] defines the red, green, and blue chromaticity coordinates
//! that determine an RGB color space's gamut. [`WhitePoint`] defines the
//! reference illuminant (chromaticity of the "achromatic" point).
//!
//! Based on reference data from kolor, brucelindbloom.com, and ASTM E308-01.

use serde::{Deserialize, Serialize};

use crate::color::chromaticity::Chromaticity;

/// A set of primary colors that define an RGB color space's gamut.
///
/// Each variant defines the CIE xy chromaticity coordinates of the red, green,
/// and blue primaries. Use [`Self::chromaticities()`] to retrieve them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub enum RgbPrimaries {
    /// No primaries (placeholder — all coordinates are zero).
    None,
    /// BT.709 / sRGB primaries.
    Bt709,
    /// BT.2020 (Rec.2020) wide-gamut primaries.
    Bt2020,
    /// ACES2065-1 primaries (AP0 — the largest practical gamut).
    Ap0,
    /// ACEScg primaries (AP1 — working space gamut).
    Ap1,
    /// P3 primaries (used by DCI-P3, Display-P3, and variations).
    P3,
    /// Adobe RGB (1998) primaries.
    Adobe1998,
    /// Adobe Wide Gamut RGB primaries.
    AdobeWide,
    /// Apple RGB primaries.
    Apple,
    /// ProPhoto RGB (ROMM) primaries.
    ProPhoto,
    /// CIE RGB primaries (historical).
    CieRgb,
    /// Identity primaries (represents XYZ color space).
    CieXyz,
    /// Custom primaries defined by user-provided chromaticities.
    Custom {
        /// Chromaticity of the red primary.
        red: Chromaticity,
        /// Chromaticity of the green primary.
        green: Chromaticity,
        /// Chromaticity of the blue primary.
        blue: Chromaticity,
    },
}

impl RgbPrimaries {
    /// Returns the chromaticities `(x, y)` of the red, green, and blue primaries
    /// as a `[Chromaticity; 3]` array.
    pub fn chromaticities(&self) -> [Chromaticity; 3] {
        match self {
            Self::None => [
                Chromaticity::new(0.0, 0.0),
                Chromaticity::new(0.0, 0.0),
                Chromaticity::new(0.0, 0.0),
            ],
            Self::Bt709 => [
                Chromaticity::new(0.64, 0.33),
                Chromaticity::new(0.30, 0.60),
                Chromaticity::new(0.15, 0.06),
            ],
            Self::Bt2020 => [
                Chromaticity::new(0.708, 0.292),
                Chromaticity::new(0.170, 0.797),
                Chromaticity::new(0.131, 0.046),
            ],
            Self::Ap0 => [
                Chromaticity::new(0.7347, 0.2653),
                Chromaticity::new(0.0000, 1.0000),
                Chromaticity::new(0.0001, -0.0770),
            ],
            Self::Ap1 => [
                Chromaticity::new(0.713, 0.293),
                Chromaticity::new(0.165, 0.830),
                Chromaticity::new(0.128, 0.044),
            ],
            Self::P3 => [
                Chromaticity::new(0.680, 0.320),
                Chromaticity::new(0.265, 0.690),
                Chromaticity::new(0.150, 0.060),
            ],
            Self::Adobe1998 => [
                Chromaticity::new(0.64, 0.33),
                Chromaticity::new(0.21, 0.71),
                Chromaticity::new(0.15, 0.06),
            ],
            Self::AdobeWide => [
                Chromaticity::new(0.735, 0.265),
                Chromaticity::new(0.115, 0.826),
                Chromaticity::new(0.157, 0.018),
            ],
            Self::Apple => [
                Chromaticity::new(0.625, 0.34),
                Chromaticity::new(0.28, 0.595),
                Chromaticity::new(0.155, 0.07),
            ],
            Self::ProPhoto => [
                Chromaticity::new(0.734699, 0.265301),
                Chromaticity::new(0.159597, 0.840403),
                Chromaticity::new(0.036598, 0.000105),
            ],
            Self::CieRgb => [
                Chromaticity::new(0.7350, 0.2650),
                Chromaticity::new(0.2740, 0.7170),
                Chromaticity::new(0.1670, 0.0090),
            ],
            Self::CieXyz => [
                Chromaticity::new(1.0, 0.0),
                Chromaticity::new(0.0, 1.0),
                Chromaticity::new(0.0, 0.0),
            ],
            Self::Custom { red, green, blue } => [*red, *green, *blue],
        }
    }
}

/// Defines the white point ("achromatic point") of a color space.
///
/// Each variant provides the CIE xy chromaticity of the reference illuminant
/// via [`Self::xy()`], and the corresponding XYZ tristimulus values (with Y=1)
/// via [`Self::xyz()`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub enum WhitePoint {
    /// No white point (placeholder — all coordinates are zero).
    None,
    /// CIE standard illuminant A (incandescent / tungsten, 2856K).
    A,
    /// CIE standard illuminant B (direct sunlight at noon, ~4874K).
    B,
    /// CIE standard illuminant C (average daylight, ~6774K).
    C,
    /// Equal energy illuminant (x = y = 1/3).
    E,
    /// D50 (ICC profile PCS, horizon light, 5003K).
    D50,
    /// D55 (mid‑morning / mid‑afternoon daylight, 5503K).
    D55,
    /// D60 (daylight, used by ACES, ~6000K).
    D60,
    /// D65 (daylight, used by sRGB, Rec.709, Adobe RGB, Display‑P3, 6504K).
    D65,
    /// D75 (north sky daylight, 7504K).
    D75,
    /// P3‑DCI theater projector white (greenish, ~6300K).
    P3Dci,
    /// CIE F2 (cool white fluorescent).
    F2,
    /// CIE F7 (daylight fluorescent, D65 simulator).
    F7,
    /// CIE F11 (Ultralume 40, Philips TL84 narrow-band fluorescent).
    F11,
    /// Custom white point defined by user-provided chromaticity.
    Custom(Chromaticity),
}

impl WhitePoint {
    /// Returns the CIE xy chromaticity of this white point.
    pub fn xy(&self) -> Chromaticity {
        match self {
            Self::None => Chromaticity::new(0.0, 0.0),
            // Values from ASTM E308-01 (via kolor and brucelindbloom.com)
            Self::A => Chromaticity::new(0.44757, 0.40745),
            Self::B => Chromaticity::new(0.34842, 0.35161),
            Self::C => Chromaticity::new(0.31006, 0.31616),
            Self::E => Chromaticity::new(1.0 / 3.0, 1.0 / 3.0),
            Self::D50 => Chromaticity::new(0.3457, 0.3585),
            Self::D55 => Chromaticity::new(0.3324, 0.3474),
            Self::D60 => Chromaticity::new(0.32168, 0.33767),
            Self::D65 => Chromaticity::new(0.3127, 0.3290),
            Self::D75 => Chromaticity::new(0.2990, 0.3149),
            Self::P3Dci => Chromaticity::new(0.314, 0.351),
            Self::F2 => Chromaticity::new(0.37208, 0.37529),
            Self::F7 => Chromaticity::new(0.31285, 0.32918),
            Self::F11 => Chromaticity::new(0.38052, 0.37713),
            Self::Custom(chroma) => *chroma,
        }
    }

    /// Returns the XYZ tristimulus values of this white point with Y = 1.
    ///
    /// Computed from the chromaticity: `X = x/y`, `Y = 1`, `Z = (1-x-y)/y`.
    /// Returns `[0, 0, 0]` for [`WhitePoint::None`] to avoid division by zero.
    pub fn xyz(&self) -> [f32; 3] {
        let xy = self.xy();
        if xy.y.abs() <= 1e-12 {
            return [0.0, 0.0, 0.0];
        }
        let x = xy.x / xy.y;
        let y = 1.0;
        let z = (1.0 - xy.x - xy.y) / xy.y;
        [x, y, z]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    #[test]
    fn white_point_xyz() {
        let d65 = WhitePoint::D65;
        let xy = d65.xy();
        assert_approx_eq!(xy.x, 0.3127, 1e-4);
        assert_approx_eq!(xy.y, 0.3290, 1e-4);

        let xyz = d65.xyz();
        // Check that Y = 1
        assert_approx_eq!(xyz[1], 1.0, 1e-6);
        // Known approximate values for D65 XYZ
        assert_approx_eq!(xyz[0], 0.95047, 3e-4);
        assert_approx_eq!(xyz[2], 1.08883, 3e-4);
    }

    #[test]
    fn primaries_chromaticities() {
        let prim = RgbPrimaries::Bt709;
        let chroma = prim.chromaticities();
        assert_approx_eq!(chroma[0].x, 0.64, 1e-6);
        assert_approx_eq!(chroma[0].y, 0.33, 1e-6);
        assert_approx_eq!(chroma[1].x, 0.30, 1e-6);
        assert_approx_eq!(chroma[1].y, 0.60, 1e-6);
        assert_approx_eq!(chroma[2].x, 0.15, 1e-6);
        assert_approx_eq!(chroma[2].y, 0.06, 1e-6);
    }
}
