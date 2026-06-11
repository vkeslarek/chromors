//! Transfer functions (OETF/EOTF) for color spaces.
//!
//! Each variant in [`TransferFn`] provides a pair of mutually inverse functions:
//! `decode` (encoded → linear light, the electro-optical transfer function, EOTF)
//! and `encode` (linear light → encoded, the opto-electronic transfer function, OETF).
//!
//! Supported transfer functions:
//! - Linear (identity)
//! - sRGB (piecewise power law)
//! - Rec.709 (piecewise power law with different knee)
//! - Simple power-law gammas: 2.2, 2.4, 2.6
//! - ProPhoto (piecewise, 1.8 gamma)
//! - PQ (SMPTE ST 2084 — HDR perceptual quantizer)
//! - HLG (Hybrid Log-Gamma — scene-referred HDR)

use serde::{Deserialize, Serialize};

/// Invertible mapping between encoded and linear light values.
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransferFn {
    /// Identity: linear light and encoded values are the same.
    Linear,
    /// sRGB piecewise transfer function (IEC 61966-2-1).
    SrgbGamma,
    /// Rec.709 piecewise transfer function (ITU-R BT.709).
    Rec709Gamma,
    /// Power law with exponent 2.2.
    Gamma22,
    /// Power law with exponent 2.4.
    Gamma24,
    /// Power law with exponent 2.6.
    Gamma26,
    /// ProPhoto RGB transfer function (piecewise, 1.8 gamma).
    ProPhotoGamma,
    /// PQ (SMPTE ST 2084) — perceptual quantizer for HDR content.
    Pq,
    /// HLG (Hybrid Log-Gamma) — scene-referred HDR transfer function.
    Hlg,
}

impl TransferFn {
    /// Decode one encoded `f32` value to linear light (EOTF).
    ///
    /// Input is typically in `[0, 1]` for SDR; HDR functions (PQ, HLG) can
    /// produce values > 1.
    pub fn decode(self, x: f32) -> f32 {
        match self {
            Self::Linear => x,
            Self::SrgbGamma => srgb_decode(x),
            Self::Rec709Gamma => rec709_decode(x),
            Self::Gamma22 => x.powf(2.2),
            Self::Gamma24 => x.powf(2.4),
            Self::Gamma26 => x.powf(2.6),
            Self::ProPhotoGamma => prophoto_decode(x),
            Self::Pq => pq_decode(x),
            Self::Hlg => hlg_decode(x),
        }
    }

    /// Encode one linear `f32` value to non-linear representation (OETF).
    ///
    /// Input linear value; output in `[0, 1]` for SDR, may exceed for HDR.
    pub fn encode(self, y: f32) -> f32 {
        match self {
            Self::Linear => y,
            Self::SrgbGamma => srgb_encode(y),
            Self::Rec709Gamma => rec709_encode(y),
            Self::Gamma22 => y.max(0.0).powf(1.0 / 2.2),
            Self::Gamma24 => y.max(0.0).powf(1.0 / 2.4),
            Self::Gamma26 => y.max(0.0).powf(1.0 / 2.6),
            Self::ProPhotoGamma => prophoto_encode(y),
            Self::Pq => pq_encode(y),
            Self::Hlg => hlg_encode(y),
        }
    }

    /// Returns `true` if this is the [`TransferFn::Linear`] variant.
    pub const fn is_linear(self) -> bool {
        matches!(self, Self::Linear)
    }

    /// Approximate power-law gamma exponent for use with `vips_gamma`.
    ///
    /// Returns `None` for complex piecewise functions (sRGB, Rec.709, PQ, HLG)
    /// where a simple power law is not an accurate approximation.
    pub fn approximate_gamma(self) -> Option<f64> {
        match self {
            Self::Gamma22 => Some(2.2),
            Self::Gamma24 => Some(2.4),
            Self::Gamma26 => Some(2.6),
            Self::ProPhotoGamma => Some(1.8),
            _ => None,
        }
    }

    /// Map a gamma decode exponent value `g` (where `g = 1/γ`) to a known
    /// transfer function, or return `None`.
    ///
    /// Used primarily when parsing embedded ICC or metadata gamma values.
    pub fn from_gamma(g: f32) -> Option<Self> {
        if (g - 1.0 / 2.2).abs() < 0.01 {
            Some(Self::Gamma22)
        } else if (g - 1.0 / 2.4).abs() < 0.01 {
            Some(Self::Gamma24)
        } else if (g - 1.0 / 2.2).abs() < 0.05 {
            Some(Self::Gamma22)
        } else if (g - 1.0).abs() < 0.01 {
            Some(Self::Linear)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Transfer function implementations
// ---------------------------------------------------------------------------

/// sRGB EOTF: encoded → linear.
fn srgb_decode(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB OETF: linear → encoded.
fn srgb_encode(y: f32) -> f32 {
    if y <= 0.0031308 {
        12.92 * y
    } else {
        1.055 * y.powf(1.0 / 2.4) - 0.055
    }
}

/// Rec.709 EOTF: encoded → linear.
fn rec709_decode(x: f32) -> f32 {
    if x < 0.081 {
        x / 4.5
    } else {
        ((x + 0.099) / 1.099).powf(1.0 / 0.45)
    }
}

/// Rec.709 OETF: linear → encoded.
fn rec709_encode(y: f32) -> f32 {
    if y < 0.018 {
        4.5 * y
    } else {
        1.099 * y.powf(0.45) - 0.099
    }
}

/// ProPhoto RGB EOTF: encoded → linear.
fn prophoto_decode(x: f32) -> f32 {
    if x <= 1.0 / 32.0 {
        x / 16.0
    } else {
        x.powf(1.8)
    }
}

/// ProPhoto RGB OETF: linear → encoded.
fn prophoto_encode(y: f32) -> f32 {
    if y <= 0.001953125 {
        16.0 * y
    } else {
        y.powf(1.0 / 1.8)
    }
}

// PQ (SMPTE ST 2084) constants
const PQ_M1: f32 = 2610.0 / 16384.0;
const PQ_M2: f32 = (2523.0 / 4096.0) * 128.0;
const PQ_C1: f32 = 3424.0 / 4096.0;
const PQ_C2: f32 = (2413.0 / 4096.0) * 32.0;
const PQ_C3: f32 = (2392.0 / 4096.0) * 32.0;

/// PQ (ST 2084) EOTF: encoded → linear (absolute luminance in cd/m², scaled to [0, 1]).
fn pq_decode(x: f32) -> f32 {
    let xm2 = x.max(0.0).powf(1.0 / PQ_M2);
    ((xm2 - PQ_C1).max(0.0) / (PQ_C2 - PQ_C3 * xm2)).powf(1.0 / PQ_M1)
}

/// PQ (ST 2084) OETF: linear → encoded.
fn pq_encode(y: f32) -> f32 {
    let ym1 = y.max(0.0).powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * ym1) / (1.0 + PQ_C3 * ym1)).powf(PQ_M2)
}

// HLG constants
const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.559_910_7;

/// HLG EOTF: signal `[0, 1]` → scene-referred linear `[0, 1]`.
fn hlg_decode(x: f32) -> f32 {
    if x <= 0.5 {
        (x * x) / 3.0
    } else {
        (((x - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

/// HLG OETF: scene-referred linear `[0, 1]` → signal `[0, 1]`.
fn hlg_encode(y: f32) -> f32 {
    let y = y.max(0.0);
    if y <= 1.0 / 12.0 {
        (3.0 * y).sqrt()
    } else {
        HLG_A * (12.0 * y - HLG_B).ln() + HLG_C
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    /// Asserts that `decode(encode(x)) == x` and `encode(decode(x)) == x`
    /// for a set of sample values throughout `[0, 1]`.
    fn assert_inverse(tf: TransferFn) {
        for x in [0.0_f32, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0] {
            assert_approx_eq!(tf.encode(tf.decode(x)), x, 1e-4);
            assert_approx_eq!(tf.decode(tf.encode(x)), x, 1e-4);
        }
    }

    #[test]
    fn linear_inverse() {
        assert_inverse(TransferFn::Linear);
    }
    #[test]
    fn srgb_inverse() {
        assert_inverse(TransferFn::SrgbGamma);
    }
    #[test]
    fn rec709_inverse() {
        assert_inverse(TransferFn::Rec709Gamma);
    }
    #[test]
    fn gamma22_inverse() {
        assert_inverse(TransferFn::Gamma22);
    }
    #[test]
    fn gamma24_inverse() {
        assert_inverse(TransferFn::Gamma24);
    }
    #[test]
    fn gamma26_inverse() {
        assert_inverse(TransferFn::Gamma26);
    }
    #[test]
    fn prophoto_inverse() {
        assert_inverse(TransferFn::ProPhotoGamma);
    }
    #[test]
    fn pq_inverse() {
        assert_inverse(TransferFn::Pq);
    }
    #[test]
    fn hlg_inverse() {
        assert_inverse(TransferFn::Hlg);
    }
}
