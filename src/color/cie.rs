//! CIE colorimetry types: XYZ tristimulus values and xyY (chromaticity + luminance).
//!
//! [`Xyz`] represents absolute CIE 1931 XYZ tristimulus values. [`Xyy`]
//! represents the same color in the xyY space: chromaticity (x, y) plus
//! luminance Y. Both are generic over the numeric type `T`.

use crate::color::chromaticity::Chromaticity;

/// CIE 1931 XYZ tristimulus values.
///
/// `X` and `Z` encode chromaticity information; `Y` represents luminance.
/// Generic over the numeric type `T` (typically `f32`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xyz<T> {
    /// The X tristimulus value.
    pub x: T,
    /// The Y tristimulus value (luminance).
    pub y: T,
    /// The Z tristimulus value.
    pub z: T,
}

impl<T> Xyz<T> {
    /// Creates a new XYZ value.
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

impl<T: Copy> Xyz<T> {
    /// Returns the values as a tuple `(x, y, z)`.
    pub fn as_tuple(&self) -> (T, T, T) {
        (self.x, self.y, self.z)
    }

    /// Returns the values as an array `[x, y, z]`.
    pub fn as_array(&self) -> [T; 3] {
        [self.x, self.y, self.z]
    }
}

impl Xyz<f32> {
    /// Converts from XYZ to xyY color space.
    ///
    /// Chromaticity `(x, y)` is computed as `x = X/(X+Y+Z)`, `y = Y/(X+Y+Z)`,
    /// and luminance is the Y component. Returns `(0, 0, Y)` if sum is zero.
    pub fn to_xyy(&self) -> Xyy<f32> {
        let sum = self.x + self.y + self.z;
        if sum == 0.0 {
            Xyy::new(0.0, 0.0, self.y)
        } else {
            Xyy::new(self.x / sum, self.y / sum, self.y)
        }
    }

    /// Extracts the chromaticity `(x, y)` from XYZ.
    ///
    /// Computes `x = X/(X+Y+Z)`, `y = Y/(X+Y+Z)`. Returns `(0, 0)` if sum is zero.
    pub fn to_chromaticity(&self) -> Chromaticity {
        let sum = self.x + self.y + self.z;
        if sum == 0.0 {
            Chromaticity::new(0.0, 0.0)
        } else {
            Chromaticity::new(self.x / sum, self.y / sum)
        }
    }
}

/// CIE xyY color: chromaticity `(x, y)` plus luminance `lum` (Y).
///
/// The chromaticity fields `x` and `y` are always `f32`, while the luminance
/// field `lum` is generic over `T`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Xyy<T> {
    /// The x chromaticity coordinate.
    pub x: f32,
    /// The y chromaticity coordinate.
    pub y: f32,
    /// The luminance (Y) value.
    pub lum: T,
}

impl<T> Xyy<T> {
    /// Creates a new xyY value.
    pub const fn new(x: f32, y: f32, lum: T) -> Self {
        Self { x, y, lum }
    }
}

impl Xyy<f32> {
    /// Converts from xyY to XYZ color space.
    ///
    /// Computes `X = x * lum/y`, `Y = lum`, `Z = (1-x-y) * lum/y`.
    /// Returns `(0, 0, 0)` if `y` is zero.
    pub fn to_xyz(&self) -> Xyz<f32> {
        if self.y == 0.0 {
            Xyz::new(0.0, 0.0, 0.0)
        } else {
            let factor = self.lum / self.y;
            Xyz::new(self.x * factor, self.lum, (1.0 - self.x - self.y) * factor)
        }
    }

    /// Returns the chromaticity `(x, y)` of this xyY value.
    pub fn chromaticity(&self) -> Chromaticity {
        Chromaticity::new(self.x, self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    #[test]
    fn xyz_to_xyy() {
        let xyz = Xyz::new(0.5, 0.3, 0.2);
        let xyy = xyz.to_xyy();
        let sum = 0.5 + 0.3 + 0.2;
        assert_approx_eq!(xyy.x, 0.5 / sum);
        assert_approx_eq!(xyy.y, 0.3 / sum);
        assert_approx_eq!(xyy.lum, 0.3);
    }

    #[test]
    fn xyy_to_xyz() {
        let xyy = Xyy::new(0.4, 0.3, 0.6);
        let xyz = xyy.to_xyz();
        let factor = 0.6 / 0.3;
        assert_approx_eq!(xyz.x, 0.4 * factor);
        assert_approx_eq!(xyz.y, 0.6);
        assert_approx_eq!(xyz.z, (1.0 - 0.4 - 0.3) * factor);
    }

    #[test]
    fn roundtrip_xyz_xyy() {
        let xyz = Xyz::new(0.7, 0.2, 0.1);
        let xyy = xyz.to_xyy();
        let xyz2 = xyy.to_xyz();
        assert_approx_eq!(xyz2.x, xyz.x, 1e-6);
        assert_approx_eq!(xyz2.y, xyz.y, 1e-6);
        assert_approx_eq!(xyz2.z, xyz.z, 1e-6);
    }
}
