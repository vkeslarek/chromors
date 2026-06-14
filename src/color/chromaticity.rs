//! CIE xy chromaticity coordinate.
//!
//! A [`Chromaticity`] is a point in the CIE 1931 xy chromaticity diagram,
//! representing the normalized color of a light source independent of its
//! luminance. z is derived as `1 - x - y`.

use serde::{Deserialize, Serialize};

/// A point in CIE xy chromaticity space.
///
/// All values are compile-time constants in the codebase, so `Eq` is safe
/// (no NaNs).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Chromaticity {
    /// The x coordinate of the chromaticity.
    pub x: f32,
    /// The y coordinate of the chromaticity.
    pub y: f32,
}

impl Eq for Chromaticity {} // safe: f32 values are compile-time constants, never NaN

impl std::hash::Hash for Chromaticity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // safe: f32 values are compile-time constants, never NaN
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

impl Chromaticity {
    /// Creates a new chromaticity point with the given `x` and `y` coordinates.
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Returns the z coordinate, derived as `1.0 - x - y`.
    pub fn z(&self) -> f32 {
        1.0 - self.x - self.y
    }

    /// Converts to XYZ tristimulus values with given luminance Y = 1.
    ///
    /// Returns `(X, Y, Z)` where `X = x/y`, `Y = 1`, `Z = z/y`.
    pub fn to_xyz(&self) -> (f32, f32, f32) {
        let x = self.x / self.y;
        let y = 1.0;
        let z = self.z() / self.y;
        (x, y, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    #[test]
    fn chromaticity_z() {
        let c = Chromaticity::new(0.3, 0.4);
        assert_approx_eq!(c.z(), 0.3);
    }

    #[test]
    fn chromaticity_to_xyz() {
        let c = Chromaticity::new(0.3, 0.4);
        let (x, y, z) = c.to_xyz();
        assert_approx_eq!(x, 0.3 / 0.4);
        assert_approx_eq!(y, 1.0);
        assert_approx_eq!(z, (1.0 - 0.3 - 0.4) / 0.4);
    }
}
