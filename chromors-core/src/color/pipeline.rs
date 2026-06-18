//! AGNOSTIC color-conversion matrix math (`docs/native-color-management.md` §6.1.1).
//!
//! Pure color science: given a source and destination [`PixelLayout`], derive
//! the pair of 3x3 matrices that take a source-space linear RGB triple to the
//! XYZ(D50) hub and from the hub to destination-space linear RGB. No backend
//! types — `ConvertParams::build` (PER-BACKEND, `src/backend/gpu/color_params.rs`)
//! packs these alongside the §3.6 GPU enum ids.

use crate::color::matrix::{Matrix3x3, bradford_cat, rgb_to_xyz_matrix};
use crate::color::primaries::WhitePoint;
use crate::error::Error;
use crate::pixel::PixelLayout;

/// Derives `(A, B)` such that `B * (A * src_linear_rgb)` is dst-space linear
/// RGB: `A` maps src primaries+white point to XYZ(D50), `B` maps XYZ(D50) to
/// dst primaries+white point. Errors if either endpoint is `Multiband` (no
/// colorimetric meaning to convert) or has degenerate (singular) primaries.
pub fn convert_matrices(
    src: PixelLayout,
    dst: PixelLayout,
) -> Result<(Matrix3x3, Matrix3x3), Error> {
    if !src.model.is_colorimetric() || !dst.model.is_colorimetric() {
        return Err(Error::TypeMismatch(
            "cannot color-convert a Multiband image".into(),
        ));
    }

    let to_xyz = |cs: crate::color::space::ColorSpace| -> Result<Matrix3x3, Error> {
        rgb_to_xyz_matrix(cs.primaries(), cs.white_point())
            .map_err(|e| Error::Backend(format!("{e:?}")))
    };

    let a = to_xyz(src.color_space)?;
    let a50 = bradford_cat(src.color_space.white_point(), WhitePoint::D50).mul(&a);

    let xyz_to_dst = to_xyz(dst.color_space)?
        .inverse()
        .map_err(|e| Error::Backend(format!("{e:?}")))?;
    let b = xyz_to_dst.mul(&bradford_cat(
        WhitePoint::D50,
        dst.color_space.white_point(),
    ));

    Ok((a50, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color::model::ColorModel;
    use crate::color::space::ColorSpace;
    use crate::pixel::{AlphaState, Storage};

    fn layout(model: ColorModel, cs: ColorSpace) -> PixelLayout {
        PixelLayout {
            storage: Storage::F32,
            model,
            alpha: AlphaState::None,
            color_space: cs,
        }
    }

    #[test]
    fn srgb_to_srgb_is_identity_in_xyz() {
        let l = layout(ColorModel::Rgb, ColorSpace::SRGB);
        let (a, b) = convert_matrices(l, l).expect("identity conversion");
        let combined = b.mul(&a);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (combined.0[j][i] - expected).abs() < 1e-4,
                    "combined[{j}][{i}] = {}, expected {expected}",
                    combined.0[j][i]
                );
            }
        }
    }

    #[test]
    fn multiband_endpoint_rejected() {
        let rgb = layout(ColorModel::Rgb, ColorSpace::SRGB);
        let multi = layout(ColorModel::Multiband(3), ColorSpace::SRGB);
        assert!(matches!(
            convert_matrices(rgb, multi),
            Err(Error::TypeMismatch(_))
        ));
        assert!(matches!(
            convert_matrices(multi, rgb),
            Err(Error::TypeMismatch(_))
        ));
    }
}
