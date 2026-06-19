//! `Convert`'s GPU param block (`docs/native-color-management.md` §6.1.1).
//!
//! PER-BACKEND: byte-identical to `ColorConvertParams` in
//! `shaders/lib/color/params.slang` (rewritten in step 5). Built here, where
//! the §3.6 `Gpu*Id` traits are in scope, from the AGNOSTIC
//! [`crate::color::pipeline::convert_matrices`] math.

use crate::ColorModel;
use crate::Error;
use crate::TransferFn;
use crate::color::matrix::Matrix3x3;
use crate::color::pipeline::convert_matrices;
use crate::pixel::{AlphaState, PixelLayout};
use crate::{GpuAlphaId, GpuModelId, GpuTransferId, ParamBlock, ReadWrap, SlangPod, WriteWrap};

/// Param block for `Convert`'s view-wrap read/write (§5.5-§5.12, §6.1).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvertParams {
    pub transfer_src: u32,
    pub transfer_dst: u32,
    /// XYZ(D50) hub matrix, src primaries+wp -> XYZ(D50). 3 rows, each padded to vec4.
    pub a: [f32; 12],
    /// XYZ(D50) -> dst primaries+wp. 3 rows, each padded to vec4.
    pub b: [f32; 12],
    pub alpha_src: u32,
    pub alpha_dst: u32,
    pub model_src: u32,
    pub model_dst: u32,
    pub nchan_src: u32,
    pub nchan_dst: u32,
}

impl ConvertParams {
    /// Builds the params for converting `src` -> `dst`. Errors if either
    /// endpoint is `Multiband` (see [`convert_matrices`]).
    pub fn build(src: PixelLayout, dst: PixelLayout) -> Result<Self, Error> {
        let (a, b) = convert_matrices(src, dst)?;
        Ok(Self {
            transfer_src: src.color_space.transfer().gpu_transfer(),
            transfer_dst: dst.color_space.transfer().gpu_transfer(),
            a: pad_rows(a),
            b: pad_rows(b),
            alpha_src: src.alpha.gpu_alpha(),
            alpha_dst: dst.alpha.gpu_alpha(),
            model_src: src.model.gpu_model(),
            model_dst: dst.model.gpu_model(),
            nchan_src: src.channel_count() as u32,
            nchan_dst: dst.channel_count() as u32,
        })
    }

    /// A no-op conversion (sRGB RGB -> sRGB RGB, identity matrices). Used as
    /// the fallback when `build` fails so `Convert::lower` can `cx.fail(e)`
    /// without panicking on a malformed param block.
    pub fn identity() -> Self {
        Self {
            transfer_src: TransferFn::SrgbGamma.gpu_transfer(),
            transfer_dst: TransferFn::SrgbGamma.gpu_transfer(),
            a: pad_rows(Matrix3x3::IDENTITY),
            b: pad_rows(Matrix3x3::IDENTITY),
            alpha_src: AlphaState::None.gpu_alpha(),
            alpha_dst: AlphaState::None.gpu_alpha(),
            model_src: ColorModel::Rgb.gpu_model(),
            model_dst: ColorModel::Rgb.gpu_model(),
            nchan_src: 3,
            nchan_dst: 3,
        }
    }
}

/// Packs a `Matrix3x3`'s 3 rows into 3 `vec4`-aligned groups (std430 layout
/// for `float3x3` packed as `float4[3]`). The 4th component of each row is padding.
fn pad_rows(m: Matrix3x3) -> [f32; 12] {
    let mut out = [0.0f32; 12];
    for (i, slot) in out.chunks_exact_mut(4).enumerate() {
        let row = m.row(i);
        slot[..3].copy_from_slice(&row);
    }
    out
}

impl SlangPod for ConvertParams {
    const SLANG_TY: &'static str = "ColorConvertParams";
}

/// Read-side color reinterpretation (§5.12, `lib/color/interp.slang`):
/// `ColorReadView<{inner}>` runs `color_convert` on every sample read through
/// the wrapped region.
pub fn color_read_wrap(p: ConvertParams) -> ReadWrap {
    ReadWrap {
        wrapper: "ColorReadView<{inner}>".into(),
        ctor: "{ {value}, {params} }".into(),
        params: ParamBlock::from_pod("cc", &p),
        module: Some("lib.color.interp"),
    }
}

/// Write-side color reinterpretation (§5.12, `lib/color/interp.slang`):
/// `ColorWriteSink<{inner}>` runs `color_convert` on every sample before it
/// reaches the wrapped sink (codec/temp).
pub fn color_write_wrap(p: ConvertParams) -> WriteWrap {
    WriteWrap {
        wrapper: "ColorWriteSink<{inner}>".into(),
        ctor: "{ {value}, {params} }".into(),
        params: ParamBlock::from_pod("cc", &p),
        module: Some("lib.color.interp"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Storage;
    use crate::color::space::ColorSpace;

    fn layout(model: ColorModel, alpha: AlphaState, cs: ColorSpace) -> PixelLayout {
        PixelLayout {
            storage: Storage::F32,
            model,
            alpha,
            color_space: cs,
        }
    }

    #[test]
    fn identity_round_trips_through_pad_rows() {
        let p = ConvertParams::identity();
        assert_eq!(p.a[0..3], [1.0, 0.0, 0.0]);
        assert_eq!(p.a[4..7], [0.0, 1.0, 0.0]);
        assert_eq!(p.a[8..11], [0.0, 0.0, 1.0]);
        assert_eq!(p.nchan_src, 3);
        assert_eq!(p.nchan_dst, 3);
    }

    #[test]
    fn build_srgb_to_srgb() {
        let src = layout(ColorModel::Rgb, AlphaState::Straight, ColorSpace::SRGB);
        let dst = layout(ColorModel::Rgb, AlphaState::None, ColorSpace::SRGB);
        let p = ConvertParams::build(src, dst).expect("srgb->srgb");
        assert_eq!(p.nchan_src, 4);
        assert_eq!(p.nchan_dst, 3);
        assert_eq!(p.alpha_src, AlphaState::Straight.gpu_alpha());
        assert_eq!(p.alpha_dst, AlphaState::None.gpu_alpha());
    }

    #[test]
    fn build_rejects_multiband() {
        let rgb = layout(ColorModel::Rgb, AlphaState::None, ColorSpace::SRGB);
        let multi = layout(ColorModel::Multiband(2), AlphaState::None, ColorSpace::SRGB);
        assert!(ConvertParams::build(rgb, multi).is_err());
    }
}
