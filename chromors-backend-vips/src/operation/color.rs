use crate::custom::run_custom;
use crate::custom::{CustomRegion, VipsCustomOperation};
use crate::{IntoVipsBandFormat, VipsBackend, VipsBuilder, VipsHandle};
use chromors_core::color::matrix::Matrix3x3;
use chromors_core::color::model::ColorModel;
use chromors_core::color::space::ColorSpace;
use chromors_core::color::transfer::TransferFn;
use chromors_core::pixel::{AlphaState, PixelLayout};
use chromors_core::*;
use std::sync::Arc;

/// `true` if both endpoints have a faithful vips `VipsInterpretation` — the
/// native `colourspace`-based chain ([`vips_native_convert`]) round-trips
/// exactly through it. Otherwise the conversion needs the CPU XYZ-hub
/// custom-region fallback ([`cpu_convert_region`]).
fn vips_can_convert(src: PixelLayout, dst: PixelLayout) -> bool {
    crate::space::to_vips_interpretation(src.model, src.color_space).is_some()
        && crate::space::to_vips_interpretation(dst.model, dst.color_space).is_some()
}

/// Native fast path: `unpremultiply -> colourspace(interp) -> premultiply/
/// flatten -> addalpha -> cast`, driven by [`PixelLayout`] via
/// [`to_vips_interpretation`] (port of `src/color/convert.rs::ColorConversion`,
/// `docs/native-color-management.md` §6.1.4).
fn vips_native_convert(
    h: &VipsHandle,
    src: PixelLayout,
    dst: PixelLayout,
) -> Result<VipsHandle, Error> {
    let mut img = h.clone();

    let from_premultiplied = matches!(src.alpha, AlphaState::Premultiplied);
    let to_straight = matches!(dst.alpha, AlphaState::Straight);
    let to_premultiplied = matches!(dst.alpha, AlphaState::Premultiplied);
    let to_opaque = matches!(dst.alpha, AlphaState::None);

    let has_alpha = unsafe { crate::ffi::vips_image_hasalpha(img.ptr) } != 0;

    if from_premultiplied && (to_straight || to_opaque) && has_alpha {
        let mut op = crate::gobject::VipsGObject::new(b"unpremultiply\0")?;
        op.set_image("in", img.ptr);
        img = op.run()?;
    }

    let src_interp = crate::space::to_vips_interpretation(src.model, src.color_space);
    let dst_interp = crate::space::to_vips_interpretation(dst.model, dst.color_space);
    if src_interp != dst_interp {
        let interp = dst_interp.expect("vips_can_convert guarantees dst has a vips interpretation");
        let mut op = crate::gobject::VipsGObject::new(b"colourspace\0")?;
        op.set_image("in", img.ptr);
        op.set_int("space", interp);
        img = op.run()?;
    }

    let has_alpha_now = unsafe { crate::ffi::vips_image_hasalpha(img.ptr) } != 0;
    if to_premultiplied && !from_premultiplied && has_alpha_now {
        let mut op = crate::gobject::VipsGObject::new(b"premultiply\0")?;
        op.set_image("in", img.ptr);
        img = op.run()?;
    } else if to_opaque && has_alpha_now {
        let mut op = crate::gobject::VipsGObject::new(b"flatten\0")?;
        op.set_image("in", img.ptr);
        img = op.run()?;
    }

    let bands_now = unsafe { crate::ffi::vips_image_get_bands(img.ptr) };
    let target_bands = dst.channel_count() as i32;
    if target_bands > bands_now && !has_alpha_now && target_bands == bands_now + 1 {
        let mut op = crate::gobject::VipsGObject::new(b"addalpha\0")?;
        op.set_image("in", img.ptr);
        img = op.run()?;
    }

    let cur_format = unsafe { crate::ffi::vips_image_get_format(img.ptr) };
    let target_format = dst.storage.into_vips_band_format();
    if cur_format != target_format {
        let mut op = crate::gobject::VipsGObject::new(b"cast\0")?;
        op.set_image("in", img.ptr);
        op.set_int("format", target_format);
        img = op.run()?;
    }

    Ok(img)
}

/// Faithful fallback for spaces vips can't represent (P3, AdobeRGB, ACEScg,
/// arbitrary primaries): a [`VipsCustomOperation`] region processor running
/// `TransferFn::decode -> Matrix3x3 A -> Matrix3x3 B -> TransferFn::encode`,
/// the same XYZ(D50)-hub math as the GPU `color_convert` (§6.1.4). Alpha
/// passes through unchanged.
fn cpu_convert_region(
    h: &VipsHandle,
    src: PixelLayout,
    dst: PixelLayout,
) -> Result<VipsHandle, Error> {
    let same_shape = src.model == ColorModel::Rgb
        && dst.model == ColorModel::Rgb
        && src.storage == dst.storage
        && src.alpha == dst.alpha;
    if !same_shape {
        return Err(Error::Backend(format!(
            "Convert (vips CPU fallback): unsupported {src:?} -> {dst:?}; \
             the CPU custom-region path only supports same-shape RGB(A) \
             color-space changes (got differing storage/model/alpha)"
        )));
    }

    let (a, b) = chromors_core::color::pipeline::convert_matrices(src, dst)?;
    run_custom(
        h,
        CpuConvertOp {
            a,
            b,
            src_tf: src.color_space.transfer(),
            dst_tf: dst.color_space.transfer(),
        },
    )
}

/// Per-pixel XYZ(D50)-hub color-space conversion, dispatched to the region's
/// concrete `Rgb`/`Rgba` storage type by [`crate::dispatch_format`].
struct CpuConvertOp {
    a: Matrix3x3,
    b: Matrix3x3,
    src_tf: TransferFn,
    dst_tf: TransferFn,
}

impl VipsCustomOperation for CpuConvertOp {
    fn generate(&self, out: &mut CustomRegion, input: &CustomRegion) -> Result<(), Error> {
        macro_rules! convert_rows {
            ($p:ty) => {{
                let (_, top, w, _h) = out.rect();
                for y in top..top + out.rect().3 {
                    let src_row = input.pixels::<$p>(y);
                    let dst_row = out.pixels_mut::<$p>(y);
                    for x in 0..w as usize {
                        let [r, g, bch, alpha] =
                            <$p as chromors_core::pixel::Pixel>::unpack(src_row[x]);
                        let lin = [
                            self.src_tf.decode(r),
                            self.src_tf.decode(g),
                            self.src_tf.decode(bch),
                        ];
                        let xyz = self.a.mul_vec(lin);
                        let dst_lin = self.b.mul_vec(xyz);
                        let out_rgba = [
                            self.dst_tf.encode(dst_lin[0]),
                            self.dst_tf.encode(dst_lin[1]),
                            self.dst_tf.encode(dst_lin[2]),
                            alpha,
                        ];
                        dst_row[x] = <$p as chromors_core::pixel::Pixel>::pack_one(
                            out_rgba,
                            chromors_core::pixel::AlphaPolicy::Straight,
                        );
                    }
                }
                Ok(())
            }};
        }
        crate::dispatch_format!(input.storage(), input.bands(), convert_rows)
    }
}

impl Lower<VipsBackend> for chromors_core::operation::color::Convert<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let h = cx.input(self.input.src());
        let src = self.input.spec.layout;
        let dst = self.target;

        let out = if vips_can_convert(src, dst) {
            vips_native_convert(&h, src, dst)
        } else {
            cpu_convert_region(&h, src, dst)
        }
        .expect("Convert (vips lowering)");

        cx.emit(out);
    }
}
