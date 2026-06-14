//! `Convert` — the universal color/format conversion operation
//! (`docs/native-color-management.md` §6).
//!
//! GPU lowering needs no bespoke kernel for the common case: the generic
//! `copy_kernel` (`lib/io.slang`) plus a `color_read_wrap`
//! (`lib/color/interp.slang`, §5.12) does the whole XYZ(D50)-hub conversion
//! on read, writing through the target layout's own codec sandwich. The one
//! exception is a 5-channel (CmykA) source: its K-alpha 5th sample can't ride
//! through the generic `float4` read wrap, so `convert_cmyka_kernel`
//! (`lib/color/convert.slang`, §6.1.3) binds the source's storage codec
//! directly.

use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::gpu::color_params::{ConvertParams, color_read_wrap};
use crate::backend::gpu::view::ParamBlock;
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::vips::custom::{CustomRegion, VipsCustomOperation, run_custom};
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::{IntoVipsBandFormat, VipsBackend, VipsBuilder, VipsHandle};
use crate::color::intent::RenderingIntent;
use crate::color::matrix::Matrix3x3;
use crate::color::model::ColorModel;
use crate::color::pipeline::convert_matrices;
use crate::color::space::{ColorSpace, to_vips_interpretation};
use crate::color::transfer::TransferFn;
use crate::data::image::{Image2D, ImageKind};
use crate::error::Error;
use crate::ffi;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::pixel::{AlphaPolicy, AlphaState, PixelLayout, Pixel, Storage};
use crate::work_unit::{Region, WorkUnit};

/// Converts an image to `target` — storage, color model, alpha state and/or
/// color space, in one pass. Pointwise: the output region equals the input
/// region (`demand`).
pub struct Convert<B: Backend> {
    pub input: Input<ImageKind, B>,
    /// The full destination pixel layout.
    pub target: PixelLayout,
    /// Reserved for future gamut mapping (§10); `lower` doesn't branch on it yet.
    pub intent: RenderingIntent,
}

impl<B: Backend> Operation<B> for Convert<B>
where
    Convert<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }

    fn output_spec(&self) -> ImageKind {
        self.input.spec.with_layout(self.target)
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write(format!("{:?}", self.target).as_bytes());
        state.write_u8(match self.intent {
            RenderingIntent::Relative => 0,
        });
    }
}

impl Lower<GpuBackend> for Convert<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let src = self.input.spec.layout;
        let params = ConvertParams::build(src, self.target).unwrap_or_else(|e| {
            cx.fail(e);
            ConvertParams::identity()
        });

        if src.channel_count() == 5 {
            // CmykA source: the K-alpha 5th channel can't ride through the
            // generic float4 read wrap — bind the storage codec directly.
            cx.param_block(ParamBlock::from_pod("cc", &params));
            cx.kernel("lib.color.convert", "convert_cmyka_kernel");
        } else {
            cx.kernel("lib.io", "copy_kernel");
            cx.read_wrap(color_read_wrap(params));
        }
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl<B: Backend> Image2D<B>
where
    Convert<B>: Lower<B>,
{
    /// Converts this image to `target` — storage/model/alpha/color-space, in
    /// one pass.
    pub fn convert(&self, target: PixelLayout) -> Self {
        self.push(Convert {
            input: self.as_input(),
            target,
            intent: RenderingIntent::Relative,
        })
    }

    /// Changes the color space (primaries/white point/transfer), keeping
    /// storage/model/alpha identical (`docs/native-color-management.md`
    /// §6.2).
    pub fn to_color_space(&self, cs: ColorSpace) -> Self {
        let mut target = self.spec.layout;
        target.color_space = cs;
        self.convert(target)
    }

    /// Changes the sample storage type, keeping model/alpha/color-space
    /// identical (`docs/native-color-management.md` §6.2).
    pub fn to_storage(&self, storage: Storage) -> Self {
        let mut target = self.spec.layout;
        target.storage = storage;
        self.convert(target)
    }

    /// Changes the color model (e.g. `Rgb` -> `Lab`), keeping
    /// storage/alpha/color-space identical (`docs/native-color-management.md`
    /// §6.2).
    pub fn to_model(&self, model: ColorModel) -> Self {
        let mut target = self.spec.layout;
        target.model = model;
        self.convert(target)
    }

    /// Switches to the linear variant of the current color space — a pure
    /// transfer-function change (`docs/native-color-management.md` §6.2).
    pub fn linearize(&self) -> Self {
        self.to_color_space(self.spec.layout.color_space.as_linear())
    }
}

// ── vips / CPU lowering (§6.1.4) ────────────────────────────────────────────

/// `true` if both endpoints have a faithful vips `VipsInterpretation` — the
/// native `colourspace`-based chain ([`vips_native_convert`]) round-trips
/// exactly through it. Otherwise the conversion needs the CPU XYZ-hub
/// custom-region fallback ([`cpu_convert_region`]).
fn vips_can_convert(src: PixelLayout, dst: PixelLayout) -> bool {
    to_vips_interpretation(src.model, src.color_space).is_some()
        && to_vips_interpretation(dst.model, dst.color_space).is_some()
}

/// Native fast path: `unpremultiply -> colourspace(interp) -> premultiply/
/// flatten -> addalpha -> cast`, driven by [`PixelLayout`] via
/// [`to_vips_interpretation`] (port of `src/color/convert.rs::ColorConversion`,
/// `docs/native-color-management.md` §6.1.4).
fn vips_native_convert(h: &VipsHandle, src: PixelLayout, dst: PixelLayout) -> Result<VipsHandle, Error> {
    let mut img = h.clone();

    let from_premultiplied = matches!(src.alpha, AlphaState::Premultiplied);
    let to_straight = matches!(dst.alpha, AlphaState::Straight);
    let to_premultiplied = matches!(dst.alpha, AlphaState::Premultiplied);
    let to_opaque = matches!(dst.alpha, AlphaState::None);

    let has_alpha = unsafe { ffi::vips_image_hasalpha(img.ptr) } != 0;

    if from_premultiplied && (to_straight || to_opaque) && has_alpha {
        let mut op = VipsGObject::new(b"unpremultiply\0")?;
        op.set_image("in", img.ptr);
        img = op.run()?;
    }

    let src_interp = to_vips_interpretation(src.model, src.color_space);
    let dst_interp = to_vips_interpretation(dst.model, dst.color_space);
    if src_interp != dst_interp {
        let interp = dst_interp.expect("vips_can_convert guarantees dst has a vips interpretation");
        let mut op = VipsGObject::new(b"colourspace\0")?;
        op.set_image("in", img.ptr);
        op.set_int("space", interp);
        img = op.run()?;
    }

    let has_alpha_now = unsafe { ffi::vips_image_hasalpha(img.ptr) } != 0;
    if to_premultiplied && !from_premultiplied && has_alpha_now {
        let mut op = VipsGObject::new(b"premultiply\0")?;
        op.set_image("in", img.ptr);
        img = op.run()?;
    } else if to_opaque && has_alpha_now {
        let mut op = VipsGObject::new(b"flatten\0")?;
        op.set_image("in", img.ptr);
        img = op.run()?;
    }

    let bands_now = unsafe { ffi::vips_image_get_bands(img.ptr) };
    let target_bands = dst.channel_count() as i32;
    if target_bands > bands_now && !has_alpha_now && target_bands == bands_now + 1 {
        let mut op = VipsGObject::new(b"addalpha\0")?;
        op.set_image("in", img.ptr);
        img = op.run()?;
    }

    let cur_format = unsafe { ffi::vips_image_get_format(img.ptr) };
    let target_format = dst.storage.into_vips_band_format();
    if cur_format != target_format {
        let mut op = VipsGObject::new(b"cast\0")?;
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
///
/// Scope: requires `src`/`dst` to both be `Rgb` with matching `storage` and
/// `alpha` (only the color space changes). `Gray`/`Lab`/`Xyz`/`Cmyk`/`ScRgb`
/// and sRGB/scRGB-linear `Rgb` always have a faithful vips interpretation
/// (`to_vips_interpretation`) and take the native path instead; other model
/// changes without a vips interpretation (e.g. `Hsv`/`Oklab`/`Yxy`/`YCbCr`)
/// are not yet supported here.
fn cpu_convert_region(h: &VipsHandle, src: PixelLayout, dst: PixelLayout) -> Result<VipsHandle, Error> {
    use crate::color::model::ColorModel;
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

    let (a, b) = convert_matrices(src, dst)?;
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
                        let [r, g, bch, alpha] = <$p as Pixel>::unpack(src_row[x]);
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
                        dst_row[x] = <$p as Pixel>::pack_one(out_rgba, AlphaPolicy::Straight);
                    }
                }
                Ok(())
            }};
        }
        crate::dispatch_format!(input.storage(), input.bands(), convert_rows)
    }
}

impl Lower<VipsBackend> for Convert<VipsBackend> {
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
