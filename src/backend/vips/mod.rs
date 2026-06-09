pub mod custom;
pub mod data;
pub mod gobject;
pub mod interpolate;
pub mod operation;
pub mod region;
pub mod sbuf;
pub mod source;
pub mod target;
pub mod working;

pub use custom::{CustomRegion, VipsCustomOperation, VipsCustomSink};
pub use interpolate::{Interpolate, InterpolationMethod};
pub use region::Region;
pub use sbuf::Sbuf;
pub use source::Source;
pub use target::Target;
pub use working::{RegionProcessor, RegionView, RegionViewMut, execute_processor};

use std::ffi::CStr;
use std::ffi::CString;
use std::ptr;

use crate::error::Error;
use crate::libvips_ffi as ffi;

pub(crate) fn null() -> *const std::ffi::c_void {
    ptr::null()
}

pub(crate) fn vips_error() -> String {
    unsafe {
        let buf = crate::libvips_ffi::vips_error_buffer();
        let s = if buf.is_null() {
            String::from("unknown error")
        } else {
            CStr::from_ptr(buf).to_string_lossy().into_owned()
        };
        crate::libvips_ffi::vips_error_clear();
        s
    }
}

pub trait IntoVipsEnum {
    fn into_vips(self) -> i32;
}

pub trait IntoVipsName {
    fn into_vips_name(self) -> &'static str;
}

pub trait IntoVipsOption {
    fn to_vips_options(&self) -> String;
}

pub trait IntoVipsBandFormat {
    fn into_vips_band_format(self) -> i32;
}

pub trait FromVipsBandFormat: Sized {
    fn from_vips_band_format(raw: i32, bands: i32) -> Self;
}

pub trait IntoVipsInterpretation {
    fn into_vips_interpretation(self) -> i32;
}

pub trait FromVipsInterpretation: Sized {
    fn from_vips_interpretation(raw: i32) -> Self;
}

use super::{
    Backend, OpenBuffer, OpenFile, Operation as SuperOperation, SourceInput, TargetOutput,
};
use crate::data::image::Image;

/// Plain marker struct for the libvips backend.
pub struct VipsBackend;

/// Opaque handle wrapping a `VipsImage` GObject pointer.
///
/// Owns a strong GObject reference; cloning increments the refcount,
/// dropping decrements it.
pub struct VipsHandle {
    pub(crate) ptr: *mut ffi::VipsImage,
}

unsafe impl Send for VipsHandle {}
unsafe impl Sync for VipsHandle {}

impl Clone for VipsHandle {
    fn clone(&self) -> Self {
        unsafe { ffi::g_object_ref(self.ptr as ffi::gpointer) };
        VipsHandle { ptr: self.ptr }
    }
}

impl Drop for VipsHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::g_object_unref(self.ptr as ffi::gpointer) };
        }
    }
}

impl Backend for VipsBackend {
    type Handle = VipsHandle;
    type Buffer = Vec<u8>;
}

impl OpenFile for VipsBackend {
    fn open_file(path: &str) -> Result<VipsHandle, Error> {
        let c = CString::new(path).map_err(|_| Error::Vips("invalid path".into()))?;
        let ptr = unsafe { ffi::vips_image_new_from_file_RW(c.as_ptr()) };
        if ptr.is_null() {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(VipsHandle { ptr })
    }
}

impl SourceInput for VipsBackend {
    type Source = Source;

    fn open_source(source: &Self::Source) -> Result<VipsHandle, Error> {
        unsafe {
            let empty = c"";
            let ptr = ffi::vips_image_new_from_source(source.ptr, empty.as_ptr(), null());
            if ptr.is_null() {
                return Err(crate::error::Error::Vips(crate::backend::vips::vips_error()));
            }
            Ok(VipsHandle { ptr })
        }
    }
}

impl TargetOutput<Image<VipsBackend>> for VipsBackend {
    type Target = Target;

    fn write_to_target(image: &Image<VipsBackend>, target: &Target) -> Result<(), Error> {
        unsafe {
            let empty = c"";
            if ffi::vips_image_write_to_target(image.vips_ptr(), empty.as_ptr(), target.ptr, null())
                != 0
            {
                return Err(crate::error::Error::Vips(crate::backend::vips::vips_error()));
            }
        }
        Ok(())
    }
}

impl OpenBuffer for VipsBackend {
    fn open_buffer(data: &[u8]) -> Result<VipsHandle, Error> {
        unsafe {
            let source = ffi::vips_source_new_from_memory(data.as_ptr() as *const _, data.len());
            if source.is_null() {
                return Err(Error::Vips(crate::backend::vips::vips_error()));
            }
            let empty = c"";
            let loaded = ffi::vips_image_new_from_source(source, empty.as_ptr(), null());
            ffi::g_object_unref(source as ffi::gpointer);
            if loaded.is_null() {
                return Err(Error::Vips(crate::backend::vips::vips_error()));
            }
            // Materialise an owned copy so the result can outlive the buffer.
            let ptr = ffi::vips_image_copy_memory(loaded);
            ffi::g_object_unref(loaded as ffi::gpointer);
            if ptr.is_null() {
                return Err(Error::Vips(crate::backend::vips::vips_error()));
            }
            Ok(VipsHandle { ptr })
        }
    }
}

// ── Image<VipsBackend>::execute ──────────────────────────────────────────────

impl Image<VipsBackend> {
    /// Execute an operation against this image.
    pub fn execute<O: crate::backend::Operation<Image<VipsBackend>>>(
        &self,
        op: &O,
    ) -> Result<O::Output, Error> {
        op.execute(self)
    }
}

use crate::backend::{HistogramTargetCapability, ImageTargetCapability};
use crate::geometry::Rect;
use crate::target::{MaterializedHistogram, MaterializedImage};

impl ImageTargetCapability for VipsBackend {
    fn pull_image(
        handle: &Self::Handle,
        rect: Rect,
        _lod: u32,
    ) -> Result<MaterializedImage<Self>, Error> {
        // Need a temporary Image around the handle to use Region::new
        let img = Image::<VipsBackend>::from_handle(handle.clone());
        let img_w = img.width();
        let img_h = img.height();
        let clamped = Rect::new(
            rect.x,
            rect.y,
            rect.width.min(img_w.saturating_sub(rect.x).max(0)),
            rect.height.min(img_h.saturating_sub(rect.y).max(0)),
        );
        if clamped.width <= 0 || clamped.height <= 0 {
            return Err(Error::Render(format!(
                "pull_image: rect out of bounds: rect={rect:?} img={img_w}x{img_h}"
            )));
        }
        let region = Region::new(&img)?;
        region.prepare(clamped.x, clamped.y, clamped.width, clamped.height)?;
        let buffer = region.fetch(clamped.x, clamped.y, clamped.width, clamped.height)?;
        let meta = img.pixel_meta();
        let buffer_rect = Rect::new(0, 0, clamped.width, clamped.height);
        Ok(MaterializedImage {
            buffer,
            meta,
            rect: clamped,
            buffer_rect,
        })
    }

    fn pull_image_batch(
        handle: &Self::Handle,
        rects: &[crate::geometry::Rect],
        lod: u32,
    ) -> Result<Vec<crate::target::MaterializedImage<Self>>, crate::error::Error> {
        rects
            .iter()
            .map(|&rect| Self::pull_image(handle, rect, lod))
            .collect()
    }
}

impl HistogramTargetCapability for VipsBackend {
    type HistogramHandle = VipsHandle;

    fn create_histogram(handle: &Self::Handle) -> Result<Self::HistogramHandle, Error> {
        let img = Image::<VipsBackend>::from_handle(handle.clone());
        let op = crate::operation::HistogramFindOperation { band: None };
        let hist_img = SuperOperation::<Image<VipsBackend>>::execute(&op, &img)?;
        Ok(hist_img.handle)
    }

    fn pull_histogram(
        handle: &Self::HistogramHandle,
    ) -> Result<MaterializedHistogram<Self>, Error> {
        let img = Image::<VipsBackend>::from_handle(handle.clone());
        let bins = img.width() as u32;
        let rect = crate::geometry::Rect::new(0, 0, img.width(), img.height());
        let materialized = Self::pull_image(&img.handle, rect, 0)?;
        Ok(MaterializedHistogram {
            _marker: std::marker::PhantomData,
            buffer: materialized.buffer,
            bins,
        })
    }
}

// ── ColorConversionCapability ─────────────────────────────────────────────────

use crate::backend::ColorConversionCapability;
use crate::backend::vips::gobject::VipsGObject;
use crate::color::space::ColorSpace;
use crate::pixel::{AlphaPolicy, PixelMeta};

impl ColorConversionCapability for VipsBackend {
    fn pixel_meta(handle: &VipsHandle) -> PixelMeta {
        let img = Image::<VipsBackend>::from_handle(handle.clone());
        let format = img.pixel_format();
        let alpha = if img.has_alpha() {
            AlphaPolicy::Straight
        } else {
            AlphaPolicy::OpaqueDrop
        };
        // Custom metadata takes priority over Vips interpretation (needed for
        // ACES/ProPhoto/etc. that don't map to a Vips-native interpretation).
        let space = img
            .get_pixors_cs()
            .unwrap_or_else(|| ColorSpace::from_vips_interpretation(img.raw_interpretation()));
        PixelMeta::new(format, space, alpha)
    }

    fn convert(handle: &VipsHandle, target: PixelMeta) -> Result<VipsHandle, Error> {
        let img = Image::<VipsBackend>::from_handle(handle.clone());
        let current = VipsBackend::pixel_meta(handle);
        vips_convert_impl(&img, current, target).map(|r| r.handle)
    }
}

/// Core Vips color + format conversion.
///
/// Uses `vips_colourspace` for Vips-native spaces (sRGB, scRGB, Rec.2020, Lab, etc.)
/// and `vips_recomb` with our Bradford-adapted RGB matrix for non-native spaces
/// (ACES AP0/AP1, ProPhoto, Wide, DCI-P3, etc.).
fn vips_convert_impl(
    img: &Image<VipsBackend>,
    current: PixelMeta,
    target: PixelMeta,
) -> Result<Image<VipsBackend>, Error> {
    use crate::operation::misc::CastOperation;

    let mut out = img.clone();

    // ── Alpha premultiplication ───────────────────────────────────────────────
    let from_pre = matches!(current.alpha_policy, AlphaPolicy::PremultiplyOnPack);
    let to_straight = matches!(target.alpha_policy, AlphaPolicy::Straight);
    let to_pre = matches!(target.alpha_policy, AlphaPolicy::PremultiplyOnPack);
    let to_opaque = matches!(target.alpha_policy, AlphaPolicy::OpaqueDrop);

    if from_pre && (to_straight || to_opaque) && out.has_alpha() {
        let mut op = VipsGObject::new(b"unpremultiply\0")?;
        op.set_image("in", out.vips_ptr());
        out = op.run()?;
    }

    // ── Color space conversion ────────────────────────────────────────────────
    if current.color_space != target.color_space {
        let src = current.color_space;
        let dst = target.color_space;

        if vips_knows_both(src, dst) {
            // Fast path: vips_colourspace handles this natively.
            use crate::backend::vips::IntoVipsInterpretation;
            let mut op = VipsGObject::new(b"colourspace\0")?;
            op.set_image("in", out.vips_ptr());
            op.set_int("space", dst.into_vips_interpretation());
            out = op.run()?;
        } else {
            // Matrix path: linearise → recomb → re-gamma.
            out = matrix_convert(out, src, dst)?;
        }
    }

    // ── Alpha output policy ───────────────────────────────────────────────────
    let target_bands = target.format.channels() as i32;
    let has_alpha = out.has_alpha();
    if target_bands > out.bands() && !has_alpha && target_bands == out.bands() + 1 {
        let mut op = VipsGObject::new(b"addalpha\0")?;
        op.set_image("in", out.vips_ptr());
        out = op.run()?;
    } else if to_pre && !from_pre && out.has_alpha() {
        let mut op = VipsGObject::new(b"premultiply\0")?;
        op.set_image("in", out.vips_ptr());
        out = op.run()?;
    } else if to_opaque && out.has_alpha() {
        let mut op = VipsGObject::new(b"flatten\0")?;
        op.set_image("in", out.vips_ptr());
        out = op.run()?;
    }

    // ── Format cast ───────────────────────────────────────────────────────────
    if target.format != current.format {
        out = out.execute(&CastOperation {
            format: target.format,
            shift: None,
        })?;
    }

    // Propagate the target color space in pixors metadata.
    let target_id = target.color_space.to_pixors_id();
    if target_id != 0 {
        out.set_pixors_cs(target.color_space);
    }

    Ok(out)
}

/// Returns `true` when both color spaces map to a Vips-native interpretation
/// that `vips_colourspace` can convert between without losing primaries.
fn vips_knows_both(src: ColorSpace, dst: ColorSpace) -> bool {
    // Vips natively knows: sRGB (22), scRGB (28), and through Lab/XYZ it can
    // handle linearisation.  We are conservative: only allow the pair when
    // BOTH are linear or BOTH are sRGB-gamma (Vips's colourspace handles the
    // linearisation step internally).
    let src_is_vips = src == ColorSpace::SRGB || src == ColorSpace::LINEAR_SRGB;
    let dst_is_vips = dst == ColorSpace::SRGB || dst == ColorSpace::LINEAR_SRGB;
    src_is_vips && dst_is_vips
}

/// Apply a full color space conversion using a Bradford-adapted RGB→RGB matrix.
fn matrix_convert(
    img: Image<VipsBackend>,
    src: ColorSpace,
    dst: ColorSpace,
) -> Result<Image<VipsBackend>, Error> {
    use crate::backend::vips::IntoVipsInterpretation;
    use crate::color::matrix::rgb_to_rgb_transform;
    use crate::operation::misc::CastOperation;
    use crate::pixel::PixelFormat;

    let mut out = img;

    // Linearise if source has a gamma curve.
    if !src.is_linear() {
        if src.primaries() == crate::color::primaries::RgbPrimaries::Bt709 {
            // sRGB / Rec.709 — Vips knows how to linearise.
            let mut op = VipsGObject::new(b"colourspace\0")?;
            op.set_image("in", out.vips_ptr());
            op.set_int("space", ColorSpace::LINEAR_SRGB.into_vips_interpretation());
            out = op.run()?;
        } else {
            // Generic power-law: cast to float then apply gamma.
            let gamma = src.transfer().approximate_gamma().unwrap_or(2.2);
            out = out.execute(&CastOperation {
                format: PixelFormat::RgbF32,
                shift: None,
            })?;
            let mut op = VipsGObject::new(b"gamma\0")?;
            op.set_image("in", out.vips_ptr());
            op.set_double("exponent", 1.0 / gamma);
            out = op.run()?;
        }
    }

    // Cast to F32 for the matrix multiply (vips_recomb needs float input).
    out = out.execute(&CastOperation {
        format: PixelFormat::RgbF32,
        shift: None,
    })?;

    // Build and apply the 3×3 linear-to-linear color matrix.
    let matrix = rgb_to_rgb_transform(
        src.primaries(),
        src.white_point(),
        dst.primaries(),
        dst.white_point(),
    )
    .map_err(|e| Error::Vips(format!("color matrix: {e:?}")))?;
    out = apply_recomb(out, &matrix)?;

    // Re-apply gamma if the target is non-linear.
    if !dst.is_linear() {
        if dst.primaries() == crate::color::primaries::RgbPrimaries::Bt709 {
            let mut op = VipsGObject::new(b"colourspace\0")?;
            op.set_image("in", out.vips_ptr());
            op.set_int("space", ColorSpace::SRGB.into_vips_interpretation());
            out = op.run()?;
        } else {
            let gamma = dst.transfer().approximate_gamma().unwrap_or(2.2);
            let mut op = VipsGObject::new(b"gamma\0")?;
            op.set_image("in", out.vips_ptr());
            op.set_double("exponent", gamma);
            out = op.run()?;
        }
    }

    Ok(out)
}

/// Apply a 3×3 linear colour matrix via `vips_recomb`.
///
/// If the image has an alpha band (4 bands), the alpha passes through
/// unchanged — the matrix is extended to 4×4 with an identity row/column
/// for the alpha channel.
fn apply_recomb(
    img: Image<VipsBackend>,
    matrix: &crate::color::matrix::Matrix3x3,
) -> Result<Image<VipsBackend>, Error> {
    use crate::libvips_ffi as ffi;

    let bands = img.bands();
    let m = &matrix.0;

    // Build the recomb matrix.  vips_recomb expects row-major f64 values:
    //   row 0 = output_R coefficients for each input band
    //   row 1 = output_G coefficients
    //   row 2 = output_B coefficients
    //   (row 3 = alpha pass-through if bands == 4)
    //
    // Matrix3x3 layout: column-major [[f32; 3]; 3]
    //   m[col][row] → flat[row * bands_in + col]
    let (nrows, ncols) = (bands as usize, bands as usize);
    let mut flat = vec![0f64; nrows * ncols];

    for row in 0..3usize {
        for col in 0..3usize {
            flat[row * ncols + col] = m[col][row] as f64;
        }
    }
    if bands == 4 {
        // Alpha row and column: pass through unchanged.
        flat[3 * ncols + 3] = 1.0;
    }

    let mat_img = unsafe {
        ffi::vips_image_new_matrix_from_array(
            ncols as i32,
            nrows as i32,
            flat.as_ptr(),
            flat.len() as i32,
        )
    };
    if mat_img.is_null() {
        return Err(Error::Vips(
            "vips_image_new_matrix_from_array failed".into(),
        ));
    }

    let mut out_ptr: *mut ffi::VipsImage = std::ptr::null_mut();
    let rc = unsafe {
        ffi::vips_recomb(
            img.vips_ptr(),
            &mut out_ptr,
            mat_img,
            std::ptr::null::<std::ffi::c_void>(),
        )
    };
    unsafe {
        ffi::g_object_unref(mat_img as *mut std::ffi::c_void);
    }

    if rc != 0 {
        return Err(Error::Vips(vips_error()));
    }
    if out_ptr.is_null() {
        return Err(Error::NullPtr);
    }
    Ok(Image::<VipsBackend>::from_vips_ptr(out_ptr))
}
