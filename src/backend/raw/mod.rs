mod decode;
pub(crate) mod handle;
mod params;

use std::sync::Arc;

use crate::backend::{Backend, OpenBuffer, OpenFile, Operation};
use crate::data::image::Image2D;
use crate::error::Error;
use crate::operation::raw::RawOperation;
use crate::pixel::PixelFormat;

pub use handle::{GpsInfo, LensInfo, RawFrame, RawHandle, RawMeta};
pub use params::{
    CameraMatrixMode, HighlightMode, ImageFormat, InterpolationQuality, IntoRawEnum,
    OutputColorSpace, ProcessWarnings, RawDecodeParams, ThumbnailFormat, WhiteBalanceSource,
    output_flags,
};
// ── Backend marker ─────────────────────────────────────────────────────────────

pub struct RawBackend;

impl Backend for RawBackend {
    type Handle = RawHandle;
    type Buffer = Vec<u8>;
}

// ── OpenFile ───────────────────────────────────────────────────────────────────

impl OpenFile for RawBackend {
    fn open_file(path: &str) -> Result<RawHandle, Error> {
        let source = Arc::new(handle::RawSource::File(path.to_owned()));
        decode::open_raw_source(source, RawDecodeParams::default())
    }
}

// ── OpenBuffer ─────────────────────────────────────────────────────────────────

impl OpenBuffer for RawBackend {
    fn open_buffer(data: &[u8]) -> Result<RawHandle, Error> {
        let source = Arc::new(handle::RawSource::Buffer(Arc::new(data.to_vec())));
        decode::open_raw_source(source, RawDecodeParams::default())
    }
}

// ── Operations (via RawOperation blanket impl) ─────────────────────────────────

impl<Op: RawOperation> Operation<Image2D<RawBackend>> for Op {
    type Output = Image2D<RawBackend>;

    /// Clone the image, update decode params, invalidate cached pixels.
    ///
    /// Does NOT run demosaic — call `Image2D::materialize()` when pixel data is
    /// needed.
    fn execute(&self, input: &Image2D<RawBackend>) -> Result<Image2D<RawBackend>, Error> {
        let mut new_handle = input.handle.clone();
        new_handle.pixels = None;
        self.apply_to_params(&mut new_handle.params);
        Ok(Image2D::from_handle(new_handle))
    }
}

// ── RawFrame — Vips conversion ─────────────────────────────────────────────────

impl RawFrame {
    /// Convert this decoded frame to a `VipsBackend` image.
    ///
    /// The Vips interpretation (sRGB=22 / scRGB=28) is derived from the gamma
    /// params that were active at materialize time.  For linear output, a
    /// `pixors-cs` metadata int is set so the GPU pipeline uses the correct
    /// primary matrix.
    pub fn to_vips_image(&self) -> Result<Image2D<crate::backend::vips::VipsBackend>, Error> {
        use crate::color::primaries::{RgbPrimaries, WhitePoint};
        use crate::color::space::ColorSpace as CS;

        let data = self.pixel_data();
        let (w, h, colors, bits) = (self.width, self.height, self.colors, self.bits);

        let (fmt, sample_bytes) = match bits {
            8 => (PixelFormat::Rgb8, 1usize),
            16 => (PixelFormat::Rgb16, 2usize),
            _ => (PixelFormat::Rgb8, 1usize),
        };

        // libraw may output 4 channels (R,G1,G2,B) for some sensor layouts.
        // Strip to 3-channel RGB — the 4th channel is NOT alpha.
        let mem_img = if colors > 3 {
            let src_bpp = colors as usize * sample_bytes;
            let dst_bpp = 3 * sample_bytes;
            let rgb: Vec<u8> = data
                .chunks_exact(src_bpp)
                .flat_map(|p| &p[..dst_bpp])
                .copied()
                .collect();
            Image2D::<crate::backend::vips::VipsBackend>::from_memory(
                &rgb, w as i32, h as i32, 3, fmt,
            )?
        } else {
            Image2D::<crate::backend::vips::VipsBackend>::from_memory(
                data,
                w as i32,
                h as i32,
                colors as i32,
                fmt,
            )?
        };

        let vips_interp: i32 = if self.gamma_is_linear { 28 } else { 22 };
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"copy\0")?;
        op.set_image("in", mem_img.vips_ptr());
        op.set_int("interpretation", vips_interp);
        let img = op.run()?;
        let flat = img.copy_to_memory()?;

        if self.gamma_is_linear {
            // Map libraw output_color to our ColorSpace, verified against
            // LibRaw/src/tables/colorconst.cpp matrices and white points.
            //
            // Key findings from libraw source:
            // - `d65_white` is used as the reference — all output spaces use D65
            //   EXCEPT ProPhoto (D50) and the Raw sensor space.
            // - `aces_rgb` comment: "adapted from ACES D60-like WP to D65" —
            //   libraw's ACES output is D65-adapted, NOT standard ACES2065-1 (D60).
            // - `wide_rgb` primaries ≈ AdobeWide (ISO 22028-2 Wide Gamut) at D65.
            let cs = match self.output_color {
                OutputColorSpace::Srgb => CS::LINEAR_SRGB, // Bt709, D65 ✓
                OutputColorSpace::Adobe => CS::linear(RgbPrimaries::Adobe1998, WhitePoint::D65), // Adobe1998, D65 ✓
                OutputColorSpace::Wide => CS::linear(RgbPrimaries::AdobeWide, WhitePoint::D65), // WideGamut, D65 (was D50 — FIXED)
                OutputColorSpace::ProPhoto => CS::linear(RgbPrimaries::ProPhoto, WhitePoint::D50), // ProPhoto, D50 ✓
                OutputColorSpace::Xyz => CS::LINEAR_SRGB, // XYZ: approximation, rarely used
                OutputColorSpace::Aces => CS::linear(RgbPrimaries::Ap0, WhitePoint::D65), // AP0, D65-adapted (not D60 — FIXED)
                OutputColorSpace::DciP3 => CS::LINEAR_DISPLAY_P3, // P3, D65 ✓
                OutputColorSpace::Rec2020 => CS::LINEAR_REC2020,  // Bt2020, D65 ✓
                OutputColorSpace::Raw => CS::LINEAR_SRGB,         // sensor → approx sRGB
            };
            flat.set_pixors_cs(cs);
        }

        for (key, value) in &self.meta.exif {
            let _ = flat.set_metadata(key, value);
        }
        Ok(flat)
    }
}

// ── Image2D<RawBackend> API ──────────────────────────────────────────────────────

impl Image2D<RawBackend> {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Open a RAW file with custom decode parameters.
    ///
    /// Does NOT run demosaic — call `materialize()` to decode pixel data.
    pub fn open_with(path: &str, params: RawDecodeParams) -> Result<Self, Error> {
        let source = Arc::new(handle::RawSource::File(path.to_owned()));
        Ok(Image2D::from_handle(decode::open_raw_source(
            source, params,
        )?))
    }

    /// Decode a RAW file from an in-memory buffer with custom parameters.
    pub fn from_buffer_with(data: &[u8], params: RawDecodeParams) -> Result<Self, Error> {
        let source = Arc::new(handle::RawSource::Buffer(Arc::new(data.to_vec())));
        Ok(Image2D::from_handle(decode::open_raw_source(
            source, params,
        )?))
    }

    // ── Operations ───────────────────────────────────────────────────────────

    /// Apply a decode operation (demosaic, white balance, color space, etc.).
    ///
    /// Lazy: returns a new image with updated params. Call `materialize()` to
    /// re-decode with the new settings.
    pub fn execute<O: Operation<Image2D<RawBackend>>>(&self, op: &O) -> Result<O::Output, Error> {
        op.execute(self)
    }

    /// Set the output colour space and bits-per-sample.
    ///
    /// Returns a new image with updated params.  Invalidates cached pixels.
    pub fn set_output_color(&self, space: OutputColorSpace, bps: u8) -> Self {
        let mut handle = self.handle.clone();
        handle.pixels = None;
        handle.params.output_color = space;
        handle.params.output_bps = bps;
        Image2D::from_handle(handle)
    }

    // ── Materialize (lazy decode) ─────────────────────────────────────────────

    /// Run the demosaic pipeline and return the decoded pixel frame.
    ///
    /// Idempotent: repeated calls with unchanged params return the same Arc.
    /// When params change via an operation, the cached frame is cleared and
    /// re-decoded on the next call.
    ///
    /// ## Preview loop pattern
    /// ```ignore
    /// // Fast preview at half resolution
    /// raw.params_mut().half_size = true;
    /// let preview = raw.materialize()?;
    /// show(preview.pixel_data(), preview.width, preview.height);
    ///
    /// // Full-resolution final decode
    /// raw.params_mut().half_size = false;
    /// let frame = raw.materialize()?;
    /// let vips = frame.to_vips_image()?;
    /// ```
    pub fn materialize(&mut self) -> Result<Arc<RawFrame>, Error> {
        decode::materialize(&mut self.handle)
    }

    /// Returns `true` if pixel data is available without re-materializing.
    pub fn is_materialized(&self) -> bool {
        self.handle.pixels.is_some()
    }

    /// Return the current materialized frame without re-decoding.
    ///
    /// Returns `None` if `materialize()` has not been called yet.
    pub fn frame(&self) -> Option<Arc<RawFrame>> {
        self.handle.pixels.clone()
    }

    // ── Static metadata (available before materializing) ──────────────────────

    pub fn meta(&self) -> &RawMeta {
        &self.handle.meta
    }
    pub fn make(&self) -> &str {
        &self.handle.meta.make
    }
    pub fn model(&self) -> &str {
        &self.handle.meta.model
    }
    pub fn normalized_make(&self) -> &str {
        &self.handle.meta.normalized_make
    }
    pub fn normalized_model(&self) -> &str {
        &self.handle.meta.normalized_model
    }
    pub fn software(&self) -> &str {
        &self.handle.meta.software
    }
    pub fn description(&self) -> &str {
        &self.handle.meta.description
    }
    pub fn artist(&self) -> &str {
        &self.handle.meta.artist
    }
    pub fn iso(&self) -> f32 {
        self.handle.meta.iso
    }
    pub fn shutter(&self) -> f32 {
        self.handle.meta.shutter
    }
    pub fn aperture(&self) -> f32 {
        self.handle.meta.aperture
    }
    pub fn focal_len(&self) -> f32 {
        self.handle.meta.focal_len
    }
    pub fn timestamp(&self) -> i64 {
        self.handle.meta.timestamp
    }
    pub fn shot_order(&self) -> u32 {
        self.handle.meta.shot_order
    }
    pub fn raw_count(&self) -> u32 {
        self.handle.meta.raw_count
    }
    pub fn is_foveon(&self) -> bool {
        self.handle.meta.is_foveon
    }
    pub fn dng_version(&self) -> u32 {
        self.handle.meta.dng_version
    }
    pub fn raw_width(&self) -> u32 {
        self.handle.meta.raw_width
    }
    pub fn raw_height(&self) -> u32 {
        self.handle.meta.raw_height
    }
    pub fn raw_pitch(&self) -> u32 {
        self.handle.meta.raw_pitch
    }
    pub fn pixel_aspect(&self) -> f64 {
        self.handle.meta.pixel_aspect
    }
    pub fn flip(&self) -> i32 {
        self.handle.meta.flip
    }
    /// Bayer filter pattern bitmask (0 = unknown / X-Trans).
    pub fn filters(&self) -> u32 {
        self.handle.meta.filters
    }
    pub fn cdesc(&self) -> &str {
        &self.handle.meta.cdesc
    }

    // ── Colour science metadata ───────────────────────────────────────────────

    pub fn cam_mul(&self) -> [f32; 4] {
        self.handle.meta.cam_mul
    }
    pub fn pre_mul(&self) -> [f32; 4] {
        self.handle.meta.pre_mul
    }
    pub fn black_level(&self) -> u32 {
        self.handle.meta.black
    }
    pub fn white_level(&self) -> u32 {
        self.handle.meta.maximum
    }
    pub fn raw_bps(&self) -> u32 {
        self.handle.meta.raw_bps
    }
    /// Camera-to-XYZ D50 matrix (4 rows × 3 cols).
    pub fn cam_xyz(&self) -> [[f32; 3]; 4] {
        self.handle.meta.cam_xyz
    }
    /// sRGB-to-camera matrix (3 rows × 4 cols).
    pub fn rgb_cam(&self) -> [[f32; 4]; 3] {
        self.handle.meta.rgb_cam
    }

    // ── Lens + GPS ────────────────────────────────────────────────────────────

    pub fn lens_info(&self) -> &LensInfo {
        &self.handle.meta.lens
    }
    pub fn gps(&self) -> Option<&GpsInfo> {
        self.handle.meta.gps.as_ref()
    }

    // ── Metadata arrays ───────────────────────────────────────────────────────

    pub fn exif_entries(&self) -> &[(String, String)] {
        &self.handle.meta.exif
    }
    pub fn xmp(&self) -> Option<&[u8]> {
        self.handle.meta.xmp.as_deref()
    }

    // ── Decode params ─────────────────────────────────────────────────────────

    pub fn params(&self) -> &RawDecodeParams {
        &self.handle.params
    }
    pub fn params_mut(&mut self) -> &mut RawDecodeParams {
        &mut self.handle.params
    }

    // ── Materialized pixel data ───────────────────────────────────────────────

    fn require_frame(&self) -> &RawFrame {
        self.handle
            .pixels
            .as_deref()
            .expect("call materialize() before accessing pixel data")
    }

    /// Width of the decoded image in pixels.  Panics if not materialized.
    pub fn width(&self) -> u32 {
        self.require_frame().width
    }
    /// Height of the decoded image in pixels.  Panics if not materialized.
    pub fn height(&self) -> u32 {
        self.require_frame().height
    }
    /// Number of colour channels in the decoded output (usually 3 = RGB).
    pub fn colors(&self) -> u16 {
        self.require_frame().colors
    }
    /// Bits per sample in the decoded output (8 or 16).
    pub fn bits(&self) -> u16 {
        self.require_frame().bits
    }
    /// Process warnings from the last `materialize()` call.
    pub fn warnings(&self) -> ProcessWarnings {
        self.require_frame().warnings
    }

    /// Raw pixel bytes from the decoded image.
    ///
    /// Layout: packed RGB (or RGBG) rows, `bits/8` bytes per sample.
    ///
    /// # Panics
    /// Panics if called before `materialize()`.
    pub fn pixel_data(&self) -> &[u8] {
        self.require_frame().pixel_data()
    }

    // ── Conversions ───────────────────────────────────────────────────────────

    /// Convert the decoded image to a `VipsBackend` image.
    ///
    /// Calls `materialize()` if not done yet, then delegates to
    /// `RawFrame::to_vips_image()`.
    pub fn to_vips_image(&mut self) -> Result<Image2D<crate::backend::vips::VipsBackend>, Error> {
        self.materialize()?.to_vips_image()
    }

    /// Extract the embedded JPEG thumbnail as a `VipsBackend` image.
    ///
    /// Returns `None` if no JPEG thumbnail is present in the file.
    pub fn to_thumbnail_image(
        &self,
    ) -> Result<Option<Image2D<crate::backend::vips::VipsBackend>>, Error> {
        let thumb = match &self.handle.thumb {
            Some(t) => t,
            None => return Ok(None),
        };
        let data = unsafe {
            std::slice::from_raw_parts(
                std::ptr::addr_of!((*thumb.ptr).data).cast::<u8>(),
                (*thumb.ptr).data_size as usize,
            )
        };
        Ok(Some(
            Image2D::<crate::backend::vips::VipsBackend>::from_buffer(data)?,
        ))
    }

    /// Returns `true` if an embedded thumbnail is available.
    pub fn has_thumbnail(&self) -> bool {
        self.handle.thumb.is_some()
    }

    // ── Library info ──────────────────────────────────────────────────────────

    /// libraw version string (e.g. `"0.21.2"`).
    pub fn libraw_version() -> &'static str {
        unsafe {
            let p = crate::libraw_ffi::libraw_version();
            if p.is_null() {
                return "unknown";
            }
            std::ffi::CStr::from_ptr(p).to_str().unwrap_or("unknown")
        }
    }

    /// Number of camera models supported by this libraw build.
    pub fn supported_camera_count() -> i32 {
        unsafe { crate::libraw_ffi::libraw_cameraCount() }
    }
}
