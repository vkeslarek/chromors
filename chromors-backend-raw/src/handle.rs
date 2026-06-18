use std::sync::Arc;

use super::decode;
use super::params::ProcessWarnings;
use crate::libraw_ffi as raw;

// ── Source (file path or in-memory buffer) ─────────────────────────────────────

pub(crate) enum RawSource {
    File(String),
    Buffer(Arc<Vec<u8>>),
}

impl RawSource {
    /// Open a new `libraw_data_t` from this source.
    ///
    /// # Safety
    /// Caller must call `libraw_close` on the returned pointer when done.
    pub(crate) unsafe fn open_ptr(&self) -> Result<*mut raw::libraw_data_t, crate::Error> {
        use std::ffi::CString;
        let ptr = unsafe { raw::libraw_init(0) };
        if ptr.is_null() {
            return Err(crate::Error::Raw("libraw_init failed".into()));
        }
        let rc = match self {
            RawSource::File(path) => {
                let c_path = CString::new(path.as_str())
                    .map_err(|_| crate::Error::Raw("path contains null byte".into()))?;
                unsafe { raw::libraw_open_file(ptr, c_path.as_ptr()) }
            }
            RawSource::Buffer(data) => unsafe {
                raw::libraw_open_buffer(ptr, data.as_ptr().cast(), data.len())
            },
        };
        if rc != 0 {
            unsafe {
                raw::libraw_close(ptr);
            }
            return Err(decode::libraw_error(rc));
        }
        Ok(ptr)
    }
}

// ── GPS ────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GpsInfo {
    /// Latitude DMS (degrees, minutes, seconds).
    pub latitude: [f32; 3],
    /// 'N' or 'S'.
    pub lat_ref: char,
    /// Longitude DMS.
    pub longitude: [f32; 3],
    /// 'E' or 'W'.
    pub lon_ref: char,
    /// Altitude in metres (positive = above sea level).
    pub altitude: f32,
    /// UTC time as (hours, minutes, seconds).
    pub utc_time: [f32; 3],
}

// ── Lens ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct LensInfo {
    pub lens_make: String,
    pub lens: String,
    pub lens_serial: String,
    /// Focal length in 35 mm equivalent.
    pub focal_length_35mm: u16,
    pub min_focal: f32,
    pub max_focal: f32,
    pub max_ap4_min_focal: f32,
    pub max_ap4_max_focal: f32,
    pub max_ap: f32,
    pub min_ap: f32,
    pub cur_focal: f32,
    pub cur_ap: f32,
}

// ── Metadata extracted at open time ───────────────────────────────────────────

/// All metadata extracted from the RAW file at open time.
///
/// Fully owned — no raw pointers. Safe to share via `Arc`.
#[derive(Debug, Clone)]
pub struct RawMeta {
    // Camera identity
    pub make: String,
    pub model: String,
    pub normalized_make: String,
    pub normalized_model: String,
    pub software: String,
    pub raw_count: u32,
    pub dng_version: u32,
    pub is_foveon: bool,

    // Sensor / Bayer
    /// Bayer filter pattern as a 32-bit mask (0 = unknown).
    pub filters: u32,
    /// Human-readable colour-channel description (e.g. "RGBG").
    pub cdesc: String,

    // Sizes
    pub raw_width: u32,
    pub raw_height: u32,
    /// Row stride in bytes (useful for accessing the raw sensor array).
    pub raw_pitch: u32,
    pub pixel_aspect: f64,
    pub flip: i32,

    // Exposure / EXIF
    pub iso: f32,
    pub shutter: f32,
    pub aperture: f32,
    pub focal_len: f32,
    pub timestamp: i64,
    pub shot_order: u32,
    pub description: String,
    pub artist: String,
    /// White balance ratios found in the file metadata.
    pub analog_balance: [f32; 4],

    // Colour science
    /// Camera white balance multipliers (RGBG) stored in the raw file.
    pub cam_mul: [f32; 4],
    /// Pre-normalisation multipliers computed by libraw.
    pub pre_mul: [f32; 4],
    /// Sensor black level.
    pub black: u32,
    /// Sensor saturation / white level.
    pub maximum: u32,
    /// Bit depth of the raw sensor data.
    pub raw_bps: u32,
    /// Camera-to-XYZ matrix (4×3, row-major).
    pub cam_xyz: [[f32; 3]; 4],
    /// RGB-to-camera matrix (3×4, row-major).
    pub rgb_cam: [[f32; 4]; 3],

    // Lens
    pub lens: LensInfo,

    // Location
    pub gps: Option<GpsInfo>,

    // Metadata blobs
    /// EXIF key-value pairs suitable for embedding in output images.
    pub exif: Vec<(String, String)>,
    /// Raw XMP packet bytes (if present in the file).
    pub xmp: Option<Vec<u8>>,
}

// ── Processed (pixel) data ─────────────────────────────────────────────────────

/// Owns the pixel buffer returned by `libraw_dcraw_make_mem_image`.
pub(crate) struct RawPixels {
    pub(crate) ptr: *mut raw::libraw_processed_image_t,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) colors: u16,
    pub(crate) bits: u16,
    pub(crate) warnings: ProcessWarnings,
}

impl Drop for RawPixels {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                raw::libraw_dcraw_clear_mem(self.ptr);
            }
        }
    }
}

unsafe impl Send for RawPixels {}
unsafe impl Sync for RawPixels {}

// ── RawFrame — public result of a materialize() call ──────────────────────────

/// The decoded pixel buffer returned by `Image2D::<RawBackend>::materialize()`.
///
/// Cheaply shared via `Arc`.  Holds:
/// - The raw pixel bytes (auto-freed on last drop via libraw).
/// - Enough context from the decode params to call `to_vips_image()`.
///
/// ## Workflow
/// ```ignore
/// // Half-size preview cycle:
/// raw.handle.params.half_size = true;
/// let preview: Arc<RawFrame> = raw.materialize()?;
/// // inspect preview.width(), preview.pixel_data(), etc.
///
/// // Full-resolution final decode:
/// raw.handle.params.half_size = false;
/// let frame: Arc<RawFrame> = raw.materialize()?;
/// let vips = frame.to_vips_image()?;
/// ```
pub struct RawFrame {
    pub(crate) pixels: Arc<RawPixels>,
    /// Width of the decoded image in pixels.
    pub width: u32,
    /// Height of the decoded image in pixels.
    pub height: u32,
    /// Number of colour channels in the output (usually 3).
    pub colors: u16,
    /// Bits per sample (8 or 16).
    pub bits: u16,
    /// Libraw process warnings.
    pub warnings: super::params::ProcessWarnings,
    /// Output colour space from the decode params.
    pub(crate) output_color: super::params::OutputColorSpace,
    /// True when `gamma_power ≈ 1.0` (linear output, no gamma curve applied).
    pub(crate) gamma_is_linear: bool,
    /// Static metadata (for EXIF embedding, colour science, etc).
    pub(crate) meta: Arc<RawMeta>,
}

impl RawFrame {
    /// Full static metadata from the RAW file.
    pub fn meta(&self) -> &RawMeta {
        &self.meta
    }
}

impl RawFrame {
    /// Raw pixel bytes from the decoded image.
    ///
    /// Layout: packed RGB rows, `bits/8` bytes per sample.
    pub fn pixel_data(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::addr_of!((*self.pixels.ptr).data).cast::<u8>(),
                (*self.pixels.ptr).data_size as usize,
            )
        }
    }

    /// Bytes per pixel (= `colors * bits / 8`).
    pub fn bytes_per_pixel(&self) -> usize {
        self.colors as usize * (self.bits as usize / 8)
    }

    /// Total byte count of the pixel buffer.
    pub fn data_size(&self) -> usize {
        unsafe { (*self.pixels.ptr).data_size as usize }
    }
}

// ── Thumbnail data ─────────────────────────────────────────────────────────────

/// Owns the buffer returned by `libraw_dcraw_make_mem_thumb`.
pub(crate) struct RawThumb {
    pub(crate) ptr: *mut raw::libraw_processed_image_t,
}

impl Drop for RawThumb {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                raw::libraw_dcraw_clear_mem(self.ptr);
            }
        }
    }
}

unsafe impl Send for RawThumb {}
unsafe impl Sync for RawThumb {}

// ── Handle ─────────────────────────────────────────────────────────────────────

pub struct RawHandle {
    /// Source for re-opening on a clone-materialize path.
    pub(crate) source: Arc<RawSource>,
    /// Decode parameters (modified by RAW operations).
    pub(crate) params: super::params::RawDecodeParams,
    /// Static metadata extracted at open time.
    pub(crate) meta: Arc<RawMeta>,
    /// Live libraw instance (null after materialize or on clones).
    ///
    /// Invariant: this pointer must only be accessed via `&mut RawHandle`.
    /// Never exposed through `&RawHandle`.
    pub(crate) ptr: *mut raw::libraw_data_t,
    /// Thumbnail extracted at open time (shared across clones via Arc).
    pub(crate) thumb: Option<Arc<RawThumb>>,
    /// Materialized pixel data (None until `materialize()` is called).
    pub(crate) pixels: Option<Arc<RawFrame>>,
}

// SAFETY:
// - `ptr` is only mutated via `&mut RawHandle` (open, materialize, drop).
// - `Arc<RawMeta>`, `Arc<RawThumb>`, `Arc<RawFrame>` are Send+Sync.
// - Sharing `&RawHandle` across threads only allows reading `meta`, `thumb`,
//   and `pixels` (all Sync); `ptr` is only touched under `&mut`.
unsafe impl Send for RawHandle {}
unsafe impl Sync for RawHandle {}

impl Clone for RawHandle {
    /// Creates a lightweight clone sharing metadata, thumb, and any already-
    /// materialized pixels. The clone starts without a live `ptr`; calling
    /// `materialize()` on it re-opens the source and re-decodes with the
    /// clone's own `params`.
    fn clone(&self) -> Self {
        Self {
            source: Arc::clone(&self.source),
            params: self.params.clone(),
            meta: Arc::clone(&self.meta),
            ptr: std::ptr::null_mut(),
            thumb: self.thumb.clone(),
            pixels: self.pixels.clone(),
        }
    }
}

impl Drop for RawHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                raw::libraw_close(self.ptr);
            }
        }
        // RawPixels and RawThumb manage their own drops.
    }
}

// ── RawHandle API ────────────────────────────────────────────────────────────────

impl RawHandle {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Open a RAW file with custom decode parameters.
    pub fn open_with(
        path: &str,
        params: super::params::RawDecodeParams,
    ) -> Result<Self, crate::Error> {
        let source = Arc::new(RawSource::File(path.to_owned()));
        decode::open_raw_source(source, params)
    }

    /// Decode a RAW file from an in-memory buffer with custom parameters.
    pub fn from_buffer_with(
        data: &[u8],
        params: super::params::RawDecodeParams,
    ) -> Result<Self, crate::Error> {
        let source = Arc::new(RawSource::Buffer(Arc::new(data.to_vec())));
        decode::open_raw_source(source, params)
    }

    pub fn set_output_color(&self, space: super::params::OutputColorSpace, bps: u8) -> Self {
        let mut handle = self.clone();
        handle.pixels = None;
        handle.params.output_color = space;
        handle.params.output_bps = bps;
        handle
    }

    // ── Materialize (lazy decode) ─────────────────────────────────────────────

    pub fn materialize(&mut self) -> Result<Arc<RawFrame>, crate::Error> {
        decode::materialize(self)
    }

    pub fn is_materialized(&self) -> bool {
        self.pixels.is_some()
    }

    pub fn frame(&self) -> Option<Arc<RawFrame>> {
        self.pixels.clone()
    }

    // ── Static metadata ───────────────────────────────────────────────────────

    pub fn meta(&self) -> &RawMeta {
        &self.meta
    }
    pub fn make(&self) -> &str {
        &self.meta.make
    }
    pub fn model(&self) -> &str {
        &self.meta.model
    }
    pub fn iso(&self) -> f32 {
        self.meta.iso
    }
    pub fn shutter(&self) -> f32 {
        self.meta.shutter
    }
    pub fn aperture(&self) -> f32 {
        self.meta.aperture
    }
    pub fn focal_len(&self) -> f32 {
        self.meta.focal_len
    }
    pub fn raw_width(&self) -> u32 {
        self.meta.raw_width
    }
    pub fn raw_height(&self) -> u32 {
        self.meta.raw_height
    }
    pub fn filters(&self) -> u32 {
        self.meta.filters
    }
    pub fn cdesc(&self) -> &str {
        &self.meta.cdesc
    }

    // ── Decode params ─────────────────────────────────────────────────────────

    pub fn params(&self) -> &super::params::RawDecodeParams {
        &self.params
    }
    pub fn params_mut(&mut self) -> &mut super::params::RawDecodeParams {
        &mut self.params
    }

    // ── Materialized pixel data ───────────────────────────────────────────────

    fn require_frame(&self) -> &RawFrame {
        self.pixels
            .as_deref()
            .expect("call materialize() before accessing pixel data")
    }

    pub fn width(&self) -> u32 {
        self.require_frame().width
    }
    pub fn height(&self) -> u32 {
        self.require_frame().height
    }
    pub fn colors(&self) -> u16 {
        self.require_frame().colors
    }
    pub fn bits(&self) -> u16 {
        self.require_frame().bits
    }
    pub fn pixel_data(&self) -> &[u8] {
        self.require_frame().pixel_data()
    }
    pub fn has_thumbnail(&self) -> bool {
        self.thumb.is_some()
    }

    // ── Library info ──────────────────────────────────────────────────────────

    pub fn libraw_version() -> &'static str {
        unsafe {
            let p = crate::libraw_ffi::libraw_version();
            if p.is_null() {
                return "unknown";
            }
            std::ffi::CStr::from_ptr(p).to_str().unwrap_or("unknown")
        }
    }
}
