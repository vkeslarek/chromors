/// Maps a Rust enum to its corresponding libraw integer constant.
///
/// Analogous to `IntoVipsEnum` — a future GPU demosaic backend would define
/// `IntoGpuRawEnum` for its own integer mapping without touching these enums.
pub trait IntoRawEnum {
    fn into_raw(self) -> i32;
}

// ── Demosaic algorithm ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum InterpolationQuality {
    Linear = 0,
    Vng = 1,
    Ppg = 2,
    Ahd = 3,
    Dcb = 4,
    Dht = 11,
    ModifiedAhd = 12,
}

impl IntoRawEnum for InterpolationQuality {
    fn into_raw(self) -> i32 {
        self as i32
    }
}

// ── Highlight recovery ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum HighlightMode {
    Clip = 0,
    Unclip = 1,
    Blend = 2,
    Rebuild3 = 3,
    Rebuild4 = 4,
    Rebuild5 = 5,
    Rebuild6 = 6,
    Rebuild7 = 7,
    Rebuild8 = 8,
    Rebuild9 = 9,
}

impl IntoRawEnum for HighlightMode {
    fn into_raw(self) -> i32 {
        self as i32
    }
}

// ── Output color space ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum OutputColorSpace {
    Raw = 0,
    Srgb = 1,
    Adobe = 2,
    Wide = 3,
    ProPhoto = 4,
    Xyz = 5,
    Aces = 6,
    DciP3 = 7,
    Rec2020 = 8,
}

impl IntoRawEnum for OutputColorSpace {
    fn into_raw(self) -> i32 {
        self as i32
    }
}

// ── White balance ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WhiteBalanceSource {
    /// Use white balance stored in the raw file.
    Camera,
    /// Auto white balance computed from the image.
    Auto,
    /// User-supplied RGBG multipliers (custom/manual white balance).
    Custom([f32; 4]),
    /// No white balance adjustment (multipliers all = 1.0).
    None,
}

// ── Camera color matrix mode ───────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraMatrixMode {
    /// Never use the camera-embedded color matrix.
    None = 0,
    /// Use camera matrix only when using camera white balance.
    WithCameraWb = 1,
    /// Always use the camera matrix (ignores white balance selection).
    Always = 3,
}

impl IntoRawEnum for CameraMatrixMode {
    fn into_raw(self) -> i32 {
        self as i32
    }
}

// ── Output flags ───────────────────────────────────────────────────────────────

/// Bitflags for `libraw_output_params_t::output_flags`.
pub mod output_flags {
    pub const NONE: i32 = 0;
    /// Omit creation timestamps from output.
    pub const NO_TIMESTAMPS: i32 = 1;
    /// Generate edge-to-edge image ignoring masked areas.
    pub const EDGE_TO_EDGE: i32 = 4;
}

// ── Process warnings ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessWarnings(pub u32);

impl ProcessWarnings {
    pub const BAD_CAMERA_WB: u32 = 1 << 0;
    pub const NO_METADATA: u32 = 1 << 1;
    pub const NO_THUMBNAIL: u32 = 1 << 3;
    pub const BAD_CROP: u32 = 1 << 5;
    pub const SUSPICIOUS_PIXELS: u32 = 1 << 8;
    pub const BAD_PIXELS: u32 = 1 << 11;

    pub fn contains(self, flag: u32) -> bool {
        self.0 & flag != 0
    }
    pub fn is_clean(self) -> bool {
        self.0 == 0
    }
}

// ── Thumbnail / image format tags ─────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ThumbnailFormat {
    Unknown = 0,
    Jpeg = 1,
    Bitmap = 2,
    Bitmap16 = 3,
    Layer = 4,
    Rollei = 5,
    H265 = 6,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ImageFormat {
    Jpeg = 1,
    Bitmap = 2,
}

// ── Complete decode parameters ────────────────────────────────────────────────

/// All decode-time parameters for `RawBackend`.
///
/// These map 1-to-1 to `libraw_output_params_t` and `libraw_raw_unpack_params_t`.
/// After calling `Image::open`, modify params through operations or directly, then
/// call `Image::materialize()` to run the demosaic pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct RawDecodeParams {
    // ── Output format ──────────────────────────────────────────────────────
    pub output_color: OutputColorSpace,
    /// Bits per sample in the output (8 or 16).
    pub output_bps: u8,
    /// Output as TIFF instead of PPM (affects `libraw_dcraw_ppm_tiff_writer`).
    pub output_tiff: bool,
    /// Bitfield of `OutputFlags` constants.
    pub output_flags: i32,

    // ── Demosaic ───────────────────────────────────────────────────────────
    pub interpolation: InterpolationQuality,
    /// Decode to half the original resolution (2×2 pixel binning).
    pub half_size: bool,
    /// Treat each Bayer channel independently (suppress green interpolation).
    pub four_color_rgb: bool,
    /// Equalize green channels to reduce color moiré artifacts.
    pub green_matching: bool,
    /// DCB demosaic iterations (-1 = off).
    pub dcb_iterations: i32,
    /// Enable DCB edge-detection enhancement.
    pub dcb_enhance: bool,
    /// FBDD noise reduction level (0 = off, 1 = light, 2 = full).
    pub fbdd_noiserd: u8,
    /// Skip the interpolation step (output raw Bayer channel values).
    pub no_interpolation: bool,

    // ── Highlight recovery ─────────────────────────────────────────────────
    pub highlights: HighlightMode,

    // ── Brightness / exposure ──────────────────────────────────────────────
    /// Disable automatic brightness normalization.
    pub no_auto_bright: bool,
    /// Auto-brightness threshold (fraction of clipped pixels; default 0.01).
    pub auto_bright_thr: f32,
    /// Manual brightness multiplier applied after auto-bright.
    pub bright: f32,
    /// Enable exposure correction.
    pub exp_correc: bool,
    /// Exposure shift in EV (0.25..8.0; default 1.0 = no shift).
    pub exp_shift: f32,
    /// Preserve highlights during exposure correction (0.0..1.0).
    pub exp_preser: f32,

    // ── Noise reduction ────────────────────────────────────────────────────
    /// Wavelet denoising threshold (0 = off).
    pub threshold: f32,
    /// Median filter passes (0 = off).
    pub med_passes: i32,

    // ── White balance ──────────────────────────────────────────────────────
    pub white_balance: WhiteBalanceSource,
    pub camera_matrix: CameraMatrixMode,
    /// User saturation adjustment (0 = default).
    pub user_sat: i32,

    // ── Gamma curve ────────────────────────────────────────────────────────
    /// Gamma curve power (reciprocal; 0.45 ≈ sRGB).
    pub gamma_power: f64,
    /// Gamma curve linear portion slope (4.5 = sRGB).
    pub gamma_slope: f64,
    /// Full gamma curve array [power, slope, c2, c3, c4, c5] (6 elements).
    /// When set, overrides `gamma_power` and `gamma_slope`.
    pub gamma_curve: Option<[f64; 6]>,

    // ── Geometry ───────────────────────────────────────────────────────────
    /// Manual flip (−1 = use EXIF, 0 = none, 3 = 180°, 5 = 90°CW, 6 = 90°CCW).
    pub user_flip: i32,
    /// Apply Fuji-specific diagonal rotation (false = disable).
    pub use_fuji_rotate: bool,
    /// Disable auto-scaling (keep raw sensor pixel count).
    pub no_auto_scale: bool,

    // ── Crop / region ──────────────────────────────────────────────────────
    /// Crop box [left, top, width, height] in sensor coordinates.
    pub cropbox: Option<[u32; 4]>,
    /// Grey-box for manual white balance sampling [left, top, width, height].
    pub greybox: Option<[u32; 4]>,

    // ── Aberration correction ──────────────────────────────────────────────
    /// Chromatic aberration correction [r_red, r_green, r_blue, r_alpha].
    pub aber: Option<[f64; 4]>,

    // ── Manual exposure levels ─────────────────────────────────────────────
    /// Override auto-detected black level (−1 = auto).
    pub user_black: i32,
    /// Per-channel black levels (0 = use user_black for all).
    pub user_cblack: [i32; 4],
    /// Adjust maximum sample value (0.0 = auto).
    pub adjust_maximum_thr: f32,

    // ── Unpack params ──────────────────────────────────────────────────────
    /// Select which shot to decode in multi-shot raw files.
    pub shot_select: u32,
    /// Maximum memory for raw decode in megabytes.
    pub max_raw_memory_mb: u32,
    /// Sony ARW2 posterization threshold (0 = default).
    pub sony_arw2_posterization_thr: i32,
    /// Use the RawSpeed library for decoding (requires RawSpeed build; 0 = auto).
    pub use_rawspeed: i32,
    /// Use the DNG SDK for DNG decoding (0 = off, 1 = on, -1 = auto).
    pub use_dngsdk: i32,

    // ── ICC profiles (file paths; empty = use libraw defaults) ────────────
    /// Path to output ICC profile file.
    pub output_profile: Option<String>,
    /// Path to camera ICC profile file.
    pub camera_profile: Option<String>,

    // ── Bad-pixel correction ───────────────────────────────────────────────
    /// Path to bad-pixels correction map file.
    pub bad_pixels: Option<String>,
    /// Path to dark-frame subtraction file.
    pub dark_frame: Option<String>,
}

impl Default for RawDecodeParams {
    fn default() -> Self {
        Self {
            // ACES AP0 (ACES2065-1) linear, 16-bit.
            // Linear gamma (power=1, slope=1) preserves the full sensor DR without
            // baking in a display-referred tone curve.  The GPU pipeline converts
            // AP0 → ACEScg when rendering; Vips tags the image as scRGB (linear)
            // which avoids the "magenta" artefact from mis-applied gamma.
            output_color: OutputColorSpace::Aces,
            output_bps: 16,
            output_tiff: false,
            output_flags: output_flags::NONE,

            interpolation: InterpolationQuality::Ahd,
            half_size: false,
            four_color_rgb: false,
            green_matching: false,
            dcb_iterations: -1,
            dcb_enhance: false,
            fbdd_noiserd: 0,
            no_interpolation: false,

            highlights: HighlightMode::Blend,

            // Linear output: no auto-bright adjustment (which assumes a display-referred curve).
            no_auto_bright: true,
            auto_bright_thr: 0.01,
            bright: 2.0,
            exp_correc: false,
            exp_shift: 1.0,
            exp_preser: 0.0,

            threshold: 0.0,
            med_passes: 0,

            white_balance: WhiteBalanceSource::Camera,
            camera_matrix: CameraMatrixMode::WithCameraWb,
            user_sat: 0,

            // Linear gamma (power=1.0, slope=1.0).
            gamma_power: 1.0,
            gamma_slope: 1.0,
            gamma_curve: None,

            user_flip: -1,
            use_fuji_rotate: true,
            no_auto_scale: false,

            cropbox: None,
            greybox: None,
            aber: None,

            user_black: -1,
            user_cblack: [0; 4],
            adjust_maximum_thr: 0.75,

            shot_select: 0,
            max_raw_memory_mb: 2048,
            sony_arw2_posterization_thr: 0,
            use_rawspeed: 0,
            use_dngsdk: -1,

            output_profile: None,
            camera_profile: None,
            bad_pixels: None,
            dark_frame: None,
        }
    }
}
