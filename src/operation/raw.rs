//! RAW decode operations.
//!
//! Each operation implements `Operation<Image<RawBackend>>` via the blanket impl
//! on `RawOperation`.  Operations are lazy: calling `execute()` returns a new
//! `Image<RawBackend>` with updated decode parameters but does NOT run the
//! demosaic pipeline.  Call `Image::materialize()` to decode pixel data.
//!
//! Future GPU path: when a GPU demosaic kernel is available, these operations
//! will also implement `Operation<Image<GpuBackend>>` using the same `IntoRawEnum`
//! mapping.

use crate::backend::raw::{
    HighlightMode, InterpolationQuality, RawDecodeParams, WhiteBalanceSource,
};

// ── Trait ─────────────────────────────────────────────────────────────────────

/// A RAW decode operation: modifies `RawDecodeParams` without running the
/// decode pipeline.
///
/// Analogous to `VipsOperation` for the VipsBackend — a blanket
/// `impl<Op: RawOperation> Operation<Image<RawBackend>> for Op` in
/// `backend::raw::mod` bridges this into the unified `Operation` framework.
pub trait RawOperation {
    fn apply_to_params(&self, params: &mut RawDecodeParams);
}

// ── Demosaic ──────────────────────────────────────────────────────────────────

/// Select the demosaic (interpolation) algorithm.
///
/// GPU path (future): maps to a compute-shader demosaic kernel via `IntoRawEnum`.
#[derive(Debug, Clone, Copy)]
pub struct DemosaicOperation {
    pub quality: InterpolationQuality,
}

impl RawOperation for DemosaicOperation {
    fn apply_to_params(&self, p: &mut RawDecodeParams) {
        p.interpolation = self.quality;
    }
}

// ── White balance ─────────────────────────────────────────────────────────────

/// Set the white balance source (camera, auto, custom, or none).
#[derive(Debug, Clone, Copy)]
pub struct WhiteBalanceOperation {
    pub source: WhiteBalanceSource,
}

impl RawOperation for WhiteBalanceOperation {
    fn apply_to_params(&self, p: &mut RawDecodeParams) {
        p.white_balance = self.source;
    }
}

// ── Highlight recovery ────────────────────────────────────────────────────────

/// Select the highlight recovery algorithm.
///
/// GPU path (future): maps to a highlight-reconstruction kernel.
#[derive(Debug, Clone, Copy)]
pub struct HighlightOperation {
    pub mode: HighlightMode,
}

impl RawOperation for HighlightOperation {
    fn apply_to_params(&self, p: &mut RawDecodeParams) {
        p.highlights = self.mode;
    }
}

// ── Exposure ──────────────────────────────────────────────────────────────────

/// Unified exposure adjustment — see `pixors_engine::operation::misc::ExposureOperation`.
impl RawOperation for crate::operation::misc::ExposureOperation {
    fn apply_to_params(&self, p: &mut RawDecodeParams) {
        p.exp_correc = true;
        p.exp_shift = self.stops;
        p.exp_preser = self.preserve;
    }
}

/// `BrightnessOperation` from `operation::misc` also implements `RawOperation`.
impl RawOperation for crate::operation::misc::BrightnessOperation {
    fn apply_to_params(&self, p: &mut RawDecodeParams) {
        p.bright = self.value;
    }
}

// ── Noise reduction ───────────────────────────────────────────────────────────

/// Wavelet denoising and FBDD noise reduction.
#[derive(Debug, Clone, Copy)]
pub struct NoiseReductionOperation {
    /// Wavelet threshold (0.0 = off).
    pub threshold: f32,
    /// FBDD level (0 = off, 1 = light, 2 = full).
    pub fbdd: u8,
}

impl RawOperation for NoiseReductionOperation {
    fn apply_to_params(&self, p: &mut RawDecodeParams) {
        p.threshold = self.threshold;
        p.fbdd_noiserd = self.fbdd;
    }
}

// ── Geometry ──────────────────────────────────────────────────────────────────

/// Decode at half the sensor resolution (fast preview).
#[derive(Debug, Clone, Copy)]
pub struct HalfSizeOperation {
    pub enabled: bool,
}

impl RawOperation for HalfSizeOperation {
    fn apply_to_params(&self, p: &mut RawDecodeParams) {
        p.half_size = self.enabled;
    }
}

/// Crop to a sub-region before demosaic.
///
/// `cropbox` is `Some([left, top, width, height])` in raw-sensor coordinates.
#[derive(Debug, Clone, Copy)]
pub struct CropOperation {
    pub cropbox: Option<[u32; 4]>,
}

impl RawOperation for CropOperation {
    fn apply_to_params(&self, p: &mut RawDecodeParams) {
        p.cropbox = self.cropbox;
    }
}

// ── Multi-shot ────────────────────────────────────────────────────────────────

/// Select which shot to decode from a multi-shot (burst / focus-bracket) file.
#[derive(Debug, Clone, Copy)]
pub struct ShotSelectOperation {
    pub index: u32,
}

impl RawOperation for ShotSelectOperation {
    fn apply_to_params(&self, p: &mut RawDecodeParams) {
        p.shot_select = self.index;
    }
}
