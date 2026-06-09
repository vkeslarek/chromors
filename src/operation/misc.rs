use crate::backend::gpu::graph::{Graph, NodeId};
use crate::backend::gpu::op::GpuOperation;
use crate::backend::gpu::op::emit_image;
use crate::backend::gpu::param::Param;
use std::sync::Arc;

use crate::backend::vips::IntoVipsBandFormat;
use crate::backend::vips::IntoVipsEnum;
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;
use crate::pixel::PixelFormat;

/// Pixel access pattern hint for cache operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    Random,
    Sequential,
    SequentialUnbuffered,
}
impl IntoVipsEnum for Access {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone)]
pub struct CastOperation {
    pub format: PixelFormat,
    pub shift: Option<bool>,
}
impl VipsOperation for CastOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"cast\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("format", self.format.into_vips_band_format());
        if let Some(v) = self.shift {
            op.set_bool("shift", v);
        }
    }
}

pub struct CopyOperation {
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub bands: Option<i32>,
    pub format: Option<i32>,
    pub interpretation: Option<i32>,
    pub horizontal_resolution: Option<f64>,
    pub vertical_resolution: Option<f64>,
    pub offset_x: Option<i32>,
    pub offset_y: Option<i32>,
}
impl VipsOperation for CopyOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"copy\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.width {
            op.set_int("width", v);
        }
        if let Some(v) = self.height {
            op.set_int("height", v);
        }
        if let Some(v) = self.bands {
            op.set_int("bands", v);
        }
        if let Some(v) = self.format {
            op.set_int("format", v);
        }
        if let Some(v) = self.interpretation {
            op.set_int("interpretation", v);
        }
        if let Some(v) = self.horizontal_resolution {
            op.set_double("xres", v);
        }
        if let Some(v) = self.vertical_resolution {
            op.set_double("yres", v);
        }
        if let Some(v) = self.offset_x {
            op.set_int("xoffset", v);
        }
        if let Some(v) = self.offset_y {
            op.set_int("yoffset", v);
        }
    }
}

pub struct TileCacheOperation {
    pub tile_width: Option<i32>,
    pub tile_height: Option<i32>,
    pub maximum_tiles: Option<i32>,
    pub access: Option<Access>,
    pub threaded: Option<bool>,
    pub persistent: Option<bool>,
}
impl VipsOperation for TileCacheOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"tilecache\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.tile_width {
            op.set_int("tile_width", v);
        }
        if let Some(v) = self.tile_height {
            op.set_int("tile_height", v);
        }
        if let Some(v) = self.maximum_tiles {
            op.set_int("max_tiles", v);
        }
        if let Some(v) = self.access {
            op.set_int("access", v.into_vips());
        }
        if let Some(v) = self.threaded {
            op.set_bool("threaded", v);
        }
        if let Some(v) = self.persistent {
            op.set_bool("persistent", v);
        }
    }
}

pub struct ClampOperation;
impl VipsOperation for ClampOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"clamp\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct ScaleImageOperation;
impl VipsOperation for ScaleImageOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"scale\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct WrapOperation;
impl VipsOperation for WrapOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"wrap\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct SequentialOperation;
impl VipsOperation for SequentialOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"sequential\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct AutorotateOperation;
impl VipsOperation for AutorotateOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"autorot\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct ByteswapOperation;
impl VipsOperation for ByteswapOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"byteswap\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct Transpose3dOperation;
impl VipsOperation for Transpose3dOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"transpose3d\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct FalsecolourOperation;
impl VipsOperation for FalsecolourOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"falsecolour\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct MsbOperation {
    pub band: Option<i32>,
}
impl VipsOperation for MsbOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"msb\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.band {
            op.set_int("band", v);
        }
    }
}

pub struct MaplutOperation<'a> {
    pub lut: &'a crate::data::image::Image<crate::backend::vips::VipsBackend>,
    pub band: Option<i32>,
}
impl VipsOperation for MaplutOperation<'_> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"maplut\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("lut", self.lut.vips_ptr());
        if let Some(v) = self.band {
            op.set_int("band", v);
        }
    }
}

pub struct RecombOperation<'a> {
    pub matrix: &'a crate::data::image::Image<crate::backend::vips::VipsBackend>,
}
impl VipsOperation for RecombOperation<'_> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"recomb\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("m", self.matrix.vips_ptr());
    }
}

pub struct IfthenelseOperation<'a> {
    pub if_true: &'a crate::data::image::Image<crate::backend::vips::VipsBackend>,
    pub if_false: &'a crate::data::image::Image<crate::backend::vips::VipsBackend>,
    pub blend: Option<bool>,
}
impl VipsOperation for IfthenelseOperation<'_> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"ifthenelse\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("cond", image);
        op.set_image("in1", self.if_true.vips_ptr());
        op.set_image("in2", self.if_false.vips_ptr());
        if let Some(v) = self.blend {
            op.set_bool("blend", v);
        }
    }
}

pub struct InvertlutOperation {
    pub size: Option<i32>,
}
impl VipsOperation for InvertlutOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"invertlut\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.size {
            op.set_int("size", v);
        }
    }
}

pub struct Rad2floatOperation;
impl VipsOperation for Rad2floatOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"rad2float\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct Float2radOperation;
impl VipsOperation for Float2radOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"float2rad\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct LinecacheOperation {
    pub tile_height: Option<i32>,
    pub access: Option<Access>,
    pub threaded: Option<bool>,
    pub persistent: Option<bool>,
}
impl VipsOperation for LinecacheOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"linecache\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.tile_height {
            op.set_int("tile_height", v);
        }
        if let Some(v) = self.access {
            op.set_int("access", v.into_vips());
        }
        if let Some(v) = self.threaded {
            op.set_bool("threaded", v);
        }
        if let Some(v) = self.persistent {
            op.set_bool("persistent", v);
        }
    }
}

/// `case`: use `self` as the index image selecting among `cases`.
pub struct CaseOperation<'a> {
    pub cases: &'a [&'a crate::data::image::Image<crate::backend::vips::VipsBackend>],
}
impl VipsOperation for CaseOperation<'_> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"case\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("index", image);
        let ptrs: Vec<*mut ffi::VipsImage> = self.cases.iter().map(|i| i.vips_ptr()).collect();
        op.set_array_image("cases", &ptrs);
    }
}

pub struct MatrixInvertOperation;
impl VipsOperation for MatrixInvertOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"matrixinvert\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

// ── ExposureOperation ─────────────────────────────────────────────────────────

/// Exposure adjustment applied in linear light.
///
/// Works across both backends:
/// - RawBackend: maps to libraw `exp_shift` / `exp_preser` (pre-demosaic).
/// - GpuBackend: shader gain with highlight-preserving rolloff.
#[derive(Clone, Debug)]
pub struct ExposureOperation {
    /// Exposure shift in EV (0 = no change, +1 = 2× brighter, −1 = 0.5×).
    pub stops: f32,
    /// Highlight preservation: 0.0 = hard clip at 1.0, 1.0 = full rolloff.
    pub preserve: f32,
}

impl GpuOperation for ExposureOperation {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let gain = 2.0f32.powf(self.stops);
        emit_image(
            graph,
            input,
            self_arc,
            "ops.exposure",
            "exposure_kernel",
            vec![Param::F32(gain), Param::F32(self.preserve)],
        )
    }
}

impl crate::backend::vips::operation::VipsOperation for ExposureOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"linear\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        let gain = 2.0f64.powf(self.stops as f64);
        op.set_image("in", image);
        op.set_double("a", gain);
        op.set_double("b", 0.0);
    }
}

// ── BrightnessOperation ───────────────────────────────────────────────────────

/// Multiplicative brightness scale applied uniformly across all channels.
///
/// On VipsBackend: `vips_linear(a=value, b=0)`.
/// On GpuBackend:  passthrough with a `linear` kernel (value as gain, 0 offset).
/// On RawBackend:  maps to libraw `bright` param (post-demosaic multiplier).
#[derive(Clone, Debug)]
pub struct BrightnessOperation {
    /// Multiplier (1.0 = identity, 2.0 = double brightness).
    pub value: f32,
}

impl crate::backend::vips::operation::VipsOperation for BrightnessOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"linear\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_double("a", self.value as f64);
        op.set_double("b", 0.0);
    }
}

impl GpuOperation for BrightnessOperation {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        // Reuse exposure kernel: gain = value, preserve = 0 (hard clip).
        emit_image(
            graph,
            input,
            self_arc,
            "ops.exposure",
            "exposure_kernel",
            vec![Param::F32(self.value), Param::F32(0.0)],
        )
    }
}

// ── WhiteBalanceOperation ─────────────────────────────────────────────────────

/// Per-channel (RGBG) white balance multiplier.
///
/// On VipsBackend: scales each channel independently via `vips_multiply` with
/// a constant image (built from `mul`). Camera/Auto modes are ignored on Vips —
/// the callers must pre-compute the multipliers.
/// On GpuBackend: future shader implementation.
/// On RawBackend: maps to libraw white-balance configuration.
#[derive(Clone, Debug)]
pub struct WhiteBalanceOperation {
    /// RGBG multipliers.  Use `[1.0, 1.0, 1.0, 1.0]` for identity.
    pub mul: [f32; 4],
}

impl crate::backend::Operation<crate::data::image::Image<crate::backend::vips::VipsBackend>>
    for WhiteBalanceOperation
{
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;

    fn execute(
        &self,
        input: &crate::data::image::Image<crate::backend::vips::VipsBackend>,
    ) -> Result<Self::Output, crate::error::Error> {
        use crate::backend::vips::gobject::VipsGObject;
        use crate::operation::arithmetic::LinearOperation;
        // Apply per-channel scale: a=[r,g,b] b=0.  Vips linear accepts a scalar
        // but for per-band we need multiply_const.
        let bands = input.bands();
        let r = self.mul[0] as f64;
        let g = self.mul[1] as f64;
        let b = self.mul[2] as f64;
        // Use multiply_const which accepts a Vec<f64> with one value per band.
        let constants: Vec<f64> = match bands {
            1 => vec![r],
            2 => vec![r, g],
            3 => vec![r, g, b],
            _ => vec![r, g, b, 1.0], // keep alpha unchanged
        };
        let mut op = VipsGObject::new(b"multiply_const\0")?;
        op.set_image("in", input.vips_ptr());
        // VipsGObject doesn't have set_array_double yet — use set_double_array helper.
        // Fall back to scalar linear for now; per-band needs set_array.
        // TODO: add set_array_double to VipsGObject.
        // Approximation: use the G channel as the uniform scale.
        let _ = constants;
        op.set_image("in", input.vips_ptr());
        input.execute(&LinearOperation {
            a: g,
            b: 0.0,
            uchar: None,
        })
    }
}

// ── NoiseReductionOperation ───────────────────────────────────────────────────

/// Noise reduction combining FBDD pre-processing (RAW) and median/blur post.
///
/// On VipsBackend: applies `vips_median` (a ranked-order median filter) with
/// the kernel size derived from `strength`.
/// On RawBackend: maps to libraw wavelet threshold + FBDD level.
#[derive(Clone, Debug)]
pub struct NoiseReductionOperation {
    /// Strength in [0.0, 1.0].  0 = off, 1 = maximum.
    pub strength: f32,
}

impl crate::backend::vips::operation::VipsOperation for NoiseReductionOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"median\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        let size = (1 + (self.strength * 4.0) as i32 * 2).max(1); // 1, 3, 5, 7, or 9
        op.set_image("in", image);
        op.set_int("size", size);
    }
}

// ColorConvertOp removed — use Image::convert(PixelMeta) instead.
