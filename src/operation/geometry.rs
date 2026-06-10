use crate::backend::gpu::datatype::ImageType;
use crate::backend::gpu::graph::{Graph, NodeId};
use crate::backend::gpu::op::emit_image;
use crate::backend::gpu::op::{GpuOperation, TypedOperation};
use crate::backend::gpu::param::Param;
use crate::geometry::Rect;
use std::sync::Arc;

use crate::backend::vips::IntoVipsEnum;
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

// -- Geometry / resample enums --
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kernel {
    Nearest,
    Linear,
    Cubic,
    Mitchell,
    Lanczos2,
    Lanczos3,
}
impl IntoVipsEnum for Kernel {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Horizontal,
    Vertical,
}
impl IntoVipsEnum for Direction {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Angle {
    D0,
    D90,
    D180,
    D270,
}
impl IntoVipsEnum for Angle {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Angle45 {
    D0,
    D45,
    D90,
    D135,
    D180,
    D225,
    D270,
    D315,
}
impl IntoVipsEnum for Angle45 {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Extend {
    Black,
    Copy,
    Repeat,
    Mirror,
    White,
    Background,
}
impl IntoVipsEnum for Extend {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Interesting {
    None,
    Centre,
    Entropy,
    Attention,
    Low,
    High,
    All,
}
impl IntoVipsEnum for Interesting {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompassDirection {
    Centre,
    North,
    East,
    South,
    West,
    NorthEast,
    SouthEast,
    SouthWest,
    NorthWest,
}
impl IntoVipsEnum for CompassDirection {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone)]
pub struct CropOperation {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}

impl VipsOperation for CropOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"crop\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("input", image);
        op.set_int("left", self.left);
        op.set_int("top", self.top);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

impl GpuOperation for CropOperation {
    fn output_dims(&self, _input_w: u32, _input_h: u32) -> Option<(u32, u32)> {
        Some((self.width as u32, self.height as u32))
    }

    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        emit_image(
            graph,
            input,
            self_arc,
            "ops.passthrough",
            "passthrough_kernel",
            vec![
                Param::I32(self.left),
                Param::I32(self.top),
                Param::I32(self.width),
                Param::I32(self.height),
            ],
        )
    }

    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        match wu {
            crate::backend::gpu::work_unit::WorkUnit::Region { rect, lod } => vec![(
                0,
                crate::backend::gpu::work_unit::WorkUnit::Region {
                    rect: Rect::new(
                        rect.x + self.left,
                        rect.y + self.top,
                        rect.width,
                        rect.height,
                    ),
                    lod: *lod,
                },
            )],
            _ => vec![(0, wu.clone())],
        }
    }
}

pub struct EmbedOperation {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub extend: Option<Extend>,
    pub background: Option<[f64; 3]>,
}

impl VipsOperation for EmbedOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"embed\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("x", self.x);
        op.set_int("y", self.y);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(v) = self.extend {
            op.set_int("extend", v.into_vips());
        }
        if let Some(v) = &self.background {
            op.set_array_double("background", v);
        }
    }
}

pub struct FlipOperation {
    pub direction: Direction,
}

impl VipsOperation for FlipOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"flip\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("direction", self.direction.into_vips());
    }
}

pub struct Rot90Operation {
    pub angle: Angle,
}

impl VipsOperation for Rot90Operation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"rot\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("angle", self.angle.into_vips());
    }
}

pub struct Rot45Operation {
    pub angle: Angle45,
}

impl VipsOperation for Rot45Operation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"rot45\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("angle", self.angle.into_vips());
    }
}

pub struct RotateOperation<'a> {
    pub angle: f64,
    pub interpolate: Option<&'a crate::backend::vips::Interpolate>,
    pub background: Option<[f64; 3]>,
    pub offset_input_x: Option<f64>,
    pub offset_input_y: Option<f64>,
    pub offset_output_x: Option<f64>,
    pub offset_output_y: Option<f64>,
}

impl VipsOperation for RotateOperation<'_> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"rotate\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_double("angle", self.angle);
        if let Some(v) = self.interpolate {
            op.set_interpolate("interpolate", v);
        }
        if let Some(v) = &self.background {
            op.set_array_double("background", v);
        }
        if let Some(v) = self.offset_input_x {
            op.set_double("idx", v);
        }
        if let Some(v) = self.offset_input_y {
            op.set_double("idy", v);
        }
        if let Some(v) = self.offset_output_x {
            op.set_double("odx", v);
        }
        if let Some(v) = self.offset_output_y {
            op.set_double("ody", v);
        }
    }
}

pub struct SmartcropOperation {
    pub width: i32,
    pub height: i32,
    pub interesting: Option<Interesting>,
}

impl VipsOperation for SmartcropOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"smartcrop\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("input", image);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(v) = self.interesting {
            op.set_int("interesting", v.into_vips());
        }
    }
}

pub struct GravityOperation {
    pub direction: CompassDirection,
    pub width: i32,
    pub height: i32,
    pub extend: Option<Extend>,
    pub background: Option<[f64; 3]>,
}

impl VipsOperation for GravityOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"gravity\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("direction", self.direction.into_vips());
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(v) = self.extend {
            op.set_int("extend", v.into_vips());
        }
        if let Some(v) = &self.background {
            op.set_array_double("background", v);
        }
    }
}

pub struct ResizeOperation {
    pub scale: f64,
    pub kernel: Option<Kernel>,
    pub vertical_scale: Option<f64>,
    pub gap: Option<f64>,
}

impl VipsOperation for ResizeOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"resize\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_double("scale", self.scale);
        if let Some(v) = self.kernel {
            op.set_int("kernel", v.into_vips());
        }
        if let Some(v) = self.vertical_scale {
            op.set_double("vscale", v);
        }
        if let Some(v) = self.gap {
            op.set_double("gap", v);
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShrinkOperation {
    pub horizontal: f64,
    pub vertical: f64,
    pub ceil: Option<bool>,
}

impl VipsOperation for ShrinkOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"shrink\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_double("hshrink", self.horizontal);
        op.set_double("vshrink", self.vertical);
        if let Some(v) = self.ceil {
            op.set_bool("ceil", v);
        }
    }
}

pub struct ReduceOperation {
    pub horizontal: f64,
    pub vertical: f64,
    pub kernel: Option<Kernel>,
    pub gap: Option<f64>,
}

impl VipsOperation for ReduceOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"reduce\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_double("hshrink", self.horizontal);
        op.set_double("vshrink", self.vertical);
        if let Some(v) = self.kernel {
            op.set_int("kernel", v.into_vips());
        }
        if let Some(v) = self.gap {
            op.set_double("gap", v);
        }
    }
}

pub struct ThumbnailOperation {
    pub width: i32,
    pub height: Option<i32>,
    pub size: Option<i32>,
    pub crop: Option<Interesting>,
    pub linear: Option<bool>,
    pub auto_rotate: Option<bool>,
    pub no_rotate: Option<bool>,
    pub import_profile: Option<String>,
    pub export_profile: Option<String>,
    pub intent: Option<i32>,
    pub fail_on: Option<i32>,
}

impl VipsOperation for ThumbnailOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"thumbnail_image\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("width", self.width);
        if let Some(v) = self.height {
            op.set_int("height", v);
        }
        if let Some(v) = self.size {
            op.set_int("size", v);
        }
        if let Some(v) = self.crop {
            op.set_int("crop", v.into_vips());
        }
        if let Some(v) = self.linear {
            op.set_bool("linear", v);
        }
        if let Some(v) = self.auto_rotate {
            op.set_bool("auto_rotate", v);
        }
        if let Some(v) = self.no_rotate {
            op.set_bool("no_rotate", v);
        }
        if let Some(ref v) = self.import_profile {
            op.set_string("import_profile", v);
        }
        if let Some(ref v) = self.export_profile {
            op.set_string("export_profile", v);
        }
        if let Some(v) = self.intent {
            op.set_int("intent", v);
        }
        if let Some(v) = self.fail_on {
            op.set_int("fail_on", v);
        }
    }
}

pub struct AffineOperation<'a> {
    pub matrix: Vec<f64>,
    pub interpolate: Option<&'a crate::backend::vips::Interpolate>,
    pub output_area: Option<[i32; 4]>,
    pub offset_input_x: Option<f64>,
    pub offset_input_y: Option<f64>,
    pub offset_output_x: Option<f64>,
    pub offset_output_y: Option<f64>,
    pub background: Option<Vec<f64>>,
    pub premultiplied: Option<bool>,
    pub extend: Option<Extend>,
}
impl VipsOperation for AffineOperation<'_> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"affine\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_array_double("matrix", &self.matrix);
        if let Some(v) = self.interpolate {
            op.set_interpolate("interpolate", v);
        }
        if let Some(v) = &self.output_area {
            op.set_array_int("oarea", v);
        }
        if let Some(v) = self.offset_input_x {
            op.set_double("idx", v);
        }
        if let Some(v) = self.offset_input_y {
            op.set_double("idy", v);
        }
        if let Some(v) = self.offset_output_x {
            op.set_double("odx", v);
        }
        if let Some(v) = self.offset_output_y {
            op.set_double("ody", v);
        }
        if let Some(v) = &self.background {
            op.set_array_double("background", v);
        }
        if let Some(v) = self.premultiplied {
            op.set_bool("premultiplied", v);
        }
        if let Some(v) = self.extend {
            op.set_int("extend", v.into_vips());
        }
    }
}

pub struct SimilarityOperation<'a> {
    pub scale: Option<f64>,
    pub angle: Option<f64>,
    pub interpolate: Option<&'a crate::backend::vips::Interpolate>,
    pub background: Option<Vec<f64>>,
    pub offset_input_x: Option<f64>,
    pub offset_input_y: Option<f64>,
    pub offset_output_x: Option<f64>,
    pub offset_output_y: Option<f64>,
}
impl VipsOperation for SimilarityOperation<'_> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"similarity\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.scale {
            op.set_double("scale", v);
        }
        if let Some(v) = self.angle {
            op.set_double("angle", v);
        }
        if let Some(v) = self.interpolate {
            op.set_interpolate("interpolate", v);
        }
        if let Some(v) = &self.background {
            op.set_array_double("background", v);
        }
        if let Some(v) = self.offset_input_x {
            op.set_double("idx", v);
        }
        if let Some(v) = self.offset_input_y {
            op.set_double("idy", v);
        }
        if let Some(v) = self.offset_output_x {
            op.set_double("odx", v);
        }
        if let Some(v) = self.offset_output_y {
            op.set_double("ody", v);
        }
    }
}

pub struct MapimOperation<'a> {
    pub index: &'a crate::data::image::Image2D<crate::backend::vips::VipsBackend>,
    pub interpolate: Option<&'a crate::backend::vips::Interpolate>,
    pub background: Option<Vec<f64>>,
    pub premultiplied: Option<bool>,
    pub extend: Option<Extend>,
}
impl VipsOperation for MapimOperation<'_> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"mapim\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("index", self.index.vips_ptr());
        if let Some(v) = self.interpolate {
            op.set_interpolate("interpolate", v);
        }
        if let Some(v) = &self.background {
            op.set_array_double("background", v);
        }
        if let Some(v) = self.premultiplied {
            op.set_bool("premultiplied", v);
        }
        if let Some(v) = self.extend {
            op.set_int("extend", v.into_vips());
        }
    }
}

pub struct QuadraticOperation<'a> {
    pub coeff: &'a crate::data::image::Image2D<crate::backend::vips::VipsBackend>,
    pub interpolate: Option<&'a crate::backend::vips::Interpolate>,
}
impl VipsOperation for QuadraticOperation<'_> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"quadratic\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("coeff", self.coeff.vips_ptr());
        if let Some(v) = self.interpolate {
            op.set_interpolate("interpolate", v);
        }
    }
}

pub struct ExtractAreaOperation {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}
impl VipsOperation for ExtractAreaOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"extract_area\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("input", image);
        op.set_int("left", self.left);
        op.set_int("top", self.top);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct SubsampleOperation {
    pub horizontal: i32,
    pub vertical: i32,
    pub point: Option<bool>,
}
impl VipsOperation for SubsampleOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"subsample\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("input", image);
        op.set_int("xfac", self.horizontal);
        op.set_int("yfac", self.vertical);
        if let Some(v) = self.point {
            op.set_bool("point", v);
        }
    }
}

pub struct ZoomOperation {
    pub horizontal: i32,
    pub vertical: i32,
}
impl VipsOperation for ZoomOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"zoom\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("input", image);
        op.set_int("xfac", self.horizontal);
        op.set_int("yfac", self.vertical);
    }
}

pub struct ReplicateOperation {
    pub across: i32,
    pub down: i32,
}
impl VipsOperation for ReplicateOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"replicate\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("across", self.across);
        op.set_int("down", self.down);
    }
}

pub struct GridOperation {
    pub tile_height: i32,
    pub across: i32,
    pub down: i32,
}
impl VipsOperation for GridOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"grid\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("tile_height", self.tile_height);
        op.set_int("across", self.across);
        op.set_int("down", self.down);
    }
}

/// Sizing constraint for thumbnail operations (`VipsSize`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Size {
    Both,
    Up,
    Down,
    Force,
}
impl IntoVipsEnum for Size {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

pub struct ReduceHorizontalOperation {
    pub shrink: f64,
    pub kernel: Option<Kernel>,
    pub gap: Option<f64>,
}
impl VipsOperation for ReduceHorizontalOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"reduceh\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_double("hshrink", self.shrink);
        if let Some(v) = self.kernel {
            op.set_int("kernel", v.into_vips());
        }
        if let Some(v) = self.gap {
            op.set_double("gap", v);
        }
    }
}

pub struct ReduceVerticalOperation {
    pub shrink: f64,
    pub kernel: Option<Kernel>,
    pub gap: Option<f64>,
}
impl VipsOperation for ReduceVerticalOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"reducev\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_double("vshrink", self.shrink);
        if let Some(v) = self.kernel {
            op.set_int("kernel", v.into_vips());
        }
        if let Some(v) = self.gap {
            op.set_double("gap", v);
        }
    }
}

pub struct ShrinkHorizontalOperation {
    pub shrink: i32,
    pub ceil: Option<bool>,
}
impl VipsOperation for ShrinkHorizontalOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"shrinkh\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("hshrink", self.shrink);
        if let Some(v) = self.ceil {
            op.set_bool("ceil", v);
        }
    }
}

pub struct ShrinkVerticalOperation {
    pub shrink: i32,
    pub ceil: Option<bool>,
}
impl VipsOperation for ShrinkVerticalOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"shrinkv\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("vshrink", self.shrink);
        if let Some(v) = self.ceil {
            op.set_bool("ceil", v);
        }
    }
}

// ── ShrinkOperation ───────────────────────────────────────────────────────────

impl TypedOperation for ShrinkOperation {
    type Output = ImageType;
}

impl GpuOperation for ShrinkOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let h = self.horizontal.ceil() as u32;
        let v = self.vertical.ceil() as u32;
        emit_image(
            graph,
            input,
            self_arc,
            "ops.shrink",
            "shrink_kernel",
            vec![Param::U32(h), Param::U32(v)],
        )
    }

    fn output_dims(&self, w: u32, h: u32) -> Option<(u32, u32)> {
        let hf = self.horizontal.ceil() as u32;
        let vf = self.vertical.ceil() as u32;
        Some((w.div_ceil(hf), h.div_ceil(vf)))
    }

    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        match wu {
            crate::backend::gpu::work_unit::WorkUnit::Region { rect, lod } => {
                let hf = self.horizontal.ceil() as i32;
                let vf = self.vertical.ceil() as i32;
                vec![(
                    0,
                    crate::backend::gpu::work_unit::WorkUnit::Region {
                        rect: Rect::new(
                            rect.x * hf,
                            rect.y * vf,
                            rect.width * hf,
                            rect.height * vf,
                        ),
                        lod: *lod,
                    },
                )]
            }
            _ => vec![(0, wu.clone())],
        }
    }
}
