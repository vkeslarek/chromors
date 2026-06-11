use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{IntoVipsEnum, VipsBackend, VipsBuilder};
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

// ── Enums ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kernel { Nearest, Linear, Cubic, Mitchell, Lanczos2, Lanczos3 }
impl IntoVipsEnum for Kernel { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction { Horizontal, Vertical }
impl IntoVipsEnum for Direction { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Angle { D0, D90, D180, D270 }
impl IntoVipsEnum for Angle { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Angle45 { D0, D45, D90, D135, D180, D225, D270, D315 }
impl IntoVipsEnum for Angle45 { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Extend { Black, Copy, Repeat, Mirror, White, Background }
impl IntoVipsEnum for Extend { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Interesting { None, Centre, Entropy, Attention, Low, High, All }
impl IntoVipsEnum for Interesting { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompassDirection { Centre, North, East, South, West, NorthEast, SouthEast, SouthWest, NorthWest }
impl IntoVipsEnum for CompassDirection { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Size { Both, Up, Down, Force }
impl IntoVipsEnum for Size { fn into_vips(self) -> i32 { self as i32 } }

// ── Operations ────────────────────────────────────────────────────────────────

pub struct Crop<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}
impl<B: Backend> Operation<B> for Crop<B> where Crop<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(Region {
            x: out.x + self.left,
            y: out.y + self.top,
            w: out.w,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind { 
        let mut spec = (*self.input.spec).clone();
        spec.width = self.width;
        spec.height = self.height;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.left);
        state.write_i32(self.top);
        state.write_i32(self.width);
        state.write_i32(self.height);
    }
}
impl Lower<VipsBackend> for Crop<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"crop\0").unwrap();
        op.set_image("input", input_handle.ptr);
        op.set_int("left", self.left);
        op.set_int("top", self.top);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────

impl Lower<GpuBackend> for Crop<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("passthrough_kernel");
        cx.output(self.output_spec().output());
    }
}

pub struct Embed<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub extend: Option<Extend>,
    pub background: Option<[f64; 3]>,
}
impl<B: Backend> Operation<B> for Embed<B> where Embed<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.x);
        state.write_i32(self.y);
        state.write_i32(self.width);
        state.write_i32(self.height);
        if let Some(e) = self.extend { state.write_i32(e.into_vips()); }
    }
}
impl Lower<VipsBackend> for Embed<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"embed\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("x", self.x);
        op.set_int("y", self.y);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(e) = self.extend { op.set_int("extend", e.into_vips()); }
        if let Some(bg) = self.background { op.set_array_double("background", &bg); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Flip<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub direction: Direction,
}
impl<B: Backend> Operation<B> for Flip<B> where Flip<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.direction.into_vips());
    }
}
impl Lower<VipsBackend> for Flip<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"flip\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("direction", self.direction.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Rot90<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub angle: Angle,
}
impl<B: Backend> Operation<B> for Rot90<B> where Rot90<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.angle.into_vips());
    }
}
impl Lower<VipsBackend> for Rot90<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"rot\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("angle", self.angle.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Rot45<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub angle: Angle45,
}
impl<B: Backend> Operation<B> for Rot45<B> where Rot45<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.angle.into_vips());
    }
}
impl Lower<VipsBackend> for Rot45<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"rot45\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("angle", self.angle.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Rotate<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub angle: f64,
    pub background: Option<[f64; 3]>,
    pub offset_input_x: Option<f64>,
    pub offset_input_y: Option<f64>,
    pub offset_output_x: Option<f64>,
    pub offset_output_y: Option<f64>,
}
impl<B: Backend> Operation<B> for Rotate<B> where Rotate<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.angle.to_bits());
        state.write_u64(self.offset_input_x.unwrap_or(0.0).to_bits());
        state.write_u64(self.offset_input_y.unwrap_or(0.0).to_bits());
        state.write_u64(self.offset_output_x.unwrap_or(0.0).to_bits());
        state.write_u64(self.offset_output_y.unwrap_or(0.0).to_bits());
    }
}
impl Lower<VipsBackend> for Rotate<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"rotate\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("angle", self.angle);
        if let Some(bg) = self.background { op.set_array_double("background", &bg); }
        if let Some(v) = self.offset_input_x { op.set_double("idx", v); }
        if let Some(v) = self.offset_input_y { op.set_double("idy", v); }
        if let Some(v) = self.offset_output_x { op.set_double("odx", v); }
        if let Some(v) = self.offset_output_y { op.set_double("ody", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Smartcrop<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub width: i32,
    pub height: i32,
    pub interesting: Option<Interesting>,
}
impl<B: Backend> Operation<B> for Smartcrop<B> where Smartcrop<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height);
        if let Some(i) = self.interesting { state.write_i32(i.into_vips()); }
    }
}
impl Lower<VipsBackend> for Smartcrop<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"smartcrop\0").unwrap();
        op.set_image("input", input_handle.ptr);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(i) = self.interesting { op.set_int("interesting", i.into_vips()); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Gravity<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub direction: CompassDirection,
    pub width: i32,
    pub height: i32,
    pub extend: Option<Extend>,
    pub background: Option<[f64; 3]>,
}
impl<B: Backend> Operation<B> for Gravity<B> where Gravity<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.direction.into_vips());
        state.write_i32(self.width);
        state.write_i32(self.height);
        if let Some(e) = self.extend { state.write_i32(e.into_vips()); }
    }
}
impl Lower<VipsBackend> for Gravity<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"gravity\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("direction", self.direction.into_vips());
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(e) = self.extend { op.set_int("extend", e.into_vips()); }
        if let Some(bg) = self.background { op.set_array_double("background", &bg); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Resize<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub scale: f64,
    pub kernel: Option<Kernel>,
    pub vertical_scale: Option<f64>,
    pub gap: Option<f64>,
}
impl<B: Backend> Operation<B> for Resize<B> where Resize<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.scale.to_bits());
        state.write_u64(self.vertical_scale.unwrap_or(0.0).to_bits());
        state.write_u64(self.gap.unwrap_or(0.0).to_bits());
        if let Some(k) = self.kernel { state.write_i32(k.into_vips()); }
    }
}
impl Lower<VipsBackend> for Resize<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"resize\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("scale", self.scale);
        if let Some(k) = self.kernel { op.set_int("kernel", k.into_vips()); }
        if let Some(v) = self.vertical_scale { op.set_double("vscale", v); }
        if let Some(g) = self.gap { op.set_double("gap", g); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Thumbnail<B: Backend> {
    pub input: Input<ImageKind, B>,
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
impl<B: Backend> Operation<B> for Thumbnail<B> where Thumbnail<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height.unwrap_or(0));
        state.write_i32(self.size.unwrap_or(0));
        if let Some(c) = self.crop { state.write_i32(c.into_vips()); }
        state.write_i32(self.intent.unwrap_or(0));
    }
}
impl Lower<VipsBackend> for Thumbnail<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"thumbnail_image\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("width", self.width);
        if let Some(h) = self.height { op.set_int("height", h); }
        if let Some(s) = self.size { op.set_int("size", s); }
        if let Some(c) = self.crop { op.set_int("crop", c.into_vips()); }
        if let Some(v) = self.linear { op.set_bool("linear", v); }
        if let Some(v) = self.auto_rotate { op.set_bool("auto_rotate", v); }
        if let Some(v) = self.no_rotate { op.set_bool("no_rotate", v); }
        if let Some(ref v) = self.import_profile { op.set_string("import_profile", v); }
        if let Some(ref v) = self.export_profile { op.set_string("export_profile", v); }
        if let Some(v) = self.intent { op.set_int("intent", v); }
        if let Some(v) = self.fail_on { op.set_int("fail_on", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Shrink<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub horizontal: f64,
    pub vertical: f64,
    pub ceil: Option<bool>,
}
impl<B: Backend> Operation<B> for Shrink<B> where Shrink<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let hf = self.horizontal.ceil() as i32;
        let vf = self.vertical.ceil() as i32;
        vec![Some(WorkUnit::Region(Region {
            x: out.x * hf,
            y: out.y * vf,
            w: out.w * hf,
            h: out.h * vf,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.horizontal.to_bits());
        state.write_u64(self.vertical.to_bits());
    }
}
impl Lower<VipsBackend> for Shrink<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"shrink\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("hshrink", self.horizontal);
        op.set_double("vshrink", self.vertical);
        if let Some(c) = self.ceil { op.set_bool("ceil", c); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Reduce<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub horizontal: f64,
    pub vertical: f64,
    pub kernel: Option<Kernel>,
    pub gap: Option<f64>,
}
impl<B: Backend> Operation<B> for Reduce<B> where Reduce<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] } // approx
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.horizontal.to_bits());
        state.write_u64(self.vertical.to_bits());
        if let Some(k) = self.kernel { state.write_i32(k.into_vips()); }
        state.write_u64(self.gap.unwrap_or(0.0).to_bits());
    }
}
impl Lower<VipsBackend> for Reduce<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"reduce\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("hshrink", self.horizontal);
        op.set_double("vshrink", self.vertical);
        if let Some(k) = self.kernel { op.set_int("kernel", k.into_vips()); }
        if let Some(g) = self.gap { op.set_double("gap", g); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct ReduceHorizontal<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub shrink: f64,
    pub kernel: Option<Kernel>,
    pub gap: Option<f64>,
}
impl<B: Backend> Operation<B> for ReduceHorizontal<B> where ReduceHorizontal<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.shrink.to_bits());
        if let Some(k) = self.kernel { state.write_i32(k.into_vips()); }
        state.write_u64(self.gap.unwrap_or(0.0).to_bits());
    }
}
impl Lower<VipsBackend> for ReduceHorizontal<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"reduceh\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("hshrink", self.shrink);
        if let Some(k) = self.kernel { op.set_int("kernel", k.into_vips()); }
        if let Some(g) = self.gap { op.set_double("gap", g); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct ReduceVertical<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub shrink: f64,
    pub kernel: Option<Kernel>,
    pub gap: Option<f64>,
}
impl<B: Backend> Operation<B> for ReduceVertical<B> where ReduceVertical<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.shrink.to_bits());
        if let Some(k) = self.kernel { state.write_i32(k.into_vips()); }
        state.write_u64(self.gap.unwrap_or(0.0).to_bits());
    }
}
impl Lower<VipsBackend> for ReduceVertical<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"reducev\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("vshrink", self.shrink);
        if let Some(k) = self.kernel { op.set_int("kernel", k.into_vips()); }
        if let Some(g) = self.gap { op.set_double("gap", g); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct ShrinkHorizontal<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub shrink: i32,
    pub ceil: Option<bool>,
}
impl<B: Backend> Operation<B> for ShrinkHorizontal<B> where ShrinkHorizontal<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) { state.write_i32(self.shrink); }
}
impl Lower<VipsBackend> for ShrinkHorizontal<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"shrinkh\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("hshrink", self.shrink);
        if let Some(c) = self.ceil { op.set_bool("ceil", c); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct ShrinkVertical<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub shrink: i32,
    pub ceil: Option<bool>,
}
impl<B: Backend> Operation<B> for ShrinkVertical<B> where ShrinkVertical<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) { state.write_i32(self.shrink); }
}
impl Lower<VipsBackend> for ShrinkVertical<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"shrinkv\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("vshrink", self.shrink);
        if let Some(c) = self.ceil { op.set_bool("ceil", c); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct ExtractArea<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}
impl<B: Backend> Operation<B> for ExtractArea<B> where ExtractArea<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(Region {
            x: out.x + self.left, y: out.y + self.top, w: out.w, h: out.h, lod: out.lod
        }))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.left);
        state.write_i32(self.top);
        state.write_i32(self.width);
        state.write_i32(self.height);
    }
}
impl Lower<VipsBackend> for ExtractArea<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"extract_area\0").unwrap();
        op.set_image("input", input_handle.ptr);
        op.set_int("left", self.left);
        op.set_int("top", self.top);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Subsample<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub horizontal: i32,
    pub vertical: i32,
    pub point: Option<bool>,
}
impl<B: Backend> Operation<B> for Subsample<B> where Subsample<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.horizontal);
        state.write_i32(self.vertical);
    }
}
impl Lower<VipsBackend> for Subsample<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"subsample\0").unwrap();
        op.set_image("input", input_handle.ptr);
        op.set_int("xfac", self.horizontal);
        op.set_int("yfac", self.vertical);
        if let Some(p) = self.point { op.set_bool("point", p); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Zoom<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub horizontal: i32,
    pub vertical: i32,
}
impl<B: Backend> Operation<B> for Zoom<B> where Zoom<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.horizontal);
        state.write_i32(self.vertical);
    }
}
impl Lower<VipsBackend> for Zoom<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"zoom\0").unwrap();
        op.set_image("input", input_handle.ptr);
        op.set_int("xfac", self.horizontal);
        op.set_int("yfac", self.vertical);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Replicate<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub across: i32,
    pub down: i32,
}
impl<B: Backend> Operation<B> for Replicate<B> where Replicate<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.across);
        state.write_i32(self.down);
    }
}
impl Lower<VipsBackend> for Replicate<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"replicate\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("across", self.across);
        op.set_int("down", self.down);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Grid<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub tile_height: i32,
    pub across: i32,
    pub down: i32,
}
impl<B: Backend> Operation<B> for Grid<B> where Grid<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.tile_height);
        state.write_i32(self.across);
        state.write_i32(self.down);
    }
}
impl Lower<VipsBackend> for Grid<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"grid\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("tile_height", self.tile_height);
        op.set_int("across", self.across);
        op.set_int("down", self.down);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}
