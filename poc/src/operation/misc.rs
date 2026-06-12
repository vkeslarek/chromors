use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{IntoVipsBandFormat, IntoVipsEnum, VipsBackend, VipsBuilder};
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::gpu::view::ParamBlock;
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::pixel::PixelFormat;
use crate::work_unit::{Region, WorkUnit};

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

// ── Cast ──────────────────────────────────────────────────────────────────────

pub struct Cast<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub format: PixelFormat,
    pub shift: Option<bool>,
}

impl<B: Backend> Operation<B> for Cast<B> where Cast<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.format = self.format;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.format.into_vips_band_format());
        if let Some(v) = self.shift { state.write_u8(v as u8); }
    }
}

impl Lower<VipsBackend> for Cast<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"cast\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("format", self.format.into_vips_band_format());
        if let Some(v) = self.shift {
            op.set_bool("shift", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Copy ──────────────────────────────────────────────────────────────────────

pub struct Copy<B: Backend> {
    pub input: Input<ImageKind, B>,
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

impl<B: Backend> Operation<B> for Copy<B> where Copy<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        if let Some(w) = self.width { spec.width = w; }
        if let Some(h) = self.height { spec.height = h; }
        // For bands/format/interpretation, one should probably map properly, but we keep it simple.
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.width { state.write_i32(v); }
        if let Some(v) = self.height { state.write_i32(v); }
        if let Some(v) = self.bands { state.write_i32(v); }
        if let Some(v) = self.format { state.write_i32(v); }
        if let Some(v) = self.interpretation { state.write_i32(v); }
        if let Some(v) = self.horizontal_resolution { state.write_u64(v.to_bits()); }
        if let Some(v) = self.vertical_resolution { state.write_u64(v.to_bits()); }
        if let Some(v) = self.offset_x { state.write_i32(v); }
        if let Some(v) = self.offset_y { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for Copy<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"copy\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.width { op.set_int("width", v); }
        if let Some(v) = self.height { op.set_int("height", v); }
        if let Some(v) = self.bands { op.set_int("bands", v); }
        if let Some(v) = self.format { op.set_int("format", v); }
        if let Some(v) = self.interpretation { op.set_int("interpretation", v); }
        if let Some(v) = self.horizontal_resolution { op.set_double("xres", v); }
        if let Some(v) = self.vertical_resolution { op.set_double("yres", v); }
        if let Some(v) = self.offset_x { op.set_int("xoffset", v); }
        if let Some(v) = self.offset_y { op.set_int("yoffset", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── TileCache ─────────────────────────────────────────────────────────────────

pub struct TileCache<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub tile_width: Option<i32>,
    pub tile_height: Option<i32>,
    pub maximum_tiles: Option<i32>,
    pub access: Option<Access>,
    pub threaded: Option<bool>,
    pub persistent: Option<bool>,
}

impl<B: Backend> Operation<B> for TileCache<B> where TileCache<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.tile_width { state.write_i32(v); }
        if let Some(v) = self.tile_height { state.write_i32(v); }
        if let Some(v) = self.maximum_tiles { state.write_i32(v); }
        if let Some(v) = self.access { state.write_i32(v.into_vips()); }
        if let Some(v) = self.threaded { state.write_u8(v as u8); }
        if let Some(v) = self.persistent { state.write_u8(v as u8); }
    }
}

impl Lower<VipsBackend> for TileCache<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"tilecache\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.tile_width { op.set_int("tile_width", v); }
        if let Some(v) = self.tile_height { op.set_int("tile_height", v); }
        if let Some(v) = self.maximum_tiles { op.set_int("max_tiles", v); }
        if let Some(v) = self.access { op.set_int("access", v.into_vips()); }
        if let Some(v) = self.threaded { op.set_bool("threaded", v); }
        if let Some(v) = self.persistent { op.set_bool("persistent", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Basic unary ops ───────────────────────────────────────────────────────────

macro_rules! define_unary_op {
    ($name:ident, $vips_name:expr) => {
        pub struct $name<B: Backend> {
            pub input: Input<ImageKind, B>,
        }
        impl<B: Backend> Operation<B> for $name<B> where $name<B>: Lower<B> {
            type Output = ImageKind;
            fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
            fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
            fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
            fn dyn_hash(&self, _state: &mut dyn Hasher) {}
        }
        impl Lower<VipsBackend> for $name<VipsBackend> {
            fn lower(&self, cx: &mut VipsBuilder) {
                let input_handle = cx.input(self.input.src());
                let mut op = crate::backend::vips::gobject::VipsGObject::new($vips_name).unwrap();
                op.set_image("in", input_handle.ptr);
                let out_handle = op.run().unwrap();
                cx.emit(out_handle);
            }
        }
    };
}

define_unary_op!(Clamp, b"clamp\0");
define_unary_op!(ScaleImage, b"scale\0");
define_unary_op!(Wrap, b"wrap\0");
define_unary_op!(Sequential, b"sequential\0");
define_unary_op!(Autorotate, b"autorot\0");
define_unary_op!(Byteswap, b"byteswap\0");
define_unary_op!(Transpose3d, b"transpose3d\0");
define_unary_op!(Falsecolour, b"falsecolour\0");
define_unary_op!(MatrixInvert, b"matrixinvert\0");
define_unary_op!(Rad2float, b"rad2float\0");
define_unary_op!(Float2rad, b"float2rad\0");

// ── Msb ───────────────────────────────────────────────────────────────────────

pub struct Msb<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub band: Option<i32>,
}

impl<B: Backend> Operation<B> for Msb<B> where Msb<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.band { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for Msb<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"msb\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.band { op.set_int("band", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Maplut ────────────────────────────────────────────────────────────────────

pub struct Maplut<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub lut: Input<ImageKind, B>,
    pub band: Option<i32>,
}

impl<B: Backend> Operation<B> for Maplut<B> where Maplut<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.lut] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.lut.spec.width,
                h: self.lut.spec.height,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.band { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for Maplut<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let lut_handle = cx.input(self.lut.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"maplut\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("lut", lut_handle.ptr);
        if let Some(v) = self.band { op.set_int("band", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Recomb ────────────────────────────────────────────────────────────────────

pub struct Recomb<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub matrix: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Recomb<B> where Recomb<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.matrix] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.matrix.spec.width,
                h: self.matrix.spec.height,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Recomb<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let matrix_handle = cx.input(self.matrix.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"recomb\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("m", matrix_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Ifthenelse ────────────────────────────────────────────────────────────────

pub struct Ifthenelse<B: Backend> {
    pub cond: Input<ImageKind, B>,
    pub if_true: Input<ImageKind, B>,
    pub if_false: Input<ImageKind, B>,
    pub blend: Option<bool>,
}

impl<B: Backend> Operation<B> for Ifthenelse<B> where Ifthenelse<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.cond, &self.if_true, &self.if_false] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Region(out.clone())),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.if_true.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.blend { state.write_u8(v as u8); }
    }
}

impl Lower<VipsBackend> for Ifthenelse<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let cond_handle = cx.input(self.cond.src());
        let true_handle = cx.input(self.if_true.src());
        let false_handle = cx.input(self.if_false.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"ifthenelse\0").unwrap();
        op.set_image("cond", cond_handle.ptr);
        op.set_image("in1", true_handle.ptr);
        op.set_image("in2", false_handle.ptr);
        if let Some(v) = self.blend { op.set_bool("blend", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Invertlut ─────────────────────────────────────────────────────────────────

pub struct Invertlut<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub size: Option<i32>,
}

impl<B: Backend> Operation<B> for Invertlut<B> where Invertlut<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.size { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for Invertlut<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"invertlut\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.size { op.set_int("size", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Linecache ─────────────────────────────────────────────────────────────────

pub struct Linecache<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub tile_height: Option<i32>,
    pub access: Option<Access>,
    pub threaded: Option<bool>,
    pub persistent: Option<bool>,
}

impl<B: Backend> Operation<B> for Linecache<B> where Linecache<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.tile_height { state.write_i32(v); }
        if let Some(v) = self.access { state.write_i32(v.into_vips()); }
        if let Some(v) = self.threaded { state.write_u8(v as u8); }
        if let Some(v) = self.persistent { state.write_u8(v as u8); }
    }
}

impl Lower<VipsBackend> for Linecache<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"linecache\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.tile_height { op.set_int("tile_height", v); }
        if let Some(v) = self.access { op.set_int("access", v.into_vips()); }
        if let Some(v) = self.threaded { op.set_bool("threaded", v); }
        if let Some(v) = self.persistent { op.set_bool("persistent", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Case ──────────────────────────────────────────────────────────────────────

pub struct Case<B: Backend> {
    pub input: Input<ImageKind, B>, // used as `index`
    pub cases: Vec<Input<ImageKind, B>>,
}

impl<B: Backend> Operation<B> for Case<B> where Case<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        let mut res: Vec<&dyn AnyInput<B>> = vec![&self.input];
        for c in &self.cases {
            res.push(c as &dyn AnyInput<B>);
        }
        res
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mut res = vec![Some(WorkUnit::Region(out.clone()))];
        for _ in 0..self.cases.len() {
            res.push(Some(WorkUnit::Region(out.clone())));
        }
        res
    }
    fn output_spec(&self) -> ImageKind { (*self.cases[0].spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Case<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let index_handle = cx.input(self.input.src());
        let mut case_ptrs: Vec<*mut crate::ffi::VipsImage> = vec![];
        for c in &self.cases {
            let handle = cx.input(c.src());
            case_ptrs.push(handle.ptr);
        }
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"case\0").unwrap();
        op.set_image("index", index_handle.ptr);
        op.set_array_image("cases", &case_ptrs);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Exposure / Brightness / NoiseReduction ────────────────────────────────────

pub struct Exposure<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub stops: f32,
    pub preserve: f32,
}

impl<B: Backend> Operation<B> for Exposure<B> where Exposure<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.stops.to_bits());
        state.write_u32(self.preserve.to_bits());
    }
}

impl Lower<VipsBackend> for Exposure<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let gain = 2.0f64.powf(self.stops as f64);
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"linear\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_array_double("a", &[gain]);
        op.set_array_double("b", &[0.0]);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct Brightness<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub value: f32,
}

impl<B: Backend> Operation<B> for Brightness<B> where Brightness<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.value.to_bits());
    }
}

impl Lower<VipsBackend> for Brightness<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"linear\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_array_double("a", &[self.value as f64]);
        op.set_array_double("b", &[0.0]);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

pub struct NoiseReduction<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub strength: f32,
}

impl<B: Backend> Operation<B> for NoiseReduction<B> where NoiseReduction<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.strength.to_bits());
    }
}

impl Lower<VipsBackend> for NoiseReduction<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let size = (1 + (self.strength * 4.0) as i32 * 2).max(1);
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"median\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("size", size);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Saturation ────────────────────────────────────────────────────────────────

pub struct Saturation<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub amount: f32,
}

impl<B: Backend> Operation<B> for Saturation<B> where Saturation<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.amount.to_bits());
    }
}

impl Lower<VipsBackend> for Saturation<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let ptr = input_handle.ptr;
        let bands = unsafe { crate::ffi::vips_image_get_bands(ptr) };
        if bands < 3 {
            // Grayscale: saturation has no effect
            unsafe { crate::ffi::g_object_ref(ptr as *mut _) };
            cx.emit(crate::backend::vips::VipsHandle { ptr });
            return;
        }

        let ext = |b| {
            let mut o = crate::backend::vips::gobject::VipsGObject::new(b"extract_band\0").unwrap();
            o.set_image("in", ptr);
            o.set_int("band", b);
            o.run().unwrap().ptr
        };
        let r = ext(0);
        let g = ext(1);
        let b = ext(2);

        let mul = |p, w: f64| {
            let mut o = crate::backend::vips::gobject::VipsGObject::new(b"linear\0").unwrap();
            o.set_image("in", p);
            o.set_array_double("a", &[w]);
            o.set_array_double("b", &[0.0]);
            o.run().unwrap().ptr
        };
        let luma_r = mul(r, 0.2126);
        let luma_g = mul(g, 0.7152);
        let luma_b = mul(b, 0.0722);

        let add = |p1, p2| {
            let mut o = crate::backend::vips::gobject::VipsGObject::new(b"add\0").unwrap();
            o.set_image("left", p1);
            o.set_image("right", p2);
            o.run().unwrap().ptr
        };
        let luma1 = add(luma_r, luma_g);
        let luma = add(luma1, luma_b);

        let mut op_rgb = crate::backend::vips::gobject::VipsGObject::new(b"extract_band\0").unwrap();
        op_rgb.set_image("in", ptr);
        op_rgb.set_int("band", 0);
        op_rgb.set_int("n", 3);
        let rgb_ptr = op_rgb.run().unwrap().ptr;

        let rgb_scaled = mul(rgb_ptr, self.amount as f64);
        let luma_scaled = mul(luma, 1.0 - self.amount as f64);

        let out_rgb = add(rgb_scaled, luma_scaled);

        let out_ptr = if bands > 3 {
            let mut op_a = crate::backend::vips::gobject::VipsGObject::new(b"extract_band\0").unwrap();
            op_a.set_image("in", ptr);
            op_a.set_int("band", 3);
            op_a.set_int("n", bands - 3);
            let a_ptr = op_a.run().unwrap().ptr;

            let mut out: *mut crate::ffi::VipsImage = std::ptr::null_mut();
            let arr = [out_rgb, a_ptr];
            let ret = unsafe { crate::ffi::vips_bandjoin(arr.as_ptr() as *mut *mut _, &mut out, 2, crate::backend::vips::null()) };
            if ret != 0 { panic!("vips_bandjoin failed"); }
            out
        } else {
            out_rgb
        };
        cx.emit(crate::backend::vips::VipsHandle { ptr: out_ptr });
    }
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────

impl Lower<GpuBackend> for Saturation<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("amount", "float", self.amount));
        cx.kernel("saturation_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Cast<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        // Cast is just a codec change in output_spec.
        cx.kernel("passthrough_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Msb<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        // Warning: MSB is currently implemented as an 8-bit scale extraction.
        cx.param_block(ParamBlock::new().param("band", "int", self.band.unwrap_or(-1)));
        cx.kernel("msb_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Exposure<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let gain = 2.0f32.powf(self.stops);
        cx.param_block(ParamBlock::new()
            .param("gain", "float", gain)
            .param("preserve", "float", self.preserve)
        );
        cx.kernel("exposure_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Brightness<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new()
            .param("gain", "float", self.value)
            .param("preserve", "float", 0.0f32)
        );
        cx.kernel("exposure_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Maplut<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new()
            .param("lut_width", "uint", self.lut.spec.width as u32)
            .param("band", "int", self.band.unwrap_or(-1))
        );
        cx.kernel("maplut_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Recomb<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new()
            .param("n", "uint", self.input.spec.format.channel_count() as u32)
        );
        cx.kernel("recomb_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Ifthenelse<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new()
            .param("blend", "uint", self.blend.unwrap_or(false) as u32)
        );
        cx.kernel("ifthenelse_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Case<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let n = self.cases.len();
        match n {
            0 => cx.kernel("passthrough_kernel"), // Fallback for 0 cases
            1 => cx.kernel("case1_kernel"),
            2 => cx.kernel("case2_kernel"),
            3 => cx.kernel("case3_kernel"),
            4 => cx.kernel("case4_kernel"),
            _ => cx.kernel("case5_kernel"), // Hard cap fallback
        };
        cx.output(self.output_spec().output());
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Cast<B>: crate::operation::Lower<B>,
{
    pub fn cast(&self, format: PixelFormat, shift: Option<bool>) -> Self {
        self.push(Cast { input: self.as_input(), format, shift })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Copy<B>: crate::operation::Lower<B>,
{
    pub fn copy(&self, width: Option<i32>, height: Option<i32>, bands: Option<i32>, format: Option<i32>, interpretation: Option<i32>, horizontal_resolution: Option<f64>, vertical_resolution: Option<f64>, offset_x: Option<i32>, offset_y: Option<i32>) -> Self {
        self.push(Copy { input: self.as_input(), width, height, bands, format, interpretation, horizontal_resolution, vertical_resolution, offset_x, offset_y })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    TileCache<B>: crate::operation::Lower<B>,
{
    pub fn tile_cache(&self, tile_width: Option<i32>, tile_height: Option<i32>, maximum_tiles: Option<i32>, access: Option<Access>, threaded: Option<bool>, persistent: Option<bool>) -> Self {
        self.push(TileCache { input: self.as_input(), tile_width, tile_height, maximum_tiles, access, threaded, persistent })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Msb<B>: crate::operation::Lower<B>,
{
    pub fn msb(&self, band: Option<i32>) -> Self {
        self.push(Msb { input: self.as_input(), band })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Maplut<B>: crate::operation::Lower<B>,
{
    pub fn maplut(&self, lut: Input<ImageKind, B>, band: Option<i32>) -> Self {
        self.push(Maplut { input: self.as_input(), lut, band })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Recomb<B>: crate::operation::Lower<B>,
{
    pub fn recomb(&self, matrix: Input<ImageKind, B>) -> Self {
        self.push(Recomb { input: self.as_input(), matrix })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Invertlut<B>: crate::operation::Lower<B>,
{
    pub fn invertlut(&self, size: Option<i32>) -> Self {
        self.push(Invertlut { input: self.as_input(), size })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Linecache<B>: crate::operation::Lower<B>,
{
    pub fn linecache(&self, tile_height: Option<i32>, access: Option<Access>, threaded: Option<bool>, persistent: Option<bool>) -> Self {
        self.push(Linecache { input: self.as_input(), tile_height, access, threaded, persistent })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Case<B>: crate::operation::Lower<B>,
{
    pub fn case(&self, cases: Vec<Input<ImageKind, B>>) -> Self {
        self.push(Case { input: self.as_input(), cases })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Ifthenelse<B>: crate::operation::Lower<B>,
{
    pub fn ifthenelse(&self, if_true: Input<ImageKind, B>, if_false: Input<ImageKind, B>, blend: Option<bool>) -> Self {
        self.push(Ifthenelse { cond: self.as_input(), if_true, if_false, blend })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Exposure<B>: crate::operation::Lower<B>,
{
    pub fn exposure(&self, stops: f32, preserve: f32) -> Self {
        self.push(Exposure { input: self.as_input(), stops, preserve })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Brightness<B>: crate::operation::Lower<B>,
{
    pub fn brightness(&self, value: f32) -> Self {
        self.push(Brightness { input: self.as_input(), value })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    NoiseReduction<B>: crate::operation::Lower<B>,
{
    pub fn noise_reduction(&self, strength: f32) -> Self {
        self.push(NoiseReduction { input: self.as_input(), strength })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Saturation<B>: crate::operation::Lower<B>,
{
    pub fn saturation(&self, amount: f32) -> Self {
        self.push(Saturation { input: self.as_input(), amount })
    }
}
