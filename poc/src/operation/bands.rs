use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{IntoVipsEnum, VipsBackend, VipsBuilder};
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::gpu::view::ParamBlock;
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation, OperationBoolean};
use crate::work_unit::{Region, WorkUnit};

// ── Boolean ───────────────────────────────────────────────────────────────────

pub struct Bandbool<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub boolean: OperationBoolean,
    pub bands: u32,
}

impl<B: Backend> Operation<B> for Bandbool<B> where Bandbool<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.boolean.into_vips());
        state.write_u32(self.bands);
    }
}

impl Lower<VipsBackend> for Bandbool<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"bandbool\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("boolean", self.boolean.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Bandfold ──────────────────────────────────────────────────────────────────

pub struct Bandfold<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub factor: u32,
}

impl<B: Backend> Operation<B> for Bandfold<B> where Bandfold<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let f = self.factor as i32;
        let y = out.y / f;
        let h = ((out.y + out.h + f - 1) / f) - y;
        vec![Some(WorkUnit::Region(Region {
            x: out.x,
            y,
            w: out.w,
            h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.height *= self.factor as i32;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.factor);
    }
}

impl Lower<VipsBackend> for Bandfold<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"bandfold\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("factor", self.factor as i32);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Bandunfold ────────────────────────────────────────────────────────────────

pub struct Bandunfold<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub factor: u32,
}

impl<B: Backend> Operation<B> for Bandunfold<B> where Bandunfold<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let f = self.factor as i32;
        vec![Some(WorkUnit::Region(Region {
            x: out.x,
            y: out.y * f,
            w: out.w,
            h: out.h * f,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.height /= self.factor as i32;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.factor);
    }
}

impl Lower<VipsBackend> for Bandunfold<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"bandunfold\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("factor", self.factor as i32);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Bandmean ──────────────────────────────────────────────────────────────────

pub struct Bandmean<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub bands: u32,
}

impl<B: Backend> Operation<B> for Bandmean<B> where Bandmean<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.bands);
    }
}

impl Lower<VipsBackend> for Bandmean<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"bandmean\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── ExtractBand ───────────────────────────────────────────────────────────────

pub struct ExtractBand<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub band: i32,
    pub count: Option<i32>,
}

impl<B: Backend> Operation<B> for ExtractBand<B> where ExtractBand<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.band);
        if let Some(c) = self.count {
            state.write_i32(c);
        }
    }
}

impl Lower<VipsBackend> for ExtractBand<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"extract_band\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("band", self.band);
        if let Some(c) = self.count {
            op.set_int("n", c);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Bandjoin ──────────────────────────────────────────────────────────────────

pub struct Bandjoin<B: Backend> {
    pub images: Vec<Input<ImageKind, B>>,
}

impl<B: Backend> Operation<B> for Bandjoin<B> where Bandjoin<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        self.images.iter().map(|i| i as &dyn AnyInput<B>).collect()
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); self.images.len()]
    }
    fn output_spec(&self) -> ImageKind { (*self.images[0].spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Bandjoin<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let mut ptrs: Vec<*mut crate::ffi::VipsImage> = vec![];
        for input in &self.images {
            let handle = cx.input(input.src());
            ptrs.push(handle.ptr);
        }
        let mut out: *mut crate::ffi::VipsImage = std::ptr::null_mut();
        let ret = unsafe {
            crate::ffi::vips_bandjoin(
                ptrs.as_mut_ptr(),
                &mut out,
                ptrs.len() as i32,
                std::ptr::null::<std::ffi::c_void>(),
            )
        };
        if ret != 0 {
            panic!("vips_bandjoin failed");
        }
        let out_handle = crate::backend::vips::VipsHandle { ptr: out };
        cx.emit(out_handle);
    }
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────

impl Lower<GpuBackend> for Bandbool<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("boolean", "uint", self.boolean.into_vips() as u32));
        cx.kernel("bandbool_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Bandfold<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("factor", "uint", self.factor));
        cx.kernel("bandfold_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Bandunfold<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("factor", "uint", self.factor));
        cx.kernel("bandunfold_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Bandmean<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("bandmean_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for ExtractBand<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("band", "int", self.band));
        cx.kernel("extract_band_kernel");
        cx.output(self.output_spec().output());
    }
}

impl Lower<GpuBackend> for Bandjoin<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let kernel = match self.images.len() {
            1 => "bandjoin1_kernel",
            2 => "bandjoin2_kernel",
            3 => "bandjoin3_kernel",
            4 => "bandjoin4_kernel",
            5 => "bandjoin5_kernel",
            _ => panic!("Bandjoin: unsupported number of inputs (max 5)"),
        };
        cx.kernel(kernel);
        cx.output(self.output_spec().output());
    }
}
