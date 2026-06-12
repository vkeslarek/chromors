use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{IntoVipsEnum, VipsBackend, VipsBuilder};
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::gpu::view::{ParamBlock, ViewAdapter};
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
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.format = spec.with_band_count(1);
        spec
    }
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
        vec![Some(WorkUnit::Region(Region {
            x: out.x * f,
            y: out.y,
            w: out.w * f,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        let bands = spec.format.channel_count() as i32;
        spec.width /= self.factor as i32;
        spec.format = spec.with_band_count(bands * self.factor as i32);
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
        let x = out.x / f;
        let w = ((out.x + out.w + f - 1) / f) - x;
        vec![Some(WorkUnit::Region(Region {
            x,
            y: out.y,
            w,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        let bands = spec.format.channel_count() as i32;
        spec.width *= self.factor as i32;
        spec.format = spec.with_band_count(bands / self.factor as i32);
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
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.format = spec.with_band_count(1);
        spec
    }
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

/// Wraps a node's value in `SwizzleView<{inner}>` — reads through one
/// component (0=x/r .. 3=w/a), broadcast `float4(v,v,v,1)`. Used by
/// `ExtractBand` for a single-band extract: zero-cost, no kernel pass.
pub fn swizzle_adapter(channel: u32) -> ViewAdapter {
    ViewAdapter {
        wrapper: "SwizzleView<{inner}>".into(),
        ctor: "{ {value}, {params}[0].{p}_channel }".into(),
        params: ParamBlock::scalar("{p}_channel", channel),
        module: "lib.region",
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
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.format = spec.with_band_count(self.count.unwrap_or(1));
        spec
    }
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
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.images[0].spec).clone();
        spec.format = spec.with_band_count(self.images.len() as i32);
        spec
    }
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
        cx.param_block(
            ParamBlock::new()
                .param("boolean", self.boolean.into_vips() as u32)
                .param("bands", self.bands),
        );
        cx.kernel("ops.bands", "bandbool_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for Bandfold<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("factor", self.factor));
        cx.kernel("ops.bands", "bandfold_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for Bandunfold<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("factor", self.factor));
        cx.kernel("ops.bands", "bandunfold_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for Bandmean<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("bands", self.bands));
        cx.kernel("ops.bands", "bandmean_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for ExtractBand<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        match self.count {
            // Single-band extract is free: alias the input through the
            // selected component instead of adding a kernel pass.
            None | Some(1) => {
                cx.adapt(swizzle_adapter(self.band as u32));
            }
            Some(count) => {
                cx.param_block(
                    ParamBlock::new()
                        .param("band", self.band as u32)
                        .param("count", count as u32),
                );
                cx.kernel("ops.bands", "extract_band_range_kernel");
            }
        }
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for Bandjoin<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let n = self.images.len();
        let kernel = match n {
            1 => "bandjoin1_kernel",
            2 => "bandjoin2_kernel",
            3 => "bandjoin3_kernel",
            4 => "bandjoin4_kernel",
            5 => "bandjoin5_kernel",
            _ => panic!("Bandjoin: unsupported number of inputs (max 5)"),
        };
        // Each input is itself a single-band image (its working temp
        // broadcasts r=g=b=value), so every source contributes channel 0.
        let mut params = ParamBlock::new();
        for i in 0..n {
            params = params.param(&format!("ch{i}"), 0u32);
        }
        cx.param_block(params);
        cx.kernel("ops.bands", kernel);
        cx.output(self.output_spec().output(cx.wu()));
    }
}


impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Bandbool<B>: crate::operation::Lower<B>,
{
    pub fn bandbool(&self, boolean: OperationBoolean, bands: u32) -> Self {
        self.push(Bandbool { input: self.as_input(), boolean, bands })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Bandfold<B>: crate::operation::Lower<B>,
{
    pub fn bandfold(&self, factor: u32) -> Self {
        self.push(Bandfold { input: self.as_input(), factor })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Bandunfold<B>: crate::operation::Lower<B>,
{
    pub fn bandunfold(&self, factor: u32) -> Self {
        self.push(Bandunfold { input: self.as_input(), factor })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Bandmean<B>: crate::operation::Lower<B>,
{
    pub fn bandmean(&self, bands: u32) -> Self {
        self.push(Bandmean { input: self.as_input(), bands })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    ExtractBand<B>: crate::operation::Lower<B>,
{
    pub fn extract_band(&self, band: i32, count: Option<i32>) -> Self {
        self.push(ExtractBand { input: self.as_input(), band, count })
    }
}
