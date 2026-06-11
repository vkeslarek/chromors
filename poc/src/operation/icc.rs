use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{VipsBackend, VipsBuilder};
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::gpu::view::ParamBlock;
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

// ── IccImport ─────────────────────────────────────────────────────────────────

pub struct IccImport<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub embedded: Option<bool>,
    pub input_profile: Option<String>,
    pub intent: Option<i32>,
}

impl<B: Backend> Operation<B> for IccImport<B> where IccImport<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.embedded { state.write_u8(v as u8); }
        if let Some(ref v) = self.input_profile { state.write(v.as_bytes()); }
        if let Some(v) = self.intent { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for IccImport<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"icc_import\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.embedded {
            op.set_bool("embedded", v);
        }
        if let Some(ref v) = self.input_profile {
            op.set_string("input_profile", v);
        }
        if let Some(v) = self.intent {
            op.set_int("intent", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── IccExport ─────────────────────────────────────────────────────────────────

pub struct IccExport<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub output_profile: Option<String>,
    pub intent: Option<i32>,
    pub depth: Option<i32>,
}

impl<B: Backend> Operation<B> for IccExport<B> where IccExport<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(ref v) = self.output_profile { state.write(v.as_bytes()); }
        if let Some(v) = self.intent { state.write_i32(v); }
        if let Some(v) = self.depth { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for IccExport<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"icc_export\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(ref v) = self.output_profile {
            op.set_string("output_profile", v);
        }
        if let Some(v) = self.intent {
            op.set_int("intent", v);
        }
        if let Some(v) = self.depth {
            op.set_int("depth", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── IccTransform ──────────────────────────────────────────────────────────────

pub struct IccTransform<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub output_profile: String,
    pub embedded: Option<bool>,
    pub input_profile: Option<String>,
    pub intent: Option<i32>,
    pub depth: Option<i32>,
}

impl<B: Backend> Operation<B> for IccTransform<B> where IccTransform<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write(self.output_profile.as_bytes());
        if let Some(v) = self.embedded { state.write_u8(v as u8); }
        if let Some(ref v) = self.input_profile { state.write(v.as_bytes()); }
        if let Some(v) = self.intent { state.write_i32(v); }
        if let Some(v) = self.depth { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for IccTransform<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"icc_transform\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_string("output_profile", &self.output_profile);
        if let Some(v) = self.embedded {
            op.set_bool("embedded", v);
        }
        if let Some(ref v) = self.input_profile {
            op.set_string("input_profile", v);
        }
        if let Some(v) = self.intent {
            op.set_int("intent", v);
        }
        if let Some(v) = self.depth {
            op.set_int("depth", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Gamma ─────────────────────────────────────────────────────────────────────

pub struct Gamma<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub exponent: Option<f64>,
}

impl<B: Backend> Operation<B> for Gamma<B> where Gamma<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.exponent { state.write(&v.to_ne_bytes()); }
    }
}

impl Lower<VipsBackend> for Gamma<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"gamma\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.exponent {
            op.set_double("exponent", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────

impl Lower<GpuBackend> for Gamma<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new()
            .param("exponent", "float", self.exponent.unwrap_or(1.0) as f32)
        );
        cx.kernel("gamma_kernel");
        cx.output(self.output_spec().output());
    }
}
