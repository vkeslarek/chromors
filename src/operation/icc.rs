//! Gamma (TRC exponent) operation.
//!
//! ICC color management is **not** done here. Profile classification happens at
//! load (`FileImageSource::new` → `IccClassification::classify_icc_profile`,
//! which tags the image's `PixelLayout::color_space`), and all color-space
//! conversion is the native `Convert` operation (`operation::color`,
//! `docs/native-color-management.md`) applied as a real Slang/CPU op — never a
//! libvips `icc_import`/`icc_export`/`icc_transform` wrapper. Those vips ICC ops
//! were removed: every backend now obeys the native color pipeline.

use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::gpu::view::ParamBlock;
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::vips::{VipsBackend, VipsBuilder};
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

// ── Gamma ─────────────────────────────────────────────────────────────────────

pub struct Gamma<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub exponent: Option<f64>,
}

impl<B: Backend> Operation<B> for Gamma<B>
where
    Gamma<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.exponent {
            state.write(&v.to_ne_bytes());
        }
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

impl Lower<GpuBackend> for Gamma<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("exponent", self.exponent.unwrap_or(1.0) as f32));
        cx.kernel("ops.icc", "gamma_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Gamma<B>: crate::operation::Lower<B>,
{
    pub fn gamma(&self, exponent: Option<f64>) -> Self {
        self.push(Gamma {
            input: self.as_input(),
            exponent,
        })
    }
}
