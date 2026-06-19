use crate::{GpuBackend, GpuBuffer, GpuBuilder, GpuContext};
use chromors_core::buffer::Buffer;
use chromors_core::data::image::ImageKind;
use chromors_core::error::Error;
use chromors_core::generator::{Constant, GenSource, Generator};
use chromors_core::io::Source;
use chromors_core::work_unit::{Region, WorkUnitFor};
use std::sync::Arc;

/// The GPU counterpart of `Generator`.
pub trait GpuGenerator: Generator {
    /// Returns the Slang module name and kernel entry point function name.
    /// The signature MUST match the `K1` shape:
    /// `kernel(idx, output, gen_ox, gen_oy, gen_lod, gen_fw, gen_fh, <own params>)`
    fn gpu_kernel(&self) -> (&'static str, &'static str);

    /// Appends the generator's specific configuration to the current node's
    /// `ParamBlock` via `cx.param(...)`. MUST be pushed in the exact layout
    /// order expected by the kernel after the 5 canonical fields.
    fn gpu_params(&self, cx: &mut GpuBuilder);
}

impl<G: GpuGenerator> Source<GpuBackend> for GenSource<G> {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        self.0.spec()
    }

    fn fetch(&self, _ctx: &GpuContext, _wu: &Region) -> Result<Buffer<GpuBackend>, Error> {
        Err(Error::Backend(
            "generator: use lower(), not fetch(), on GPU".into(),
        ))
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let Some(r) = Region::typed(&wu) else {
            cx.fail(Error::InvalidWorkUnit("generator expects a Region".into()));
            return;
        };

        let (module, entry) = self.0.gpu_kernel();

        // 1. Start the generator step
        cx.kernel(module, entry);

        // 2. Push canonical coordinates (the first 5 parameters in Slang)
        // Note: gen_ox, gen_oy, gen_lod, gen_fw, gen_fh are the K1 parameters.
        cx.param("gen_ox", r.x);
        cx.param("gen_oy", r.y);
        cx.param("gen_lod", r.lod.0 as i32);

        let spec = self.0.spec();
        cx.param("gen_fw", spec.width as i32);
        cx.param("gen_fh", spec.height as i32);

        // 3. Push generator specific parameters
        self.0.gpu_params(cx);

        // 4. Conclude with output wrap for ImageKind
        use crate::gpu::GpuView;
        cx.output(spec.output(&wu));
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        self.0.dyn_hash(state);
    }
}

impl GpuGenerator for Constant {
    fn gpu_kernel(&self) -> (&'static str, &'static str) {
        ("ops.generators", "constant_kernel")
    }

    fn gpu_params(&self, cx: &mut GpuBuilder) {
        cx.param("color_r", self.color[0]);
        cx.param("color_g", self.color[1]);
        cx.param("color_b", self.color[2]);
        cx.param("color_a", self.color[3]);
    }
}

use chromors_core::generator::{GaussNoise, LinearGradient, Xyz};

impl GpuGenerator for LinearGradient {
    fn gpu_kernel(&self) -> (&'static str, &'static str) {
        ("ops.generators", "linear_gradient_kernel")
    }

    fn gpu_params(&self, cx: &mut GpuBuilder) {
        cx.param("c0_r", self.c0[0]);
        cx.param("c0_g", self.c0[1]);
        cx.param("c0_b", self.c0[2]);
        cx.param("c0_a", self.c0[3]);
        cx.param("c1_r", self.c1[0]);
        cx.param("c1_g", self.c1[1]);
        cx.param("c1_b", self.c1[2]);
        cx.param("c1_a", self.c1[3]);
        cx.param("angle", self.angle);
    }
}

impl GpuGenerator for Xyz {
    fn gpu_kernel(&self) -> (&'static str, &'static str) {
        ("ops.generators", "xyz_kernel")
    }

    fn gpu_params(&self, _cx: &mut GpuBuilder) {
        // No extra params
    }
}

impl GpuGenerator for GaussNoise {
    fn gpu_kernel(&self) -> (&'static str, &'static str) {
        ("ops.generators", "gaussnoise_kernel")
    }

    fn gpu_params(&self, cx: &mut GpuBuilder) {
        cx.param("mean", self.mean);
        cx.param("sigma", self.sigma);
        cx.param("seed", self.seed);
    }
}
