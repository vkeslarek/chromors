use crate::prelude::*;

impl Lower<GpuBackend> for crate::Gamma<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("exponent", self.exponent.unwrap_or(1.0) as f32));
        cx.kernel("ops.icc", "gamma_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}
