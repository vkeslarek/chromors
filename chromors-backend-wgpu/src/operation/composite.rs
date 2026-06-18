use crate::prelude::*;

impl Lower<GpuBackend> for crate::Composite2<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("mode", self.mode.into_vips() as u32)
                .param("x", self.x.unwrap_or(0))
                .param("y", self.y.unwrap_or(0)),
        );
        cx.kernel("ops.composite", "compose_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Join<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("direction", self.direction.into_vips())
                .param("shim", self.shim.unwrap_or(0))
                .param("align", self.align.map(|a| a.into_vips()).unwrap_or(0)),
        );
        cx.kernel("ops.composite", "join_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Insert<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("x", self.x).param("y", self.y));
        cx.kernel("ops.composite", "insert_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

