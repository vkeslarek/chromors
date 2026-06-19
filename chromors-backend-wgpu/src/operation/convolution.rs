use crate::prelude::*;

impl Lower<GpuBackend> for crate::Compass<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("mw", self.mask.spec.width as u32)
                .param("mh", self.mask.spec.height as u32),
        );
        cx.kernel("ops.convolution", "compass_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Convolution<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("mw", self.mask.spec.width as u32)
                .param("mh", self.mask.spec.height as u32),
        );
        cx.kernel("ops.convolution", "convolution_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Morph<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        // vips_morph casts to uchar via vips_cast(..., shift=false), i.e.
        // CLIP(0, raw_value, 255) on the *raw* sample value -- not a rescale.
        // Our working float is normalised to [0,1] by the codec, so recover
        // the raw value via `component_max_f64` before clamping to a byte.
        let src_max = self.input.spec.layout.component_max_f64() as f32;
        // Field order must match morph_kernel's parameter order exactly --
        // kernel args are bound positionally from this block.
        cx.param_block(
            ParamBlock::new()
                .param("morph", self.morph.into_vips() as u32)
                .param("mw", self.mask.spec.width as u32)
                .param("mh", self.mask.spec.height as u32)
                .param("src_max", src_max),
        );
        cx.kernel("ops.convolution", "morph_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Conva<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("mw", self.mask.spec.width as u32)
                .param("mh", self.mask.spec.height as u32),
        );
        cx.kernel("ops.convolution", "convolution_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Convf<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("mw", self.mask.spec.width as u32)
                .param("mh", self.mask.spec.height as u32),
        );
        cx.kernel("ops.convolution", "convolution_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Convi<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("mw", self.mask.spec.width as u32)
                .param("mh", self.mask.spec.height as u32),
        );
        cx.kernel("ops.convolution", "convolution_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Convsep<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("mw", self.mask.spec.width as u32)
                .param("mh", self.mask.spec.height as u32),
        );
        cx.kernel("ops.convolution", "convolution_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Convasep<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("mw", self.mask.spec.width as u32)
                .param("mh", self.mask.spec.height as u32),
        );
        cx.kernel("ops.convolution", "convolution_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}
