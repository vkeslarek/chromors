use crate::prelude::*;

impl Lower<GpuBackend> for crate::Bandbool<GpuBackend> {
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

impl Lower<GpuBackend> for crate::Bandfold<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let in_bands = self.input.spec.layout.channel_count() as u32;
        let out_bands = self.output_spec().layout.channel_count() as u32;
        // Field order must match bandfold_kernel's parameter order exactly --
        // kernel args are bound positionally from this block.
        cx.param_block(
            ParamBlock::new()
                .param("factor", self.factor)
                .param("in_bands", in_bands)
                .param("out_bands", out_bands),
        );
        cx.kernel("ops.bands", "bandfold_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Bandunfold<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let in_bands = self.input.spec.layout.channel_count() as u32;
        let out_bands = self.output_spec().layout.channel_count() as u32;
        // Field order must match bandunfold_kernel's parameter order exactly --
        // kernel args are bound positionally from this block.
        cx.param_block(
            ParamBlock::new()
                .param("factor", self.factor)
                .param("in_bands", in_bands)
                .param("out_bands", out_bands),
        );
        cx.kernel("ops.bands", "bandunfold_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Bandmean<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("bands", self.bands));
        cx.kernel("ops.bands", "bandmean_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::ExtractBand<GpuBackend> {
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

impl Lower<GpuBackend> for crate::Bandjoin<GpuBackend> {
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

pub fn swizzle_adapter(channel: u32) -> ViewAdapter {
    ViewAdapter {
        wrapper: "SwizzleView<{inner}>".into(),
        ctor: "{ {value}, {params}[0].{p}_channel }".into(),
        params: ParamBlock::scalar("{p}_channel", channel),
        module: "lib.region",
    }
}
