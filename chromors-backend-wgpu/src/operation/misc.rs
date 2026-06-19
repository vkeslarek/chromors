use crate::prelude::*;

impl Lower<GpuBackend> for crate::Copy<GpuBackend> {
    /// `copy` is a pixel-identity passthrough: it only rewrites metadata
    /// (resolution, offset, declared extent). On the GPU it is therefore a
    /// zero-cost alias of its input — `forward()` adds no kernel step; the
    /// value flows straight through to the consumer (or the encoder, if this is
    /// the root), re-encoded under the output spec. It fuses away entirely.
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.forward();
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Invertlut<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let size = self.size.unwrap_or(256) as u32;
        let height = self.input.spec.entries;
        let bands = self.input.spec.bands.saturating_sub(1);
        cx.dispatch((size, 1));
        cx.param_block(
            ParamBlock::new()
                .param("height", height)
                .param("size", size)
                .param("bands", bands),
        );
        cx.kernel("ops.misc", "invertlut_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::NoiseReduction<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("sz", self.size() as u32));
        cx.kernel("ops.filters", "median_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Saturation<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("amount", self.amount));
        cx.kernel("ops.misc", "saturation_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Cast<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        // Cast is just a codec change in output_spec.
        cx.kernel("ops.misc", "passthrough_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Msb<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        // Warning: MSB is currently implemented as an 8-bit scale extraction.
        cx.param_block(ParamBlock::new().param("band", self.band.unwrap_or(-1)));
        cx.kernel("ops.misc", "msb_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Exposure<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let gain = 2.0f32.powf(self.stops);
        cx.param_block(
            ParamBlock::new()
                .param("gain", gain)
                .param("preserve", self.preserve),
        );
        cx.kernel("ops.misc", "exposure_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Brightness<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("gain", self.value)
                .param("preserve", 0.0f32),
        );
        cx.kernel("ops.misc", "exposure_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Maplut<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            ParamBlock::new()
                .param("lut_width", self.lut.spec.entries as u32)
                .param("band", self.band.unwrap_or(-1)),
        );
        cx.kernel("ops.misc", "maplut_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Recomb<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("n", self.input.spec.layout.channel_count() as u32));
        cx.kernel("ops.misc", "recomb_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Ifthenelse<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("blend", self.blend.unwrap_or(false) as u32));
        cx.kernel("ops.misc", "ifthenelse_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Case<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let n = self.cases.len();
        match n {
            0 => cx.kernel("ops.misc", "passthrough_kernel"), // Fallback for 0 cases
            1 => cx.kernel("ops.misc", "case1_kernel"),
            2 => cx.kernel("ops.misc", "case2_kernel"),
            3 => cx.kernel("ops.misc", "case3_kernel"),
            4 => cx.kernel("ops.misc", "case4_kernel"),
            _ => cx.kernel("ops.misc", "case5_kernel"), // Hard cap fallback
        };
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Boolean<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::scalar("op", self.boolean_op.into_vips() as u32));
        cx.kernel("ops.misc", "boolean_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Relational<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::scalar("op", self.relational.into_vips() as u32));
        cx.kernel("ops.misc", "relational_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::BooleanConst<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let mut c_arr = [0.0f32; 4];
        let c_len = self.c.len();
        let src_max = self.input.spec.layout.component_max_f64() as f32;
        for i in 0..4 {
            c_arr[i] = self.c[i.min(c_len.saturating_sub(1))] as f32;
        }
        cx.param_block(
            ParamBlock::new()
                .param("boolean_op", self.boolean_op.into_vips() as u32)
                .param("src_max", src_max)
                .param("c0", c_arr[0])
                .param("c1", c_arr[1])
                .param("c2", c_arr[2])
                .param("c3", c_arr[3]),
        );
        cx.kernel("ops.misc", "boolean_const_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::RelationalConst<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let mut c_arr = [0.0f32; 4];
        let c_len = self.c.len();
        let src_max = self.input.spec.layout.component_max_f64() as f32;
        for i in 0..4 {
            c_arr[i] = self.c[i.min(c_len.saturating_sub(1))] as f32;
        }
        cx.param_block(
            ParamBlock::new()
                .param("relational", self.relational.into_vips() as u32)
                .param("src_max", src_max)
                .param("c0", c_arr[0])
                .param("c1", c_arr[1])
                .param("c2", c_arr[2])
                .param("c3", c_arr[3]),
        );
        cx.kernel("ops.misc", "relational_const_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}
