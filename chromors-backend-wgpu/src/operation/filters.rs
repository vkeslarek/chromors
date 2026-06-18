use crate::prelude::*;

impl Lower<GpuBackend> for crate::Sharpen<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let scale = if let WorkUnit::Region(r) = &wu {
            r.lod.scale_factor() as f32
        } else {
            1.0
        };
        let sigma = self.sigma.unwrap_or(0.5) as f32 / scale;
        let m1 = self.smooth.unwrap_or(1.0) as f32;
        cx.param_block(ParamBlock::new().param("sigma", sigma).param("m1", m1));
        cx.kernel("ops.filters", "sharpen_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Canny<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let scale = if let WorkUnit::Region(r) = &wu {
            r.lod.scale_factor() as f32
        } else {
            1.0
        };
        let sigma = self.sigma.unwrap_or(1.4) as f32 / scale;
        cx.kernel("ops.filters", "canny_kernel");
        cx.param("sigma", sigma);
        cx.param("radius", gauss_radius(sigma));
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Median<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("sz", self.size as u32));
        cx.kernel("ops.filters", "median_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Blur<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let scale = if let WorkUnit::Region(r) = &wu {
            r.lod.scale_factor() as f32
        } else {
            1.0
        };
        let sigma = self.sigma / scale;
        // Single-pass 2D kernel (not separable H/V): a separable fused
        // two-step pass would have the V step read NEIGHBOR threads' H
        // output, which a single dispatch can't order across workgroups.
        cx.kernel("ops.filters", "blur_kernel");
        cx.param("sigma", sigma);
        cx.param("radius", gauss_radius(sigma));
        cx.output(self.output_spec().output(cx.wu()));
    }
}

