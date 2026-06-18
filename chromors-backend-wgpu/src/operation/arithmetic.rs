use crate::prelude::*;

impl Lower<GpuBackend> for crate::Add<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.arithmetic", "add_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Subtract<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.arithmetic", "subtract_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Multiply<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.arithmetic", "multiply_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Divide<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.arithmetic", "divide_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::MaxPair<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.arithmetic", "max_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::MinPair<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.arithmetic", "min_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Remainder<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let src_max = self.left.spec.layout.component_max_f64() as f32;
        cx.param_block(ParamBlock::scalar("src_max", src_max));
        cx.kernel("ops.arithmetic", "remainder_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Complexform<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.arithmetic", "complexform_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Complex2<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::scalar("op", self.cmplx.into_vips() as u32));
        cx.kernel("ops.arithmetic", "complex2_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Math<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::scalar("op", self.math.into_vips() as u32));
        cx.kernel("ops.arithmetic", "math_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Round<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let src_max = self.input.spec.layout.component_max_f64() as f32;
        // Field order must match round_kernel's parameter order exactly --
        // kernel args are bound positionally from this block.
        cx.param_block(
            ParamBlock::new()
                .param("op", self.round.into_vips() as u32)
                .param("src_max", src_max),
        );
        cx.kernel("ops.arithmetic", "round_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Math2<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::scalar("op", self.math2.into_vips() as u32));
        cx.kernel("ops.arithmetic", "math2_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Linear<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let mut a_arr = [0.0f32; 4];
        let mut b_arr = [0.0f32; 4];
        let a_len = self.a.len();
        let b_len = self.b.len();
        let src_max = self.input.spec.layout.component_max_f64();
        for i in 0..4 {
            a_arr[i] = self.a[i.min(a_len.saturating_sub(1))] as f32;
            b_arr[i] = (self.b[i.min(b_len.saturating_sub(1))] / src_max) as f32;
        }
        cx.param_block(
            ParamBlock::new()
                .param("a0", a_arr[0])
                .param("a1", a_arr[1])
                .param("a2", a_arr[2])
                .param("a3", a_arr[3])
                .param("b0", b_arr[0])
                .param("b1", b_arr[1])
                .param("b2", b_arr[2])
                .param("b3", b_arr[3]),
        );
        cx.kernel("ops.arithmetic", "linear_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::Math2Const<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let mut c_arr = [0.0f32; 4];
        let c_len = self.c.len();
        let src_max = self.input.spec.layout.component_max_f64() as f32;
        for i in 0..4 {
            c_arr[i] = self.c[i.min(c_len.saturating_sub(1))] as f32;
        }
        cx.param_block(
            ParamBlock::new()
                .param("math2", self.math2.into_vips() as u32)
                .param("src_max", src_max)
                .param("c0", c_arr[0])
                .param("c1", c_arr[1])
                .param("c2", c_arr[2])
                .param("c3", c_arr[3]),
        );
        cx.kernel("ops.arithmetic", "math2_const_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for crate::RemainderConst<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let mut c_arr = [0.0f32; 4];
        let c_len = self.c.len();
        for i in 0..4 {
            c_arr[i] = self.c[i.min(c_len.saturating_sub(1))] as f32;
        }
        let src_max = self.input.spec.layout.component_max_f64() as f32;
        cx.param_block(
            ParamBlock::new()
                .param("src_max", src_max)
                .param("c0", c_arr[0])
                .param("c1", c_arr[1])
                .param("c2", c_arr[2])
                .param("c3", c_arr[3]),
        );
        cx.kernel("ops.arithmetic", "remainder_const_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

