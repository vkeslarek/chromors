use crate::prelude::*;
use chromors_core::backend::Backend;
use crate::{GpuBackend, GpuBuilder, GpuView};
use chromors_core::operation::Lower;
use chromors_core::operation::edge::{Invert, Sign, Abs, Hypot, Sobel, Prewitt, Scharr};

impl Lower<GpuBackend> for Invert<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.edge", "invert_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for Sign<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.edge", "sign_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for Abs<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.edge", "abs_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for Hypot<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.edge", "hypot_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for Sobel<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.edge", "sobel_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for Prewitt<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.edge", "prewitt_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for Scharr<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.edge", "scharr_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}
