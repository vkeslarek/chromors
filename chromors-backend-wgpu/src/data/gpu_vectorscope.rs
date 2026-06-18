use crate::prelude::*;
use chromors_core::data::vectorscope::{VectorscopeKind, VectorscopeOp};
use crate::view::{OutBuffer, OutputWrap, ParamBlock, View};

impl GpuView for VectorscopeKind {
    fn input(&self) -> View {
        View::new("uint", "HistogramOut", "{ {buf}, {params}[0].bin_count }")
    }
    /// Reduction output: written directly into the target (atomic), no sandwich.
    fn output(&self, _wu: &WorkUnit) -> OutputWrap {
        OutputWrap {
            arg: View::new("uint", "HistogramOut", "{ {buf}, {params}[0].bin_count }"),
            dest: OutBuffer::Target,
            encode: None,
            // `bin_count` (= grid*grid) is consumed only by the output ctor's
            // bounds check, not a kernel arg.
            params: ParamBlock::scalar("bin_count", self.grid * self.grid),
        }
    }
}

impl Lower<GpuBackend> for VectorscopeOp<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let (w, h) = self.input.spec.dims();
        cx.dispatch((w.max(0) as u32, h.max(0) as u32));
        cx.param_block(ParamBlock::scalar("grid_size", self.grid));
        cx.kernel("ops.vectorscope", "vectorscope_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::io::Target<VectorscopeKind, GpuBackend> for crate::data::histogram::RawTarget {
    type Out = Vec<u8>;

    fn extract(
        &self,
        buf: &chromors_core::buffer::Buffer<GpuBackend>,
        _wu: &chromors_core::work_unit::Atomic,
        ctx: &crate::context::GpuContext,
    ) -> Result<Self::Out, chromors_core::error::Error> {
        buf.payload.read_to_cpu(ctx)
    }
}
