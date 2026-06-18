use crate::prelude::*;

impl Lower<GpuBackend> for chromors_core::operation::equalize::EqualizeLut<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let bins = self.histogram.spec.bins;
        let bands = self.histogram.spec.bands.max(1);
        cx.dispatch((bins, 1));
        cx.kernel("ops.histogram", "equalize_lut_kernel")
            .param("bins", bins)
            .param("bands", bands);
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for chromors_core::operation::equalize::HistogramCumulative<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let bins = self.histogram.spec.bins;
        let bands = self.histogram.spec.bands.max(1);
        cx.dispatch((bins, 1));
        cx.kernel("ops.histogram", "histogram_cumulative_kernel")
            .param("bins", bins)
            .param("bands", bands);
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl Lower<GpuBackend> for chromors_core::operation::equalize::HistogramNormalize<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let bins = self.histogram.spec.bins;
        let bands = self.histogram.spec.bands.max(1);
        cx.dispatch((bins, 1));
        cx.kernel("ops.histogram", "histogram_normalize_kernel")
            .param("bins", bins)
            .param("bands", bands);
        cx.output(self.output_spec().output(cx.wu()));
    }
}
