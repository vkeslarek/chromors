use crate::data::histogram::GpuImageExt;
use crate::prelude::*;

impl chromors_core::operation::Lower<GpuBackend> for crate::HistogramFind<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let img = Image2D::<GpuBackend> {
            root: self.input.src.clone(),
            ctx: cx.ctx().clone(),
            spec: self.input.spec.clone(),
        };
        let out_spec = self.output_spec();
        let bands = out_spec.layout.channel_count() as u32;
        let hist = match self.band {
            Some(c) => img.histogram(256, c as u32),
            None => img.histogram_multi(256, bands),
        };
        let hist_buf = match hist.materialize(Atomic) {
            Ok(b) => b,
            Err(e) => {
                cx.fail(e);
                return;
            }
        };
        cx.extra_input(
            hist.spec.input(),
            hist.spec.source_params(&WorkUnit::Atomic),
            hist_buf.payload,
        );
        cx.kernel("ops.histogram", "histogram_to_image_kernel")
            .param("bins", 256u32)
            .param("bands", bands);
        cx.output(out_spec.output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<GpuBackend>
    for chromors_core::operation::stats::HistogramCumulative<GpuBackend>
{
    fn lower(&self, cx: &mut GpuBuilder) {
        if !stage_histogram_image(cx, &self.input) {
            return;
        }
        let bins = self.input.spec.width as u32;
        let bands = self.input.spec.layout.channel_count() as u32;
        cx.kernel("ops.histogram", "hist_image_cumulative_kernel")
            .param("bins", bins)
            .param("bands", bands);
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<GpuBackend>
    for chromors_core::operation::stats::HistogramNormalize<GpuBackend>
{
    fn lower(&self, cx: &mut GpuBuilder) {
        if !stage_histogram_image(cx, &self.input) {
            return;
        }
        let bins = self.input.spec.width as u32;
        let bands = self.input.spec.layout.channel_count() as u32;
        cx.kernel("ops.histogram", "hist_image_normalize_kernel")
            .param("bins", bins)
            .param("bands", bands);
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<GpuBackend> for crate::HistogramPlot<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        if !stage_histogram_image(cx, &self.input) {
            return;
        }
        let width = self.input.spec.width as u32;
        let bands = self.input.spec.layout.channel_count() as u32;
        cx.kernel("ops.histogram", "hist_plot_kernel")
            .param("width", width)
            .param("bands", bands);
        cx.output(self.output_spec().output(cx.wu()));
    }
}

fn stage_histogram_image(cx: &mut GpuBuilder, input: &Input<ImageKind, GpuBackend>) -> bool {
    let img = Image2D::<GpuBackend> {
        root: input.src.clone(),
        ctx: cx.ctx().clone(),
        spec: input.spec.clone(),
    };
    let region = Region::full((input.spec.width, input.spec.height), Lod(0));
    let buf = match img.materialize(region.clone()) {
        Ok(b) => b,
        Err(e) => {
            cx.fail(e);
            return false;
        }
    };
    cx.extra_input(
        input.spec.input(),
        input.spec.source_params(&WorkUnit::Region(region)),
        buf.payload,
    );
    true
}

use chromors_core::operation::stats::HistogramEqualize;

impl chromors_core::operation::Lower<GpuBackend> for HistogramEqualize<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let img = Image2D::<GpuBackend> {
            root: self.input.src.clone(),
            ctx: cx.ctx().clone(),
            spec: self.input.spec.clone(),
        };
        let bands = match self.band {
            Some(_) => 1u32,
            None => (self.input.spec.layout.channel_count() as u32).min(4),
        };
        use crate::data::histogram::GpuImageExt;
        let hist = match self.band {
            Some(c) => img.histogram(256, c as u32),
            None => img.histogram_multi(256, bands),
        };
        use crate::stage_ext::StageExt;
        let lut = hist.stage().equalize_lut();
        use chromors_core::work_unit::Range;
        let range = Range {
            start: 0,
            end: lut.spec.entries as i32,
        };
        let lut_buf = match lut.materialize(range.clone()) {
            Ok(b) => b,
            Err(e) => {
                cx.fail(e);
                return;
            }
        };
        cx.extra_input(
            lut.spec.input(),
            lut.spec.source_params(&WorkUnit::Range(range)),
            lut_buf.payload,
        );
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("lut_width", lut.spec.entries)
                .param("band", self.band.unwrap_or(-1)),
        );
        cx.kernel("ops.misc", "maplut_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}
