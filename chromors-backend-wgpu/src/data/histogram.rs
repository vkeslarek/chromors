use crate::prelude::*;
use chromors_core::data::histogram::Histogram;

pub struct RawTarget;

impl Target<HistogramKind, GpuBackend> for RawTarget {
    type Out = Vec<u8>;

    fn extract(
        &self,
        buf: &Buffer<GpuBackend>,
        _wu: &Atomic,
        ctx: &GpuContext,
    ) -> Result<Self::Out, Error> {
        buf.payload.read_to_cpu(ctx)
    }
}

// GpuView impl
impl GpuView for HistogramKind {
    fn input(&self) -> View {
        View::new("uint", "HistogramIn", "{ {buf}, {params}[0].bin_count }")
    }
    fn output(&self, _wu: &WorkUnit) -> OutputWrap {
        OutputWrap {
            arg: View::new("uint", "HistogramOut", "{ {buf}, {params}[0].bin_count }"),
            dest: OutBuffer::Target,
            encode: None,
            params: ParamBlock::scalar("bin_count", self.bins * self.bands.max(1)),
        }
    }
    fn source_params(&self, _wu: &WorkUnit) -> ParamBlock {
        ParamBlock::scalar("bin_count", self.bins * self.bands.max(1))
    }
}

// GPU-only histogram operations
pub struct HistogramOp {
    pub input: Input<ImageKind, GpuBackend>,
    pub bins: u32,
    pub channel: u32,
}

impl Operation<GpuBackend> for HistogramOp {
    type Output = HistogramKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<GpuBackend>> { vec![&self.input] }
    fn demand(&self, _out: &Atomic) -> Vec<Option<WorkUnit>> {
        let (w, h) = self.input.spec.dims();
        vec![Some(WorkUnit::Region(Region::full((w, h), Lod(0))))]
    }
    fn output_spec(&self) -> HistogramKind { HistogramKind { bins: self.bins, bands: 1 } }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.bins);
        state.write_u32(self.channel);
    }
}

impl Lower<GpuBackend> for HistogramOp {
    fn lower(&self, cx: &mut GpuBuilder) {
        let (w, h) = self.input.spec.dims();
        cx.dispatch((w.max(0) as u32, h.max(0) as u32));
        cx.kernel("ops.histogram", "histogram_kernel").param("channel", self.channel);
        cx.output(self.output_spec().output(cx.wu()));
    }
}

pub struct HistogramMultiOp {
    pub input: Input<ImageKind, GpuBackend>,
    pub bins: u32,
    pub bands: u32,
}

impl Operation<GpuBackend> for HistogramMultiOp {
    type Output = HistogramKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<GpuBackend>> { vec![&self.input] }
    fn demand(&self, _out: &Atomic) -> Vec<Option<WorkUnit>> {
        let (w, h) = self.input.spec.dims();
        vec![Some(WorkUnit::Region(Region::full((w, h), Lod(0))))]
    }
    fn output_spec(&self) -> HistogramKind { HistogramKind { bins: self.bins, bands: self.bands } }
    fn dyn_hash(&self, state: &mut dyn Hasher) { state.write_u32(self.bins); state.write_u32(self.bands); }
}

impl Lower<GpuBackend> for HistogramMultiOp {
    fn lower(&self, cx: &mut GpuBuilder) {
        let (w, h) = self.input.spec.dims();
        cx.dispatch((w.max(0) as u32, h.max(0) as u32));
        cx.kernel("ops.histogram", "histogram_multi_kernel").param("bins", self.bins).param("bands", self.bands);
        cx.output(self.output_spec().output(cx.wu()));
    }
}

// Extension trait for Image2D<GpuBackend>
pub trait GpuImageExt {
    fn histogram(&self, bins: u32, channel: u32) -> Histogram<GpuBackend>;
    fn histogram_multi(&self, bins: u32, bands: u32) -> Histogram<GpuBackend>;
    /// Histogram-equalize all bands. `bins` and `bands` are informational;
    /// the GPU lower currently uses 256 bins.
    fn equalize(&self, bins: u32, bands: u32) -> Image2D<GpuBackend>;
}

impl GpuImageExt for Image2D<GpuBackend> {
    fn histogram(&self, bins: u32, channel: u32) -> Histogram<GpuBackend> {
        self.push(HistogramOp { input: self.as_input(), bins, channel })
    }
    fn histogram_multi(&self, bins: u32, bands: u32) -> Histogram<GpuBackend> {
        self.push(HistogramMultiOp { input: self.as_input(), bins, bands: bands.min(4) })
    }
    fn equalize(&self, _bins: u32, _bands: u32) -> Image2D<GpuBackend> {
        use chromors_core::operation::stats::HistogramEqualize;
        self.push(HistogramEqualize { input: self.as_input(), band: None })
    }
}
