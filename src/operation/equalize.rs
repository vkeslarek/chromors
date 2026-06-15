//! Histogram→LUT and Histogram→Histogram GPU ops — the CDF/equalize pipeline.
//!
//! All three ops read a *staged* (fully materialized) [`HistogramKind`]
//! input: their kernels loop over the whole `bins * bands` buffer, which only
//! exists once the reduction pass that produced it has finished. Callers must
//! `.stage()` the histogram before pushing any of these (see
//! [`crate::data::image::Image2D::equalize`] for the canonical pipeline).

use std::hash::Hasher;

use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::data::histogram::{Histogram, HistogramKind};
use crate::data::lut::LutKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Atomic, Range, WorkUnit};

// ── EqualizeLut: HistogramKind -> LutKind ───────────────────────────────────

/// CDF of a staged histogram, as a `[0, 1]`-valued LUT (one entry per bin,
/// `bands` channels). Feed into [`crate::data::image::Image2D::maplut`] to
/// equalize.
pub struct EqualizeLut {
    pub histogram: Input<HistogramKind, GpuBackend>,
}

impl Operation<GpuBackend> for EqualizeLut {
    type Output = LutKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<GpuBackend>> {
        vec![&self.histogram]
    }
    fn demand(&self, _out: &Range) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Atomic)]
    }
    fn output_spec(&self) -> LutKind {
        LutKind::new(self.histogram.spec.bins, self.histogram.spec.bands.max(1))
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<GpuBackend> for EqualizeLut {
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

// ── HistogramCumulative: HistogramKind -> HistogramKind ─────────────────────

/// Running sum per band: `out[b*bins+i] = sum(in[b*bins .. b*bins+i])`.
pub struct HistogramCumulative {
    pub histogram: Input<HistogramKind, GpuBackend>,
}

impl Operation<GpuBackend> for HistogramCumulative {
    type Output = HistogramKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<GpuBackend>> {
        vec![&self.histogram]
    }
    fn demand(&self, _out: &Atomic) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Atomic)]
    }
    fn output_spec(&self) -> HistogramKind {
        (*self.histogram.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<GpuBackend> for HistogramCumulative {
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

// ── HistogramNormalize: HistogramKind -> HistogramKind ──────────────────────

/// Scales each band so its maximum bin value maps to `bins - 1`.
pub struct HistogramNormalize {
    pub histogram: Input<HistogramKind, GpuBackend>,
}

impl Operation<GpuBackend> for HistogramNormalize {
    type Output = HistogramKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<GpuBackend>> {
        vec![&self.histogram]
    }
    fn demand(&self, _out: &Atomic) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Atomic)]
    }
    fn output_spec(&self) -> HistogramKind {
        (*self.histogram.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<GpuBackend> for HistogramNormalize {
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

// ── Ergonomics ───────────────────────────────────────────────────────────────

impl Histogram<GpuBackend> {
    /// CDF of `self`, as a `[0, 1]`-valued LUT (one entry per bin, `bands`
    /// channels). `self` must be staged (fully materialized) — the kernel
    /// reads the whole histogram.
    pub fn equalize_lut(&self) -> crate::data::lut::Lut<GpuBackend> {
        self.push(EqualizeLut {
            histogram: self.as_input(),
        })
    }

    /// Running sum per band. `self` must be staged (fully materialized) —
    /// the kernel reads the whole histogram.
    pub fn cumulative(&self) -> Histogram<GpuBackend> {
        self.push(HistogramCumulative {
            histogram: self.as_input(),
        })
    }

    /// Scale each band so its max bin value maps to `bins - 1`. `self` must
    /// be staged.
    pub fn normalize(&self) -> Histogram<GpuBackend> {
        self.push(HistogramNormalize {
            histogram: self.as_input(),
        })
    }
}

impl crate::data::image::Image2D<GpuBackend> {
    /// Histogram-equalize `channel` (0=R 1=G 2=B 3=A 4=luma) via a
    /// `bins`-entry CDF LUT applied to all channels. Two staging barriers,
    /// three passes, fully lazy.
    pub fn equalize(&self, bins: u32, channel: u32) -> Self {
        let lut = self.histogram(bins, channel).stage().equalize_lut().stage();
        self.maplut(lut.as_input(), None)
    }
}
