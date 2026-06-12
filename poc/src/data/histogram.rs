//! The histogram datatype — a GPU-only, `Atomic`-shaped reduction.
//!
//! `HistogramKind` has no `VipsBand`, so `Data<HistogramKind, VipsBackend>`
//! does not type-check: it is GPU-only *by construction*, enforced by the type
//! system, never a runtime "unsupported backend" error.

use std::any::Any;
use std::hash::Hasher;

use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::gpu::view::{OutBuffer, OutputWrap, ParamBlock, View};
use crate::data::image::ImageKind;
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Atomic, Region, Shape, WorkUnit};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// A 1-D histogram of `bins` atomic counters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistogramKind {
    pub bins: u32,
}

impl AnyKind for HistogramKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn shape(&self) -> Shape {
        Shape::Atomic
    }
    fn byte_size(&self, _wu: &WorkUnit) -> u64 {
        (self.bins as u64 * 4).max(16)
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.bins);
    }
}

impl Kind for HistogramKind {
    type WorkUnit = Atomic;
}

impl GpuView for HistogramKind {
    /// A histogram is never a graph input in practice, but the wrapper that
    /// would read it back.
    fn input(&self) -> View {
        View::new("uint", "HistogramOut", "{ {buf}, {params}[0].bin_count }")
    }
    /// Reduction output: the kernel writes the atomic-accumulate wrapper
    /// directly into the target — no working scratch, no encode step.
    fn output(&self) -> OutputWrap {
        OutputWrap {
            arg_type: "HistogramOut".into(),
            arg_ctor: "{ {buf}, {params}[0].bin_count }".into(),
            arg_buffer: OutBuffer::Target,
            buffer_type: "uint".into(),
            encode: None,
        }
    }
    fn params(&self, _wu: &WorkUnit) -> ParamBlock {
        ParamBlock::scalar("bin_count", "uint", self.bins)
    }
}

/// What the user holds. GPU-only (no `VipsBand` on the Kind).
pub type Histogram = Data<HistogramKind, GpuBackend>;

// ── Operation: Image → Histogram (cross-Kind) ─────────────────────────────────

/// Per-pixel histogram accumulation. `channel`: 0=R 1=G 2=B 3=A 4=luma.
pub struct HistogramOp {
    input: Input<ImageKind, GpuBackend>,
    bins: u32,
    channel: u32,
}

impl Operation<GpuBackend> for HistogramOp {
    type Output = HistogramKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<GpuBackend>> {
        vec![&self.input]
    }
    fn demand(&self, _out: &Atomic) -> Vec<Option<WorkUnit>> {
        // One thread per input pixel ⇒ demand the whole image.
        let (w, h) = self.input.spec.dims();
        vec![Some(WorkUnit::Region(Region::full((w, h), crate::work_unit::Lod(0))))]
    }
    fn output_spec(&self) -> HistogramKind {
        HistogramKind { bins: self.bins }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.bins);
        state.write_u32(self.channel);
    }
}

impl Lower<GpuBackend> for HistogramOp {
    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        // Inputs come from the source leaf, not here. Reduction output: the
        // kernel writes the atomic-accumulate wrapper directly (no sandwich).
        // bin_count is consumed by the output ctor only, not a kernel arg.
        cx.output_params(self.output_spec().params(&wu));
        cx.kernel("histogram_kernel").param("channel", self.channel);
        cx.output(self.output_spec().output());
    }
}

// ── Ergonomic method on the image ─────────────────────────────────────────────

impl crate::data::image::Image2D<GpuBackend> {
    pub fn histogram(&self, bins: u32, channel: u32) -> Histogram {
        self.push(HistogramOp { input: self.as_input(), bins, channel })
    }
}

// ── Target ───────────────────────────────────────────────────────────────────

/// Reads an `Atomic`-shaped GPU buffer (e.g. histogram bins) back to host RAM
/// as raw `u32` counter bytes.
pub struct RawTarget;

impl crate::io::Target<HistogramKind, GpuBackend> for RawTarget {
    type Out = Vec<u8>;

    fn extract(
        &self,
        buf: &crate::buffer::Buffer<GpuBackend>,
        _wu: &Atomic,
        ctx: &crate::backend::gpu::context::GpuContext,
    ) -> Result<Self::Out, crate::error::Error> {
        buf.payload.read_to_cpu(ctx)
    }
}
