//! The vectorscope datatype — a GPU-only 2-D Cb/Cr density grid. Like the
//! histogram it is `Atomic`-shaped (the whole grid is one reduction) and
//! GPU-only (no `VipsBand`).

use std::any::Any;
use std::hash::Hasher;

use crate::backend::gpu::view::{OutBuffer, OutputWrap, ParamBlock, View};
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::data::image::ImageKind;
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Atomic, Lod, Region, WorkUnit};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// A `grid × grid` Cb/Cr density grid of atomic counters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorscopeKind {
    pub grid: u32,
}

impl AnyKind for VectorscopeKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, _wu: &WorkUnit) -> u64 {
        (self.grid as u64 * self.grid as u64 * 4).max(16)
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.grid);
    }
}

impl Kind for VectorscopeKind {
    type WorkUnit = Atomic;
}

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
    /// Atomic-shaped: no region geometry, just the grid's total cell count.
    fn source_params(&self, _wu: &WorkUnit) -> ParamBlock {
        ParamBlock::scalar("bin_count", self.grid * self.grid)
    }
}

/// What the user holds. GPU-only.
pub type Vectorscope = Data<VectorscopeKind, GpuBackend>;

// ── Operation: Image → Vectorscope ────────────────────────────────────────────

pub struct VectorscopeOp {
    input: Input<ImageKind, GpuBackend>,
    grid: u32,
}

impl Operation<GpuBackend> for VectorscopeOp {
    type Output = VectorscopeKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<GpuBackend>> {
        vec![&self.input]
    }
    fn demand(&self, _out: &Atomic) -> Vec<Option<WorkUnit>> {
        let (w, h) = self.input.spec.dims();
        vec![Some(WorkUnit::Region(Region::full((w, h), Lod(0))))]
    }
    fn output_spec(&self) -> VectorscopeKind {
        VectorscopeKind { grid: self.grid }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.grid);
    }
}

impl Lower<GpuBackend> for VectorscopeOp {
    fn lower(&self, cx: &mut GpuBuilder) {
        // Inputs come from the source leaf. Reduction output: written directly.
        // bin_count (= grid*grid) is carried in `OutputWrap.params` (merged by
        // `cx.output`); grid_size is a real kernel arg used in gx/gy math. The
        // output itself is `Atomic`-shaped, so the dispatch domain must be set
        // explicitly from the input image's dims.
        let (w, h) = self.input.spec.dims();
        cx.dispatch((w.max(0) as u32, h.max(0) as u32));
        cx.param_block(ParamBlock::scalar("grid_size", self.grid));
        cx.kernel("ops.vectorscope", "vectorscope_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

// ── Ergonomic method on the image ─────────────────────────────────────────────

impl crate::data::image::Image2D<GpuBackend> {
    pub fn vectorscope(&self, grid: u32) -> Vectorscope {
        self.push(VectorscopeOp {
            input: self.as_input(),
            grid,
        })
    }
}

// ── Target ───────────────────────────────────────────────────────────────────

impl crate::io::Target<VectorscopeKind, GpuBackend> for crate::data::histogram::RawTarget {
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
