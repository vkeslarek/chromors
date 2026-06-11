//! The vectorscope datatype — a GPU-only 2-D Cb/Cr density grid. Like the
//! histogram it is `Atomic`-shaped (the whole grid is one reduction) and
//! GPU-only (no `VipsBand`).

use std::any::Any;
use std::hash::Hasher;

use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::gpu::view::{OutBuffer, OutputWrap, ParamBlock, View};
use crate::data::image::ImageKind;
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Atomic, Lod, Region, Shape, WorkUnit};

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
    fn shape(&self) -> Shape {
        Shape::Atomic
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
        View::new("uint", "VectorscopeOut", "{ {buf}, {params}[0].grid_size }")
    }
    /// Reduction output: written directly into the target (atomic), no sandwich.
    fn output(&self) -> OutputWrap {
        OutputWrap {
            arg_type: "VectorscopeOut".into(),
            arg_ctor: "{ {buf}, {params}[0].grid_size }".into(),
            arg_buffer: OutBuffer::Target,
            encode: None,
        }
    }
    fn params(&self, _wu: &WorkUnit) -> ParamBlock {
        ParamBlock::scalar("grid_size", "uint", self.grid)
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
        let wu = cx.wu().clone();
        // Inputs come from the source leaf. Reduction output: written directly.
        cx.param_block(self.output_spec().params(&wu));
        cx.kernel("vectorscope_kernel");
        cx.output(self.output_spec().output());
    }
}

// ── Ergonomic method on the image ─────────────────────────────────────────────

impl crate::data::image::Image2D<GpuBackend> {
    pub fn vectorscope(&self, grid: u32) -> Vectorscope {
        self.push(VectorscopeOp { input: self.as_input(), grid })
    }
}
