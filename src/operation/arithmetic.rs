use super::{
    OperationBoolean, OperationComplex, OperationComplex2, OperationComplexget, OperationMath,
    OperationMath2, OperationRelational, OperationRound,
};
use crate::backend::vips::IntoVipsEnum;
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

use crate::backend::Backend;
use crate::backend::gpu::datatype::ImageType;
use crate::backend::gpu::graph::{Graph, GraphNode, KernelSpec, NodeEval, NodeId};
use crate::backend::gpu::op::{
    GpuOperation, TypedOperation, emit_image, splice_sibling, working_image_type,
};
use crate::backend::gpu::param::Param;
use std::sync::Arc;

/// Build a params vec with a fixed count of up to 10 f32 constants, padded with 0.0.
fn const_params(constants: &[f64]) -> Vec<Param> {
    let n = constants.len() as u32;
    let mut params = Vec::with_capacity(2 + 10);
    params.push(Param::U32(n));
    for i in 0..10 {
        let v = constants.get(i).copied().unwrap_or(0.0) as f32;
        params.push(Param::F32(v));
    }
    params
}

fn const_params_with_op(op_val: u32, constants: &[f64]) -> Vec<Param> {
    let mut params = vec![Param::U32(op_val)];
    params.append(&mut const_params(constants));
    params
}

// ═══════════════════════════════════════════════════════════════════════════════
// AddOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct AddOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
}

impl<B: Backend> std::fmt::Debug for AddOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddOperation").finish()
    }
}

impl<B: Backend> Clone for AddOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
        }
    }
}

impl VipsOperation for AddOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"add\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
    }
}

impl TypedOperation for AddOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for AddOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "add_kernel",
            }),
            params: vec![],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SubtractOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct SubtractOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
}

impl<B: Backend> std::fmt::Debug for SubtractOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubtractOperation").finish()
    }
}

impl<B: Backend> Clone for SubtractOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
        }
    }
}

impl VipsOperation for SubtractOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"subtract\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
    }
}

impl TypedOperation for SubtractOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for SubtractOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "subtract_kernel",
            }),
            params: vec![],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MultiplyOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct MultiplyOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
}

impl<B: Backend> std::fmt::Debug for MultiplyOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiplyOperation").finish()
    }
}

impl<B: Backend> Clone for MultiplyOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
        }
    }
}

impl VipsOperation for MultiplyOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"multiply\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
    }
}

impl TypedOperation for MultiplyOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for MultiplyOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "multiply_kernel",
            }),
            params: vec![],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DivideOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct DivideOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
}

impl<B: Backend> std::fmt::Debug for DivideOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DivideOperation").finish()
    }
}

impl<B: Backend> Clone for DivideOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
        }
    }
}

impl VipsOperation for DivideOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"divide\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
    }
}

impl TypedOperation for DivideOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for DivideOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "divide_kernel",
            }),
            params: vec![],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MaxPairOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct MaxPairOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
}

impl<B: Backend> std::fmt::Debug for MaxPairOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxPairOperation").finish()
    }
}

impl<B: Backend> Clone for MaxPairOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
        }
    }
}

impl VipsOperation for MaxPairOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"maxpair\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
    }
}

impl TypedOperation for MaxPairOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for MaxPairOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "max_kernel",
            }),
            params: vec![],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MinPairOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct MinPairOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
}

impl<B: Backend> std::fmt::Debug for MinPairOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MinPairOperation").finish()
    }
}

impl<B: Backend> Clone for MinPairOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
        }
    }
}

impl VipsOperation for MinPairOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"minpair\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
    }
}

impl TypedOperation for MinPairOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for MinPairOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "min_kernel",
            }),
            params: vec![],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RemainderOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct RemainderOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
}

impl<B: Backend> std::fmt::Debug for RemainderOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemainderOperation").finish()
    }
}

impl<B: Backend> Clone for RemainderOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
        }
    }
}

impl VipsOperation for RemainderOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"remainder\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
    }
}

impl TypedOperation for RemainderOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for RemainderOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "remainder_kernel",
            }),
            params: vec![],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Complex2Operation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct Complex2Operation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
    pub cmplx: OperationComplex2,
}

impl<B: Backend> std::fmt::Debug for Complex2Operation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Complex2Operation")
            .field("cmplx", &self.cmplx)
            .finish()
    }
}

impl<B: Backend> Clone for Complex2Operation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
            cmplx: self.cmplx,
        }
    }
}

impl VipsOperation for Complex2Operation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"complex2\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
        op.set_int("cmplx", self.cmplx.into_vips());
    }
}

impl TypedOperation for Complex2Operation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for Complex2Operation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "complex2_kernel",
            }),
            params: vec![Param::U32(self.cmplx as u32)],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ComplexformOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct ComplexformOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
}

impl<B: Backend> std::fmt::Debug for ComplexformOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComplexformOperation").finish()
    }
}

impl<B: Backend> Clone for ComplexformOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
        }
    }
}

impl VipsOperation for ComplexformOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"complexform\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
    }
}

impl TypedOperation for ComplexformOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for ComplexformOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "complexform_kernel",
            }),
            params: vec![],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MathOperation (single-image + enum, no struct change needed)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct MathOperation {
    pub math: OperationMath,
}

impl VipsOperation for MathOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"math\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("math", self.math.into_vips());
    }
}

impl TypedOperation for MathOperation {
    type Output = ImageType;
}

impl GpuOperation for MathOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        emit_image(
            graph,
            input,
            self_arc,
            "ops.arithmetic",
            "math_kernel",
            vec![Param::U32(self.math as u32)],
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RoundOperation
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RoundOperation {
    pub round: OperationRound,
}

impl VipsOperation for RoundOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"round\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("round", self.round.into_vips());
    }
}

impl TypedOperation for RoundOperation {
    type Output = ImageType;
}

impl GpuOperation for RoundOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        emit_image(
            graph,
            input,
            self_arc,
            "ops.arithmetic",
            "round_kernel",
            vec![Param::U32(self.round as u32)],
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Math2Operation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct Math2Operation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
    pub math2: OperationMath2,
}

impl<B: Backend> std::fmt::Debug for Math2Operation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Math2Operation")
            .field("math2", &self.math2)
            .finish()
    }
}

impl<B: Backend> Clone for Math2Operation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
            math2: self.math2,
        }
    }
}

impl VipsOperation for Math2Operation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"math2\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
        op.set_int("math2", self.math2.into_vips());
    }
}

impl TypedOperation for Math2Operation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for Math2Operation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "math2_kernel",
            }),
            params: vec![Param::U32(self.math2 as u32)],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BooleanOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct BooleanOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
    pub boolean: OperationBoolean,
}

impl<B: Backend> std::fmt::Debug for BooleanOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BooleanOperation")
            .field("boolean", &self.boolean)
            .finish()
    }
}

impl<B: Backend> Clone for BooleanOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
            boolean: self.boolean,
        }
    }
}

impl VipsOperation for BooleanOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"boolean\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
        op.set_int("boolean", self.boolean.into_vips());
    }
}

impl TypedOperation for BooleanOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for BooleanOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "boolean_kernel",
            }),
            params: vec![Param::U32(self.boolean as u32)],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RelationalOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct RelationalOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
    pub relational: OperationRelational,
}

impl<B: Backend> std::fmt::Debug for RelationalOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RelationalOperation")
            .field("relational", &self.relational)
            .finish()
    }
}

impl<B: Backend> Clone for RelationalOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
            relational: self.relational,
        }
    }
}

impl VipsOperation for RelationalOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"relational\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("left", image);
        op.set_image("right", self.right.vips_ptr());
        op.set_int("relational", self.relational.into_vips());
    }
}

impl TypedOperation for RelationalOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for RelationalOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.arithmetic",
                function: "relational_kernel",
            }),
            params: vec![Param::U32(self.relational as u32)],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ComplexOperation
// ═══════════════════════════════════════════════════════════════════════════════
//
// Channels (r,g) = complex pair 0, (b,a) = complex pair 1.
// polar: (real,imag) -> (mag,angle); rect: inverse; conj: conjugate.

#[derive(Debug, Clone)]
pub struct ComplexOperation {
    pub cmplx: OperationComplex,
}

impl VipsOperation for ComplexOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"complex\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("cmplx", self.cmplx.into_vips());
    }
}

impl TypedOperation for ComplexOperation {
    type Output = ImageType;
}

impl GpuOperation for ComplexOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        emit_image(
            graph,
            input,
            self_arc,
            "ops.arithmetic",
            "complex_kernel",
            vec![Param::U32(self.cmplx as u32)],
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ComplexgetOperation
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct ComplexgetOperation {
    pub get: OperationComplexget,
}

impl VipsOperation for ComplexgetOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"complexget\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("get", self.get.into_vips());
    }
}

impl TypedOperation for ComplexgetOperation {
    type Output = ImageType;
}

impl GpuOperation for ComplexgetOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        emit_image(
            graph,
            input,
            self_arc,
            "ops.arithmetic",
            "complexget_kernel",
            vec![Param::U32(self.get as u32)],
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Math2ConstOperation (single-image + enum + constant array)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct Math2ConstOperation {
    pub math2: OperationMath2,
    pub constants: Vec<f64>,
}

impl VipsOperation for Math2ConstOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"math2_const\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("math2", self.math2.into_vips());
        op.set_array_double("c", &self.constants);
    }
}

impl TypedOperation for Math2ConstOperation {
    type Output = ImageType;
}

impl GpuOperation for Math2ConstOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let params = const_params_with_op(self.math2 as u32, &self.constants);
        emit_image(
            graph,
            input,
            self_arc,
            "ops.arithmetic",
            "math2_const_kernel",
            params,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BooleanConstOperation
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct BooleanConstOperation {
    pub boolean: OperationBoolean,
    pub constants: Vec<f64>,
}

impl VipsOperation for BooleanConstOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"boolean_const\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("boolean", self.boolean.into_vips());
        op.set_array_double("c", &self.constants);
    }
}

impl TypedOperation for BooleanConstOperation {
    type Output = ImageType;
}

impl GpuOperation for BooleanConstOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let params = const_params_with_op(self.boolean as u32, &self.constants);
        emit_image(
            graph,
            input,
            self_arc,
            "ops.arithmetic",
            "boolean_const_kernel",
            params,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RelationalConstOperation
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RelationalConstOperation {
    pub relational: OperationRelational,
    pub constants: Vec<f64>,
}

impl VipsOperation for RelationalConstOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"relational_const\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("relational", self.relational.into_vips());
        op.set_array_double("c", &self.constants);
    }
}

impl TypedOperation for RelationalConstOperation {
    type Output = ImageType;
}

impl GpuOperation for RelationalConstOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let params = const_params_with_op(self.relational as u32, &self.constants);
        emit_image(
            graph,
            input,
            self_arc,
            "ops.arithmetic",
            "relational_const_kernel",
            params,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LinearOperation (single-image + scalar a, b)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct LinearOperation {
    pub a: f64,
    pub b: f64,
    pub uchar: Option<bool>,
}

impl VipsOperation for LinearOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"linear\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_double("a", self.a);
        op.set_double("b", self.b);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
    }
}

impl TypedOperation for LinearOperation {
    type Output = ImageType;
}

impl GpuOperation for LinearOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        emit_image(
            graph,
            input,
            self_arc,
            "ops.arithmetic",
            "linear_kernel",
            vec![Param::F32(self.a as f32), Param::F32(self.b as f32)],
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RemainderConstOperation
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RemainderConstOperation {
    pub constants: Vec<f64>,
}

impl VipsOperation for RemainderConstOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"remainder_const\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_array_double("c", &self.constants);
    }
}

impl TypedOperation for RemainderConstOperation {
    type Output = ImageType;
}

impl GpuOperation for RemainderConstOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let params = const_params(&self.constants);
        emit_image(
            graph,
            input,
            self_arc,
            "ops.arithmetic",
            "remainder_const_kernel",
            params,
        )
    }
}
