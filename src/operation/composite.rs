use crate::backend::gpu::datatype::ImageType;
use crate::backend::gpu::graph::NodeEval;
use crate::backend::gpu::graph::{Graph, GraphNode, KernelSpec, NodeId};
use crate::backend::gpu::op::working_image_type;
use crate::backend::gpu::op::{GpuOperation, TypedOperation};
use crate::backend::gpu::param::Param;
use crate::geometry::Rect;
use std::sync::Arc;

use super::Direction;
use crate::backend::Backend;
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::backend::vips::{IntoVipsEnum, IntoVipsInterpretation};
use crate::libvips_ffi as ffi;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Align {
    Low,
    Centre,
    High,
}
impl IntoVipsEnum for Align {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    Clear,
    Source,
    Over,
    In,
    Out,
    Atop,
    Dest,
    DestOver,
    DestIn,
    DestOut,
    DestAtop,
    Xor,
    Add,
    Saturate,
}
impl IntoVipsEnum for BlendMode {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

impl std::fmt::Display for BlendMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            BlendMode::Over => "Normal",
            BlendMode::Add => "Addition",
            BlendMode::Clear => "Erase",
            BlendMode::Source => "Replace",
            BlendMode::In => "Mask In",
            BlendMode::Out => "Mask Out",
            BlendMode::Atop => "Atop",
            BlendMode::Dest => "Destination",
            BlendMode::DestOver => "Behind",
            BlendMode::DestIn => "Dest In",
            BlendMode::DestOut => "Dest Out",
            BlendMode::DestAtop => "Dest Atop",
            BlendMode::Xor => "XOR",
            BlendMode::Saturate => "Saturate",
        };
        write!(f, "{}", name)
    }
}

pub struct Composite2Operation<B: Backend> {
    pub overlay: crate::data::image::Image2D<B>,
    pub mode: BlendMode,
    pub x: Option<i32>,
    pub y: Option<i32>,
    pub compositing_space: Option<crate::color::space::ColorSpace>,
    pub premultiplied: Option<bool>,
}

impl<B: Backend> std::fmt::Debug for Composite2Operation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Composite2Operation")
            .field("mode", &self.mode)
            .field("x", &self.x)
            .field("y", &self.y)
            .field("compositing_space", &self.compositing_space)
            .field("premultiplied", &self.premultiplied)
            .finish()
    }
}

impl<B: Backend> Clone for Composite2Operation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            overlay: self.overlay.clone(),
            mode: self.mode,
            x: self.x,
            y: self.y,
            compositing_space: self.compositing_space,
            premultiplied: self.premultiplied,
        }
    }
}

impl VipsOperation for Composite2Operation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"composite2\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("base", image);
        op.set_image("overlay", self.overlay.vips_ptr());
        op.set_int("mode", self.mode.into_vips());
        if let Some(v) = self.x {
            op.set_int("x", v);
        }
        if let Some(v) = self.y {
            op.set_int("y", v);
        }
        if let Some(cs) = self.compositing_space {
            op.set_int("compositing_space", cs.into_vips_interpretation());
        }
        if let Some(v) = self.premultiplied {
            op.set_bool("premultiplied", v);
        }
    }
}

pub struct JoinOperation<B: Backend> {
    pub right: crate::data::image::Image2D<B>,
    pub direction: Direction,
    pub expand: Option<bool>,
    pub shim: Option<i32>,
    pub background: Option<[f64; 3]>,
    pub align: Option<Align>,
}

impl<B: Backend> std::fmt::Debug for JoinOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JoinOperation")
            .field("direction", &self.direction)
            .field("expand", &self.expand)
            .field("shim", &self.shim)
            .field("align", &self.align)
            .finish()
    }
}

impl<B: Backend> Clone for JoinOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            right: self.right.clone(),
            direction: self.direction,
            expand: self.expand,
            shim: self.shim,
            background: self.background,
            align: self.align,
        }
    }
}

impl VipsOperation for JoinOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"join\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in1", image);
        op.set_image("in2", self.right.vips_ptr());
        op.set_int("direction", self.direction.into_vips());
        if let Some(v) = self.expand {
            op.set_bool("expand", v);
        }
        if let Some(v) = self.shim {
            op.set_int("shim", v);
        }
        if let Some(v) = &self.background {
            op.set_array_double("background", v);
        }
        if let Some(v) = self.align {
            op.set_int("align", v.into_vips());
        }
    }
}

impl TypedOperation for JoinOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for JoinOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let right_id = crate::backend::gpu::op::splice_sibling(graph, &self.right);

        let bg = self.background.unwrap_or([0.0, 0.0, 0.0]);
        let dir = self.direction as i32 as u32;
        let shim = self.shim.unwrap_or(0);
        let (src0_w, src0_h) = graph
            .source_dimensions(input)
            .unwrap_or((self.right.width(), self.right.height()));
        let (src1_w, src1_h) = (self.right.width(), self.right.height());

        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.composite",
                function: "join_kernel",
            }),
            params: vec![
                Param::U32(dir),
                Param::I32(shim),
                Param::U32(src0_w),
                Param::U32(src0_h),
                Param::U32(src1_w),
                Param::U32(src1_h),
                Param::F32(bg[0] as f32),
                Param::F32(bg[1] as f32),
                Param::F32(bg[2] as f32),
            ],
            op: self_arc,
            datatype: working_image_type(),
        })
    }

    fn output_dims(&self, w: u32, h: u32) -> Option<(u32, u32)> {
        let shim = self.shim.unwrap_or(0) as u32;
        Some(match self.direction {
            Direction::Horizontal => (w + shim + self.right.width(), h.max(self.right.height())),
            Direction::Vertical => (w.max(self.right.width()), h + shim + self.right.height()),
        })
    }

    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

pub struct InsertOperation<B: Backend> {
    pub sub: crate::data::image::Image2D<B>,
    pub x: i32,
    pub y: i32,
    pub expand: Option<bool>,
    pub background: Option<[f64; 3]>,
}

impl<B: Backend> std::fmt::Debug for InsertOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InsertOperation")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("expand", &self.expand)
            .finish()
    }
}

impl<B: Backend> Clone for InsertOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            sub: self.sub.clone(),
            x: self.x,
            y: self.y,
            expand: self.expand,
            background: self.background,
        }
    }
}

impl VipsOperation for InsertOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"insert\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("main", image);
        op.set_image("sub", self.sub.vips_ptr());
        op.set_int("x", self.x);
        op.set_int("y", self.y);
        if let Some(v) = self.expand {
            op.set_bool("expand", v);
        }
        if let Some(v) = &self.background {
            op.set_array_double("background", v);
        }
    }
}

impl TypedOperation for InsertOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for InsertOperation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let sub_id = crate::backend::gpu::op::splice_sibling(graph, &self.sub);

        let bg = self.background.unwrap_or([0.0, 0.0, 0.0]);

        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, sub_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.composite",
                function: "insert_kernel",
            }),
            params: vec![
                Param::I32(self.x),
                Param::I32(self.y),
                Param::U32(self.sub.width()),
                Param::U32(self.sub.height()),
                Param::F32(bg[0] as f32),
                Param::F32(bg[1] as f32),
                Param::F32(bg[2] as f32),
            ],
            op: self_arc,
            datatype: working_image_type(),
        })
    }

    fn output_dims(&self, w: u32, h: u32) -> Option<(u32, u32)> {
        if self.expand.unwrap_or(false) {
            let ow = (w as i32).max(self.x + self.sub.width() as i32).max(0) as u32;
            let oh = (h as i32).max(self.y + self.sub.height() as i32).max(0) as u32;
            Some((ow, oh))
        } else {
            Some((w, h))
        }
    }

    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]
    }
}

// ── Composite2Operation ───────────────────────────────────────────────────────

impl TypedOperation for Composite2Operation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for Composite2Operation<crate::backend::gpu::GpuBackend> {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let overlay_node_id = crate::backend::gpu::op::splice_sibling(graph, &self.overlay);

        let mode = self.mode as i32 as u32;
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, overlay_node_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.composite",
                function: "compose_kernel",
            }),
            params: vec![
                Param::U32(mode),
                Param::I32(self.x.unwrap_or(0)),
                Param::I32(self.y.unwrap_or(0)),
            ],
            op: self_arc,
            datatype: working_image_type(),
        })
    }

    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        let (ox, oy) = (self.x.unwrap_or(0), self.y.unwrap_or(0));
        match wu {
            crate::backend::gpu::work_unit::WorkUnit::Region { rect, lod } => vec![
                (0, wu.clone()),
                (
                    1,
                    crate::backend::gpu::work_unit::WorkUnit::Region {
                        rect: Rect::new(rect.x - ox, rect.y - oy, rect.width, rect.height),
                        lod: *lod,
                    },
                ),
            ],
            _ => vec![(0, wu.clone()), (1, wu.clone())],
        }
    }
}
