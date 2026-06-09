use super::OperationMorphology;
use crate::backend::vips::IntoVipsEnum;
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

use crate::backend::gpu::graph::{Graph, GraphNode, KernelSpec, NodeEval, NodeId};
use crate::backend::gpu::op::GpuOperation;
use crate::backend::gpu::param::Param;
use crate::backend::gpu::value::ValueKind;
use crate::backend::Backend;
use crate::geometry::Rect;
use std::sync::Arc;

/// Inject a GPU image as a source node into the graph, returning its node id.
fn inject_gpu_source(
    image: &crate::data::image::Image<crate::backend::gpu::GpuBackend>,
    graph: &mut Graph,
) -> NodeId {
    if graph.get_node(image.root_id()).is_some() || graph.get_source(image.root_id()).is_some() {
        return image.root_id();
    }
    let (w, h) = (image.width(), image.height());
    let target = crate::target::ImageTarget::new(image.clone());
    let mat = target
        .pull(crate::geometry::Rect::new(0, 0, w as i32, h as i32), 0)
        .expect("inject_gpu_source: failed to pull image");
    let img_buf = Arc::new(crate::backend::gpu::buffer::ImageBuffer {
        buffer: mat.buffer.clone(),
        width: mat.buffer_rect.width as u32,
        height: mat.buffer_rect.height as u32,
        meta: mat.meta,
    });
    let source = crate::backend::gpu::source::GpuSource::new_buffer(
        img_buf,
        image.handle.node.ctx.clone(),
    );
    graph.add_source(Arc::new(source))
}

/// Halo (half mask size) struct for inverse_map.
struct MaskHalo {
    hw: i32,
    hh: i32,
}

impl MaskHalo {
    fn from_dims(mask_w: u32, mask_h: u32) -> Self {
        Self {
            hw: (mask_w as i32) / 2,
            hh: (mask_h as i32) / 2,
        }
    }

    fn expand(&self, rect: Rect, w: u32, h: u32) -> Rect {
        let bounds = Rect::new(0, 0, w as i32, h as i32);
        let halo = self.hw.max(self.hh);
        rect.expand(halo, bounds)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ConvolutionOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct ConvolutionOperation<B: Backend> {
    pub mask: crate::data::image::Image<B>,
    pub precision: Option<i32>,
    pub layers: Option<i32>,
    pub cluster: Option<i32>,
}

impl<B: Backend> std::fmt::Debug for ConvolutionOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvolutionOperation")
            .field("precision", &self.precision)
            .finish()
    }
}

impl<B: Backend> Clone for ConvolutionOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mask: self.mask.clone(),
            precision: self.precision,
            layers: self.layers,
            cluster: self.cluster,
        }
    }
}

impl VipsOperation for ConvolutionOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"conv\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("mask", self.mask.vips_ptr());
        if let Some(v) = self.precision {
            op.set_int("precision", v);
        }
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        if let Some(v) = self.cluster {
            op.set_int("cluster", v);
        }
    }
}

impl GpuOperation for ConvolutionOperation<crate::backend::gpu::GpuBackend> {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let mask_id = inject_gpu_source(&self.mask, graph);
        let mw = self.mask.width();
        let mh = self.mask.height();
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, mask_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.convolution",
                function: "convolution_kernel",
            }),
            params: vec![Param::U32(mw), Param::U32(mh)],
            op: self_arc,
            output: ValueKind::Image,
        })
    }

    fn inverse_map(
        &self,
        output_rect: Rect,
        w: u32,
        h: u32,
        _lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, Rect)> {
        let halo = MaskHalo::from_dims(self.mask.width(), self.mask.height());
        let expanded = halo.expand(output_rect, w, h);
        vec![(0, expanded), (1, Rect::new(0, 0, self.mask.width() as i32, self.mask.height() as i32))]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CompassOperation — edge detection with compass operators (GPU TODO)
// ═══════════════════════════════════════════════════════════════════════════════

pub struct CompassOperation<B: Backend> {
    pub mask: crate::data::image::Image<B>,
}

impl<B: Backend> std::fmt::Debug for CompassOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompassOperation").finish()
    }
}

impl<B: Backend> Clone for CompassOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mask: self.mask.clone(),
        }
    }
}

impl VipsOperation for CompassOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"compass\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("mask", self.mask.vips_ptr());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MorphOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct MorphOperation<B: Backend> {
    pub mask: crate::data::image::Image<B>,
    pub morph: OperationMorphology,
}

impl<B: Backend> std::fmt::Debug for MorphOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MorphOperation")
            .field("morph", &self.morph)
            .finish()
    }
}

impl<B: Backend> Clone for MorphOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mask: self.mask.clone(),
            morph: self.morph,
        }
    }
}

impl VipsOperation for MorphOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"morph\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("mask", self.mask.vips_ptr());
        op.set_int("morph", self.morph.into_vips());
    }
}

impl GpuOperation for MorphOperation<crate::backend::gpu::GpuBackend> {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let mask_id = inject_gpu_source(&self.mask, graph);
        let mw = self.mask.width();
        let mh = self.mask.height();
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, mask_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.convolution",
                function: "morph_kernel",
            }),
            params: vec![Param::U32(self.morph as u32), Param::U32(mw), Param::U32(mh)],
            op: self_arc,
            output: ValueKind::Image,
        })
    }

    fn inverse_map(
        &self,
        output_rect: Rect,
        w: u32,
        h: u32,
        _lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, Rect)> {
        let halo = MaskHalo::from_dims(self.mask.width(), self.mask.height());
        let expanded = halo.expand(output_rect, w, h);
        vec![(0, expanded), (1, Rect::new(0, 0, self.mask.width() as i32, self.mask.height() as i32))]
    }
}

// ── Precision enum ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    Integer,
    Float,
    Approximate,
}
impl IntoVipsEnum for Precision {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ConvaOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct ConvaOperation<B: Backend> {
    pub mask: crate::data::image::Image<B>,
    pub layers: Option<i32>,
    pub cluster: Option<i32>,
}

impl<B: Backend> std::fmt::Debug for ConvaOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvaOperation")
            .field("layers", &self.layers)
            .finish()
    }
}

impl<B: Backend> Clone for ConvaOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mask: self.mask.clone(),
            layers: self.layers,
            cluster: self.cluster,
        }
    }
}

impl VipsOperation for ConvaOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"conva\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("mask", self.mask.vips_ptr());
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        if let Some(v) = self.cluster {
            op.set_int("cluster", v);
        }
    }
}

impl GpuOperation for ConvaOperation<crate::backend::gpu::GpuBackend> {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let mask_id = inject_gpu_source(&self.mask, graph);
        let mw = self.mask.width();
        let mh = self.mask.height();
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, mask_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.convolution",
                function: "convolution_kernel",
            }),
            params: vec![Param::U32(mw), Param::U32(mh)],
            op: self_arc,
            output: ValueKind::Image,
        })
    }

    fn inverse_map(
        &self,
        output_rect: Rect,
        w: u32,
        h: u32,
        _lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, Rect)> {
        let halo = MaskHalo::from_dims(self.mask.width(), self.mask.height());
        let expanded = halo.expand(output_rect, w, h);
        vec![(0, expanded), (1, Rect::new(0, 0, self.mask.width() as i32, self.mask.height() as i32))]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ConvfOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct ConvfOperation<B: Backend> {
    pub mask: crate::data::image::Image<B>,
}

impl<B: Backend> std::fmt::Debug for ConvfOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvfOperation").finish()
    }
}

impl<B: Backend> Clone for ConvfOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mask: self.mask.clone(),
        }
    }
}

impl VipsOperation for ConvfOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"convf\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("mask", self.mask.vips_ptr());
    }
}

impl GpuOperation for ConvfOperation<crate::backend::gpu::GpuBackend> {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let mask_id = inject_gpu_source(&self.mask, graph);
        let mw = self.mask.width();
        let mh = self.mask.height();
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, mask_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.convolution",
                function: "convolution_kernel",
            }),
            params: vec![Param::U32(mw), Param::U32(mh)],
            op: self_arc,
            output: ValueKind::Image,
        })
    }

    fn inverse_map(
        &self,
        output_rect: Rect,
        w: u32,
        h: u32,
        _lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, Rect)> {
        let halo = MaskHalo::from_dims(self.mask.width(), self.mask.height());
        let expanded = halo.expand(output_rect, w, h);
        vec![(0, expanded), (1, Rect::new(0, 0, self.mask.width() as i32, self.mask.height() as i32))]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ConviOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct ConviOperation<B: Backend> {
    pub mask: crate::data::image::Image<B>,
}

impl<B: Backend> std::fmt::Debug for ConviOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConviOperation").finish()
    }
}

impl<B: Backend> Clone for ConviOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mask: self.mask.clone(),
        }
    }
}

impl VipsOperation for ConviOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"convi\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("mask", self.mask.vips_ptr());
    }
}

impl GpuOperation for ConviOperation<crate::backend::gpu::GpuBackend> {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let mask_id = inject_gpu_source(&self.mask, graph);
        let mw = self.mask.width();
        let mh = self.mask.height();
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, mask_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.convolution",
                function: "convolution_kernel",
            }),
            params: vec![Param::U32(mw), Param::U32(mh)],
            op: self_arc,
            output: ValueKind::Image,
        })
    }

    fn inverse_map(
        &self,
        output_rect: Rect,
        w: u32,
        h: u32,
        _lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, Rect)> {
        let halo = MaskHalo::from_dims(self.mask.width(), self.mask.height());
        let expanded = halo.expand(output_rect, w, h);
        vec![(0, expanded), (1, Rect::new(0, 0, self.mask.width() as i32, self.mask.height() as i32))]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ConvsepOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct ConvsepOperation<B: Backend> {
    pub mask: crate::data::image::Image<B>,
    pub precision: Option<Precision>,
    pub layers: Option<i32>,
    pub cluster: Option<i32>,
}

impl<B: Backend> std::fmt::Debug for ConvsepOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvsepOperation")
            .field("precision", &self.precision)
            .finish()
    }
}

impl<B: Backend> Clone for ConvsepOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mask: self.mask.clone(),
            precision: self.precision,
            layers: self.layers,
            cluster: self.cluster,
        }
    }
}

impl VipsOperation for ConvsepOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"convsep\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("mask", self.mask.vips_ptr());
        if let Some(v) = self.precision {
            op.set_int("precision", v.into_vips());
        }
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        if let Some(v) = self.cluster {
            op.set_int("cluster", v);
        }
    }
}

impl GpuOperation for ConvsepOperation<crate::backend::gpu::GpuBackend> {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let mask_id = inject_gpu_source(&self.mask, graph);
        let mw = self.mask.width();
        let mh = self.mask.height();
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, mask_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.convolution",
                function: "convolution_kernel",
            }),
            params: vec![Param::U32(mw), Param::U32(mh)],
            op: self_arc,
            output: ValueKind::Image,
        })
    }

    fn inverse_map(
        &self,
        output_rect: Rect,
        w: u32,
        h: u32,
        _lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, Rect)> {
        let halo = MaskHalo::from_dims(self.mask.width(), self.mask.height());
        let expanded = halo.expand(output_rect, w, h);
        vec![(0, expanded), (1, Rect::new(0, 0, self.mask.width() as i32, self.mask.height() as i32))]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ConvasepOperation
// ═══════════════════════════════════════════════════════════════════════════════

pub struct ConvasepOperation<B: Backend> {
    pub mask: crate::data::image::Image<B>,
    pub layers: Option<i32>,
}

impl<B: Backend> std::fmt::Debug for ConvasepOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvasepOperation")
            .field("layers", &self.layers)
            .finish()
    }
}

impl<B: Backend> Clone for ConvasepOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mask: self.mask.clone(),
            layers: self.layers,
        }
    }
}

impl VipsOperation for ConvasepOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"convasep\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("mask", self.mask.vips_ptr());
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
    }
}

impl GpuOperation for ConvasepOperation<crate::backend::gpu::GpuBackend> {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let mask_id = inject_gpu_source(&self.mask, graph);
        let mw = self.mask.width();
        let mh = self.mask.height();
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, mask_id],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.convolution",
                function: "convolution_kernel",
            }),
            params: vec![Param::U32(mw), Param::U32(mh)],
            op: self_arc,
            output: ValueKind::Image,
        })
    }

    fn inverse_map(
        &self,
        output_rect: Rect,
        w: u32,
        h: u32,
        _lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, Rect)> {
        let halo = MaskHalo::from_dims(self.mask.width(), self.mask.height());
        let expanded = halo.expand(output_rect, w, h);
        vec![(0, expanded), (1, Rect::new(0, 0, self.mask.width() as i32, self.mask.height() as i32))]
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FastcorOperation — fast correlation (GPU TODO: frequency-domain operation)
// ═══════════════════════════════════════════════════════════════════════════════

pub struct FastcorOperation<B: Backend> {
    pub reference: crate::data::image::Image<B>,
}

impl<B: Backend> std::fmt::Debug for FastcorOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastcorOperation").finish()
    }
}

impl<B: Backend> Clone for FastcorOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            reference: self.reference.clone(),
        }
    }
}

impl VipsOperation for FastcorOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"fastcor\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("ref", self.reference.vips_ptr());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SpcorOperation — spatial correlation (GPU TODO)
// ═══════════════════════════════════════════════════════════════════════════════

pub struct SpcorOperation<B: Backend> {
    pub reference: crate::data::image::Image<B>,
}

impl<B: Backend> std::fmt::Debug for SpcorOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpcorOperation").finish()
    }
}

impl<B: Backend> Clone for SpcorOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            reference: self.reference.clone(),
        }
    }
}

impl VipsOperation for SpcorOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"spcor\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("ref", self.reference.vips_ptr());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PhasecorOperation — phase correlation (GPU TODO: FFT-based)
// ═══════════════════════════════════════════════════════════════════════════════

pub struct PhasecorOperation<B: Backend> {
    pub second: crate::data::image::Image<B>,
}

impl<B: Backend> std::fmt::Debug for PhasecorOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhasecorOperation").finish()
    }
}

impl<B: Backend> Clone for PhasecorOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            second: self.second.clone(),
        }
    }
}

impl VipsOperation for PhasecorOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"phasecor\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("in2", self.second.vips_ptr());
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FreqmultOperation — frequency domain multiply (GPU TODO: FFT-based)
// ═══════════════════════════════════════════════════════════════════════════════

pub struct FreqmultOperation<B: Backend> {
    pub mask: crate::data::image::Image<B>,
}

impl<B: Backend> std::fmt::Debug for FreqmultOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FreqmultOperation").finish()
    }
}

impl<B: Backend> Clone for FreqmultOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mask: self.mask.clone(),
        }
    }
}

impl VipsOperation for FreqmultOperation<crate::backend::vips::VipsBackend> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"freqmult\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("mask", self.mask.vips_ptr());
    }
}
