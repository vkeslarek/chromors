use super::param::Param;
use super::work_unit::WorkUnit;
use crate::backend::ColorConversionCapability;
use std::fmt::Debug;
use std::sync::Arc;

use super::datatype::{DataType, ImageType};
use super::graph::NodeEval;
use super::graph::{Graph, GraphNode, KernelSpec, NodeId};
use crate::color::space::ColorSpace;
use crate::pixel::AlphaPolicy;
use crate::pixel::PixelFormat;

// ── OutputCodec ───────────────────────────────────────────────────────────────

/// Target color space + pixel format carried by [`OutputDecoder::WorkingEncodeRegion`].
///
/// `None` in the codec field means "inherit from source / plan default".
#[derive(Clone, Debug)]
pub struct OutputCodec {
    pub color_space: ColorSpace,
    pub format: PixelFormat,
}

// ── InputEncoder / OutputDecoder ──────────────────────────────────────────────

/// Slang wrapper the emitter generates around reading one input slot.
///
/// Each variant corresponds 1:1 to a Slang struct type.  DataType-specific:
/// image ops use [`WorkingDecodeRegion`] or [`CodecRegion`]; FFT ops use
/// [`ComplexRegion`]; mask ops use [`MaskRegion`].
///
/// Declared **per INPUT** — `GpuOperation::input_encoders(n)` returns one
/// value per input slot.
///
/// [`WorkingDecodeRegion`]: InputEncoder::WorkingDecodeRegion
/// [`CodecRegion`]: InputEncoder::CodecRegion
/// [`ComplexRegion`]: InputEncoder::ComplexRegion
/// [`MaskRegion`]: InputEncoder::MaskRegion
#[derive(Clone, Debug, PartialEq)]
pub enum InputEncoder {
    /// `WorkingDecodeRegion<CodecRegion<C,CH>>` — decode raw uint → ACEScg float4.
    WorkingDecodeRegion,
    /// `CodecRegion<C,CH>` — raw pixel values, no color conversion.
    CodecRegion,
    /// `ComplexRegion` — `StructuredBuffer<float2>` for FFT complex data.
    ComplexRegion,
    /// `MaskRegion` — `StructuredBuffer<float>` for separable/morphology masks.
    MaskRegion,
}

/// Slang wrapper the emitter generates around writing the node's output.
///
/// Each variant corresponds 1:1 to a Slang struct type.  DataType-specific:
/// image ops use [`WorkingEncodeRegion`]; histogram/reduction ops use
/// [`HistogramOut`]; FFT ops use [`RWComplexRegion`]; mask ops use [`RWMaskRegion`].
///
/// Declared **for the OUTPUT** — `GpuOperation::output_decoder()` returns one
/// value for the node's single output.
///
/// [`WorkingEncodeRegion`]: OutputDecoder::WorkingEncodeRegion
/// [`HistogramOut`]: OutputDecoder::HistogramOut
/// [`RWComplexRegion`]: OutputDecoder::RWComplexRegion
/// [`RWMaskRegion`]: OutputDecoder::RWMaskRegion
#[derive(Clone, Debug)]
pub enum OutputDecoder {
    /// `from_working` + `codec::encode` → uint buffer.
    ///
    /// `codec = None` inherits color space + format from the source / plan.
    WorkingEncodeRegion { codec: Option<OutputCodec> },
    /// `RWRegion` — raw float4 write, no color conversion.
    RWRegion,
    /// `HistogramOut` — atomic-add bin accumulation.
    HistogramOut,
    /// `RWComplexRegion` — `RWStructuredBuffer<float2>` for FFT output.
    RWComplexRegion,
    /// `RWMaskRegion` — `RWStructuredBuffer<float>` for mask output.
    RWMaskRegion,
}

// ── GpuOperation ─────────────────────────────────────────────────────────────

/// A logical operation in the fused GPU graph.
pub trait GpuOperation: Send + Sync + Debug {
    /// Emit this operation into the graph and return the output node id.
    fn emit(&self, inputs: &[NodeId], graph: &mut Graph, self_arc: Arc<dyn GpuOperation>)
    -> NodeId;

    /// Output pixel dimensions.  Default: identity (`Some((input_w, input_h))`).
    /// Return `None` for ops with no spatial output (histograms, reductions).
    fn output_dims(&self, input_w: u32, input_h: u32) -> Option<(u32, u32)> {
        Some((input_w, input_h))
    }

    /// Map an output work-unit to the input work-units this op needs.
    ///
    /// `(index, work_unit)` — `index` is the input slot (0 = primary, 1+ = extras).
    /// Default: single passthrough — input 0 needs the same WU as the output.
    fn input_demands(&self, wu: &WorkUnit) -> Vec<(usize, WorkUnit)> {
        vec![(0, wu.clone())]
    }

    /// Scale `params` for dispatch at `lod`.  Default: no scaling.
    ///
    /// Override for ops whose params represent pixel-space magnitudes (e.g. blur
    /// sigma) that must be divided by `lod.scale_factor()` at reduced LODs.
    fn scale_params_for_lod(&self, params: &[Param], _lod: super::Lod) -> Vec<Param> {
        params.to_vec()
    }

    /// Slang wrapper generated around each input slot when the kernel reads it.
    ///
    /// Index 0 = primary input, 1+ = extras — matches the ordering of `inputs`
    /// in the `GraphNode`.  Default: every slot gets [`InputEncoder::WorkingDecodeRegion`]
    /// (decode raw uint → ACEScg working space).
    fn input_encoders(&self, num_inputs: usize) -> Vec<InputEncoder> {
        vec![InputEncoder::WorkingDecodeRegion; num_inputs]
    }

    /// Slang wrapper generated around writing this node's output.
    ///
    /// Default: [`OutputDecoder::WorkingEncodeRegion`] with `codec = None`
    /// (inherit color space + format from source / plan).  Non-image outputs
    /// (histograms, masks, FFT) override to the appropriate variant.
    fn output_decoder(&self) -> OutputDecoder {
        OutputDecoder::WorkingEncodeRegion { codec: None }
    }

    /// Which rect drives this op's compute-shader thread grid.
    ///
    /// Default `Output` — the kernel writes one result per dispatched thread
    /// at its own output shape.  Reductions (`HistogramOp`, `VectorscopeOp`)
    /// override to `Input(0)`: the thread grid covers the input region being
    /// scanned, not the output's placeholder shape.
    fn dispatch_grid(&self) -> DispatchGrid {
        DispatchGrid::Output
    }
}

/// Statically declares the [`DataType`] an operation's emitted node produces.
///
/// Kept separate from [`GpuOperation`] (which stays object-safe and is stored
/// as `Arc<dyn GpuOperation>` on every [`GraphNode`]) — `Output` varies per
/// concrete op (`ImageType` for most, `HistogramType` for reductions), so it
/// can only be used in generic position. [`super::datatype::Executable::execute`]
/// is the generic entry point that ties `O::Output` to the returned typed handle.
pub trait TypedOperation: GpuOperation {
    type Output: DataType;
}

// ── DispatchGrid ──────────────────────────────────────────────────────────────

/// Declares which node's demanded rect sizes a kernel's thread grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DispatchGrid {
    /// Dispatch over this node's own output rect.
    Output,
    /// Dispatch over input slot `idx`'s demanded rect — used by reductions
    /// whose output has no natural 2-D dispatch shape of its own.
    Input(usize),
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Working-space image datatype shared by ordinary fused image ops — ACEScg
/// linear float4. The final output color space/format is decided later by
/// the layout pass (`output_decoder()` codec override / plan default), not
/// by this tag.
pub fn working_image_type() -> Arc<dyn DataType> {
    Arc::new(ImageType {
        color_space: ColorSpace::ACES_CG,
        format: PixelFormat::RgbaF32,
    })
}

pub fn emit_image(
    graph: &mut Graph,
    input: NodeId,
    op: Arc<dyn GpuOperation>,
    module: &'static str,
    function: &'static str,
    params: Vec<Param>,
) -> NodeId {
    graph.add_node(GraphNode {
        id: NodeId(0),
        inputs: vec![input],
        eval: NodeEval::Kernel(KernelSpec { module, function }),
        params,
        op,
        datatype: working_image_type(),
    })
}

/// Splice another image's subgraph into `graph`, returning the node id
/// (within `graph`) corresponding to `other`'s root.
///
/// Used by multi-input ops (composite, join, insert) to fuse a sibling
/// image's graph lazily instead of eagerly pulling it to host bytes and
/// re-injecting it as a baked source.
///
/// If `other` shares the same underlying graph as `graph` (e.g. compositing
/// an image onto itself), `other.root_id()` is already a valid id in `graph`
/// and no merge is needed — detected via `try_lock`, since re-locking the
/// same `Mutex` from the same thread would deadlock.
pub fn splice_sibling(
    graph: &mut Graph,
    other: &crate::data::image::Image2D<super::GpuBackend>,
) -> NodeId {
    match other.handle.graph.try_lock() {
        Ok(other_graph) => {
            let remap = graph.merge_from(&other_graph);
            remap[&other.root_id()]
        }
        Err(_) => other.root_id(),
    }
}

pub fn emit_unary(
    graph: &mut Graph,
    input: NodeId,
    op: Arc<dyn GpuOperation>,
    module: &'static str,
    function: &'static str,
    params: Vec<Param>,
    datatype: Arc<dyn DataType>,
) -> NodeId {
    graph.add_node(GraphNode {
        id: NodeId(0),
        inputs: vec![input],
        eval: NodeEval::Kernel(KernelSpec { module, function }),
        params,
        op,
        datatype,
    })
}

use super::GpuBackend;
use super::GraphNodeHandle;
use crate::data::image::Image2D;

// ── ColorConversionCapability ─────────────────────────────────────────────────

use crate::pixel::PixelMeta;

impl ColorConversionCapability for GpuBackend {
    /// Construct the `PixelMeta` (format + color space) `handle`'s root
    /// node will produce on pull, derived from the graph (see
    /// [`Graph::resolve_image_type`]). Alpha policy is always `Straight` at
    /// the handle boundary — premultiplication is handled inside the GPU
    /// working-space pipeline. Panics if the root node has no image output.
    fn pixel_meta(handle: &GraphNodeHandle) -> PixelMeta {
        let g = handle.graph.lock().unwrap();
        let it = g
            .resolve_image_type(handle.root_id)
            .expect("GpuBackend::pixel_meta: root node has no image output");
        PixelMeta {
            format: it.format,
            color_space: it.color_space,
            alpha_policy: AlphaPolicy::Straight,
        }
    }

    fn convert(
        handle: &GraphNodeHandle,
        target: PixelMeta,
    ) -> Result<GraphNodeHandle, crate::Error> {
        let _sw = crate::utils::Stopwatch::new("gpu.convert");
        let img = Image2D::<GpuBackend>::from_handle(handle.clone());
        // Emit a passthrough kernel with the target codec so the materialiser
        // applies the correct color-space decode on output.
        let converted = img.execute(&GpuColorConvertOperation { dst: target })?;
        Ok(converted.handle)
    }
}

/// Internal GPU color-conversion node.  Not exposed as a public Operation —
/// use `Image2D<GpuBackend>::convert(meta)` instead.
#[derive(Debug, Clone)]
pub(crate) struct GpuColorConvertOperation {
    pub dst: PixelMeta,
}

impl TypedOperation for GpuColorConvertOperation {
    type Output = ImageType;
}

impl GpuOperation for GpuColorConvertOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.passthrough",
                function: "passthrough_kernel",
            }),
            params: vec![],
            op: self_arc,
            datatype: Arc::new(ImageType {
                color_space: self.dst.color_space,
                format: self.dst.format,
            }),
        })
    }

    fn output_decoder(&self) -> OutputDecoder {
        OutputDecoder::WorkingEncodeRegion {
            codec: Some(OutputCodec {
                color_space: self.dst.color_space,
                format: self.dst.format,
            }),
        }
    }
}
