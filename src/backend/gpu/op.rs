use super::param::Param;
use super::work_unit::WorkUnit;
use crate::backend::ColorConversionCapability;
use std::fmt::Debug;
use std::sync::Arc;

use super::graph::NodeEval;
use super::graph::{Graph, GraphNode, KernelSpec, NodeId};
use super::value::ValueKind;
use crate::color::space::ColorSpace;
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
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId;

    /// The [`ValueKind`] this operation produces.  Default: `ValueKind::Image`.
    fn output_kind(&self, _input_w: u32, _input_h: u32) -> ValueKind {
        ValueKind::Image
    }

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
        output: ValueKind::Image,
    })
}

pub fn emit_unary(
    graph: &mut Graph,
    input: NodeId,
    op: Arc<dyn GpuOperation>,
    module: &'static str,
    function: &'static str,
    params: Vec<Param>,
    output: ValueKind,
) -> NodeId {
    graph.add_node(GraphNode {
        id: NodeId(0),
        inputs: vec![input],
        eval: NodeEval::Kernel(KernelSpec { module, function }),
        params,
        op,
        output,
    })
}

use super::GpuBackend;
use super::GpuImageHandle;
use crate::data::image::Image;

// ── ColorConversionCapability ─────────────────────────────────────────────────

use crate::pixel::PixelMeta;

impl ColorConversionCapability for GpuBackend {
    fn pixel_meta(handle: &GpuImageHandle) -> PixelMeta {
        handle.pixel_meta()
    }

    fn convert(handle: &GpuImageHandle, target: PixelMeta) -> Result<GpuImageHandle, crate::Error> {
        let _sw = crate::utils::Stopwatch::new("gpu.convert");
        let img = Image::<GpuBackend>::from_handle(handle.clone());
        // Emit a passthrough kernel with the target codec so the materialiser
        // applies the correct color-space decode on output.
        let converted = img.execute(&GpuColorConvertOperation { dst: target })?;
        Ok(converted.handle)
    }
}

/// Internal GPU color-conversion node.  Not exposed as a public Operation —
/// use `Image<GpuBackend>::convert(meta)` instead.
#[derive(Debug, Clone)]
pub(crate) struct GpuColorConvertOperation {
    pub dst: PixelMeta,
}

impl GpuOperation for GpuColorConvertOperation {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input],
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.passthrough",
                function: "passthrough_kernel",
            }),
            params: vec![],
            op: self_arc,
            output: ValueKind::Image,
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
