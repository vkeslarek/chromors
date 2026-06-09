use super::param::Param;
use crate::backend::ColorConversionCapability;
use std::fmt::Debug;
use std::sync::Arc;

use super::graph::NodeEval;
use super::graph::{Graph, GraphNode, KernelSpec, NodeId};
use super::value::ValueKind;
use crate::color::space::ColorSpace;
use crate::geometry::Rect;
use crate::pixel::PixelFormat;

// ── OutputSpec ────────────────────────────────────────────────────────────────

/// Describes what kind of output an operation produces and its dimensions.
///
/// `output_spec(input_w, input_h)` replaces the old `output_size()` +
/// `output_kind()` + `output_capacity_hint()` trinity.
#[derive(Clone, Debug, PartialEq)]
pub enum OutputSpec {
    /// A 2-D pixel image with the given output dimensions.
    Image { width: u32, height: u32 },
    /// Fixed-size histogram accumulator (`bins` uint atomics).
    Histogram { bins: u32 },
    /// Atomic-append coordinate list.
    PointList { capacity: u32 },
    /// Single float scalar.
    Scalar,
    /// Multi-channel feature map.
    FeatureMap {
        channels: u32,
        width: u32,
        height: u32,
    },
}

impl OutputSpec {
    /// Output pixel dimensions, if this spec produces an image.
    pub fn image_dims(&self) -> Option<(u32, u32)> {
        match self {
            OutputSpec::Image { width, height } => Some((*width, *height)),
            OutputSpec::FeatureMap { width, height, .. } => Some((*width, *height)),
            _ => None,
        }
    }

    pub fn to_value_kind(&self) -> ValueKind {
        match self {
            OutputSpec::Image { .. } => ValueKind::Image,
            OutputSpec::Histogram { bins } => ValueKind::Histogram { bins: *bins },
            OutputSpec::PointList { capacity } => ValueKind::PointList {
                capacity: *capacity,
            },
            OutputSpec::Scalar => ValueKind::Scalar,
            OutputSpec::FeatureMap { channels, .. } => ValueKind::Features {
                channels: *channels,
            },
        }
    }
}

// ── OutputCodec ───────────────────────────────────────────────────────────────

/// Output color space + format override for a graph node.
///
/// When set, the emitter uses this instead of the source color space / default
/// `RgbaF32` format to generate the `dst_cs` shader constant and to plan the
/// final `from_working → codec::encode` step.
///
/// Replaces the old `dst_meta: Option<PixelMeta>` field on `GraphNode`.
#[derive(Clone, Debug)]
pub struct OutputCodec {
    pub color_space: ColorSpace,
    pub format: PixelFormat,
}

// ── WorkUnit ────────────────────────────────────────────────────────────────

/// The division strategy a [`ValueKind`] declares for how its output can be
/// split into independently-fetchable/cacheable/dispatchable chunks — and the
/// wire format `input_demands` uses to propagate "what do I need from you"
/// across the graph between nodes of (possibly different) DataTypes.
///
/// Each variant corresponds to a family of DataTypes' natural shape:
/// `Region` — Image, Mask2D, Fft2D (2-D grids that subdivide into rects);
/// `Atomic` — Histogram, VectorScope, Scalar (indivisible: the only unit is
/// the whole result, there's no meaningful sub-piece). `Range` for 1-D types
/// (Mask1D, Fft1D) and `Frame`/`FrameFragment` for video join this catalog
/// once an operation actually needs to express demand in those shapes.
///
/// Stays a closed enum (not a per-DataType associated type) because nodes are
/// stored as `Arc<dyn GpuOperation>` in a flat heterogeneous graph — demand has
/// to cross DataType boundaries dynamically, so both sides need a shared
/// vocabulary to translate through (same reason `Param` is `I32|U32|F32`
/// rather than generic).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkUnit {
    /// 2-D sub-rectangle — the division strategy of Image, Mask2D, Fft2D.
    Region(Rect),
    /// No subdivision exists — the division strategy of Histogram, VectorScope,
    /// Scalar. The only unit is the entire result.
    Atomic,
}

impl WorkUnit {
    /// Resolve to a concrete bounding rect against a node's full output
    /// dimensions. `Atomic` becomes the full output rect (its one and only
    /// unit); `Region` passes through unchanged. Used wherever a concrete
    /// `Rect` is still required (source fetches, kernel dispatch sizing) —
    /// today every `ValueKind` that can appear mid-graph is region-shaped at
    /// the storage level, even when its *demand* semantics are `Atomic`.
    pub fn resolve(&self, w: u32, h: u32) -> Rect {
        match self {
            WorkUnit::Region(r) => *r,
            WorkUnit::Atomic => Rect::new(0, 0, w as i32, h as i32),
        }
    }
}

// ── GpuOperation ─────────────────────────────────────────────────────────────

/// A logical image operation in the fused GPU graph.
pub trait GpuOperation: Send + Sync + Debug {
    /// Emit this operation into the graph and return the output node id.
    ///
    /// `self_arc` must be stored on the leaf node so `inverse_map` is
    /// reachable during the materialize walk.
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId;

    /// Declare what this operation produces and its output dimensions.
    /// Default: identity image (same dims as input).
    fn output_spec(&self, input_w: u32, input_h: u32) -> OutputSpec {
        OutputSpec::Image {
            width: input_w,
            height: input_h,
        }
    }

    /// Given an output rect, return which input rects this op needs.
    /// Returns `(input_index, rect)` — index 0 = primary, 1+ = extras.
    /// `lod` is the level-of-detail being materialised; ops with spatially
    /// dependent kernels (e.g. blur) should scale their halo by `1/lod.scale_factor()`.
    fn inverse_map(
        &self,
        output_rect: Rect,
        _w: u32,
        _h: u32,
        _lod: super::Lod,
    ) -> Vec<(usize, Rect)> {
        vec![(0, output_rect)]
    }

    /// Given an output demand, return what input demands this op needs.
    /// Image ops map `Region→Region` (with halo, via `inverse_map`); reductions
    /// (histogram, vectorscope) override this directly to always demand `Atomic`
    /// regardless of what the consumer asked for — their division strategy has
    /// no sub-pieces. Default: `Region` delegates to `inverse_map`; `Atomic`
    /// resolves to the full input rect (no halo needed — a full-bounds rect
    /// already saturates any halo expansion via clamping).
    fn input_demands(
        &self,
        out: &WorkUnit,
        w: u32,
        h: u32,
        lod: super::Lod,
    ) -> Vec<(usize, WorkUnit)> {
        match out {
            WorkUnit::Region(r) => self
                .inverse_map(*r, w, h, lod)
                .into_iter()
                .map(|(i, r)| (i, WorkUnit::Region(r)))
                .collect(),
            WorkUnit::Atomic => {
                let s = lod.scale_factor();
                let full = Rect::new(
                    0,
                    0,
                    (w as f64 / s).ceil() as i32,
                    (h as f64 / s).ceil() as i32,
                );
                vec![(0, WorkUnit::Region(full))]
            }
        }
    }

    /// Indices (0-based within this op's `params` list) of parameters that
    /// represent pixel-space magnitudes and must be divided by `lod.scale_factor()`
    /// before GPU dispatch when `lod > 0`.  Default: no such params.
    fn lod_scale_param_indices(&self) -> &'static [usize] {
        &[]
    }

    /// Override the output color space and pixel format for this node's result.
    ///
    /// The emitter uses this to generate the correct `dst_cs` shader constant
    /// and the final `from_working → codec::encode` step.  Return `None` (the
    /// default) to inherit the source color space and use `RgbaF32`.
    ///
    /// Only `ColorConvertOp` needs to override this.  All other ops leave it `None`.
    fn output_codec_override(&self) -> Option<OutputCodec> {
        None
    }

    /// Per-input-slot wrap the emitter generates around reading a source/temp
    /// (index 0 = primary, 1+ = extras — same ordering as `inputs`/`inverse_map`).
    ///
    /// Default: every slot gets `WorkingSpace` — the current sandwich behavior
    /// (`WorkingDecodeRegion<CodecRegion<...>>`, decode to ACEScg/sRGB-linear
    /// f32). Non-color-bearing ops (histogram, mask, FFT, point-list passthrough)
    /// override per slot to `Passthrough` — read raw encoded bytes, no transform,
    /// no wasted bandwidth on a color conversion the kernel doesn't use.
    fn input_decoders(&self, num_inputs: usize) -> Vec<Decoder> {
        vec![Decoder::WorkingSpace; num_inputs]
    }

    /// Wrap the emitter generates around writing this node's output.
    ///
    /// Default: `WorkingSpace` — the current `from_working` + `codec::encode`
    /// sandwich, using `output_codec_override()` if set. Non-image outputs
    /// (histogram bins, scalars, raw masks) override to `Passthrough` — write
    /// the kernel's raw result directly, no encode step.
    fn output_encoder(&self) -> Encoder {
        Encoder::WorkingSpace {
            codec: self.output_codec_override(),
        }
    }

    /// Which rect drives this op's compute-shader thread grid.
    ///
    /// Default `Output` — the kernel writes one result per dispatched thread
    /// at its own output shape (every pixel-wise op, and any future op that
    /// generates Mask/FFT data directly at its declared dimensions).
    ///
    /// Reductions (`HistogramOp`, `VectorscopeOp`) override to `Input(0)`:
    /// the kernel scans an *input* region and folds it atomically into an
    /// indivisible result — the thread grid must cover the region being
    /// scanned, not the output's placeholder shape (`Rect(0, 0, bins, 1)`).
    /// Mirrors `Decoder`/`Encoder`/`WorkUnit` — closed catalog, op-declared,
    /// because the emitter needs a fixed vocabulary to pick codegen by.
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

// ── Decoder / Encoder ─────────────────────────────────────────────────────────

/// Slang wrap the emitter generates around reading an input edge.
///
/// A closed, finite catalog — like [`Param`] / [`WorkUnit`] — because the
/// emitter pattern-matches each variant to a fixed Slang wrap template at
/// codegen time. New variants (e.g. an angular-coordinate decode for point
/// lists) are added here when an op actually needs one, not speculatively.
#[derive(Clone, Debug, PartialEq)]
pub enum Decoder {
    /// No wrap — kernel reads the raw encoded bytes (`CodecRegion<...>`) as-is.
    Passthrough,
    /// `WorkingDecodeRegion<CodecRegion<...>>` — decode + color-convert to
    /// ACEScg/sRGB-linear `float4` before the kernel sees it. Today's default.
    WorkingSpace,
}

/// Slang wrap the emitter generates around writing a node's output.
#[derive(Clone, Debug)]
pub enum Encoder {
    /// No wrap — the kernel's raw result is written directly to its target
    /// (histogram bins, scalars, raw mask/FFT data).
    Passthrough,
    /// `from_working` + `codec::encode` — convert from working space back to
    /// the destination color space/format. Today's default for image outputs.
    WorkingSpace { codec: Option<OutputCodec> },
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

    fn output_codec_override(&self) -> Option<crate::backend::gpu::op::OutputCodec> {
        Some(crate::backend::gpu::op::OutputCodec {
            color_space: self.dst.color_space,
            format: self.dst.format,
        })
    }
}
