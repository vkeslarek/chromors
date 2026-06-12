pub mod context;
pub mod buffer;
pub mod view;
pub mod emit;
pub mod compile;
pub mod materialize;
pub mod slang;

pub use context::*;
pub use buffer::*;
pub use view::*;

use std::collections::HashMap;
use std::sync::Arc;
use crate::error::Error;
use crate::kind::Kind;
use crate::node::Node;
use crate::work_unit::WorkUnit;

pub struct GpuBackend;

/// Where a kernel step reads one of its arguments from.
#[derive(Clone, Copy, Debug)]
pub enum StepInput {
    /// A source leaf's decoded buffer (`in_{i}`, a `CodecRegion`).
    Source(usize),
    /// A prior step's working temp (`work_{j}`, read as an `RWRegion`).
    Step(usize),
    /// A source leaf's decoded buffer, read through one swizzled component
    /// and broadcast `float4(v,v,v,1)` via `SwizzleView`. Produced by
    /// `GpuBuilder::alias` (e.g. a "free" `ExtractBand` on a freshly-opened
    /// image, with no prior kernel step to alias).
    SwizzleSource(usize, u32),
    /// A prior step's working temp, read through one swizzled component
    /// (`work_{j}[channel]`, broadcast `float4(v,v,v,1)` via `SwizzleView`).
    /// Produced by `GpuBuilder::alias` (e.g. a "free" `ExtractBand`).
    SwizzleStep(usize, u32),
    RemapSource(usize, RemapKind, RemapParams),
    RemapStep(usize, RemapKind, RemapParams),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemapKind {
    Identity = 0,
    FlipH = 1,
    FlipV = 2,
    Rot180 = 3,
    Scale = 4,
    Tile = 5,
    Translate = 6,
    Rot90 = 7,
    Rot270 = 8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RemapParams {
    pub out_w: u32,
    pub out_h: u32,
    pub sx: f32,
    pub sy: f32,
    pub in_w: u32,
    pub in_h: u32,
    pub tx: i32,
    pub ty: i32,
}

impl Default for RemapParams {
    fn default() -> Self {
        Self { out_w: 0, out_h: 0, sx: 1.0, sy: 1.0, in_w: 0, in_h: 0, tx: 0, ty: 0 }
    }
}

/// Element type of a step's working temp buffer (`work_{s}`) — a data-driven
/// descriptor in the same style as `View`/`OutputWrap`, not a closed enum.
/// Determines the temp's Slang buffer type and which `IRegion` wrapper
/// structs (plain + swizzled) read it. Adding a new temp shape (e.g. a
/// point-list step writing `float2`) is just a new `TempElem` constant +
/// matching wrapper structs in `lib/region.slang` — `GpuBuilder`/`emit.rs`
/// stay generic, no match arms to extend.
#[derive(Clone, Copy, Debug)]
pub struct TempElem {
    /// Slang element type for the `work_{k}` buffer declaration (e.g. `"float4"`).
    pub buffer_ty: &'static str,
    /// Wrapper struct name for a plain (whole-value) read of this temp
    /// (e.g. `"RWRegion"`). Also the `R` of `SwizzleView<R>` for
    /// `StepInput::SwizzleStep` — `SwizzleView` is generic over any
    /// `IRegion`, so no separate swizzle-wrapper name is needed.
    pub region_wrapper: &'static str,
    /// Field accessor per component (e.g. `["x","y","z","w"]`).
    pub components: &'static [&'static str],
    /// Broadcast template for the final encode when the DAG root is an
    /// `alias`; `{v}` is replaced with the scalar expression (e.g.
    /// `"float4({v}, {v}, {v}, 1.0)"`).
    pub broadcast: &'static str,
}

impl TempElem {
    /// The image working-pixel temp: `RWStructuredBuffer<float4>`, read via
    /// `RWRegion` (plain) or `SwizzleView<RWRegion>` (single component,
    /// broadcast `float4(v,v,v,1)`).
    pub const F4: TempElem = TempElem {
        buffer_ty: "float4",
        region_wrapper: "RWRegion",
        components: &["x", "y", "z", "w"],
        broadcast: "float4({v}, {v}, {v}, 1.0)",
    };

    /// Field accessor for swizzled component `c` of this temp's value.
    pub fn component(&self, c: u32) -> &'static str {
        self.components[c as usize]
    }
    /// Broadcast a scalar expression back into this temp's full shape.
    pub fn broadcast_expr(&self, scalar_expr: &str) -> String {
        self.broadcast.replace("{v}", scalar_expr)
    }
}

impl Default for TempElem {
    fn default() -> Self {
        TempElem::F4
    }
}

/// One kernel invocation in the fused pass. Steps are emitted in topo order;
/// each writes its own working temp, so a node reachable by several consumers
/// (a diamond) is computed once and read by index — exactly the old engine's
/// per-node-temp model.
pub struct Step {
    pub kernel: &'static str,
    pub inputs: Vec<StepInput>,
    pub params: Vec<String>,
    /// Element type of this step's `work_{s}` temp (see `TempElem`).
    pub temp_elem: TempElem,
}

pub struct GpuBuilder {
    /// Decode views for each source input, in binding order (== `source_buffers`).
    pub input_views: Vec<View>,
    /// Kernel steps, topo order. Each writes its own temp (`work_{step}`).
    pub steps: Vec<Step>,
    /// How the final output is written, as declared by its Kind.
    pub output: Option<OutputWrap>,
    pub source_buffers: Vec<Arc<crate::backend::gpu::buffer::GpuBuffer>>,
    pub params: ParamBlock,

    /// node-pointer → source slot index (set when a Source leaf lowers).
    source_of: HashMap<usize, usize>,
    /// node-pointer → its final step index (set as an op's kernels lower).
    last_step_of: HashMap<usize, usize>,
    /// node-pointer → (base input, swizzle component), set by `alias`. A
    /// downstream consumer resolving this node as an input gets
    /// `SwizzleSource`/`SwizzleStep` instead of `Source`/`Step`.
    alias_swizzles: HashMap<usize, (StepInput, u32)>,
    alias_remaps: HashMap<usize, (StepInput, RemapKind, RemapParams)>,
    /// The node currently lowering, and its resolved input edges — so the
    /// node's *first* kernel reads its graph inputs and later kernels chain.
    cur_node: Option<usize>,
    cur_inputs: Vec<StepInput>,
    cur_started: bool,

    /// Scalar field names registered via `param_block` since the last
    /// `kernel()` call. Drained into the next step's `params` (its trailing
    /// kernel-call args, in declaration order) — lets ops call `param_block`
    /// before `kernel`, matching the existing convention everywhere.
    pending_params: Vec<String>,

    /// Set by `alias` when the *current* (possibly terminal) node is a pure
    /// swizzle of its input. If this node turns out to be the root, the
    /// emitter broadcasts this component through the final encode instead of
    /// the full working pixel. Cleared at the start of every `enter`.
    output_swizzle: Option<u32>,

    ctx: Arc<GpuContext>,
    error: Option<Error>,
    current_wu: Option<WorkUnit>,
}

impl GpuBuilder {
    pub fn new(ctx: Arc<GpuContext>) -> Self {
        Self {
            input_views: Vec::new(),
            steps: Vec::new(),
            output: None,
            source_buffers: Vec::new(),
            params: ParamBlock::default(),
            source_of: HashMap::new(),
            last_step_of: HashMap::new(),
            alias_swizzles: HashMap::new(),
            alias_remaps: HashMap::new(),
            cur_node: None,
            cur_inputs: Vec::new(),
            cur_started: false,
            pending_params: Vec::new(),
            output_swizzle: None,
            ctx,
            error: None,
            current_wu: None,
        }
    }

    /// Materializer hook: announce the node about to be lowered — its pointer
    /// identity, the identities of its input nodes (resolved to source slots or
    /// prior steps), and its resolved WorkUnit.
    pub fn enter(&mut self, node_key: usize, input_keys: &[usize], wu: WorkUnit) {
        self.current_wu = Some(wu);
        self.cur_node = Some(node_key);
        self.cur_started = false;
        self.output_swizzle = None;
        self.cur_inputs = input_keys
            .iter()
            .map(|k| {
                if let Some(&(base, c)) = self.alias_swizzles.get(k) {
                    match base {
                        StepInput::Source(i) => StepInput::SwizzleSource(i, c),
                        StepInput::Step(j) => StepInput::SwizzleStep(j, c),
                        // Aliasing an alias: the broadcast value is uniform
                        // across components, so the inner swizzle is reused
                        sw @ (StepInput::SwizzleSource(..) | StepInput::SwizzleStep(..) | StepInput::RemapSource(..) | StepInput::RemapStep(..)) => sw,
                    }
                } else if let Some(&(base, kind, params)) = self.alias_remaps.get(k) {
                    match base {
                        StepInput::Source(i) => StepInput::RemapSource(i, kind, params),
                        StepInput::Step(j) => StepInput::RemapStep(j, kind, params),
                        sw => sw, // ignoring nested aliases for now
                    }
                } else if let Some(&si) = self.source_of.get(k) {
                    StepInput::Source(si)
                } else if let Some(&st) = self.last_step_of.get(k) {
                    StepInput::Step(st)
                } else {
                    // Pruned/absent input reached a consumer — shouldn't happen
                    // for a live node; fall back to source 0 to stay total.
                    StepInput::Source(0)
                }
            })
            .collect();
    }

    pub fn ctx(&self) -> &Arc<GpuContext> {
        &self.ctx
    }
    pub fn fail(&mut self, e: Error) {
        if self.error.is_none() {
            self.error = Some(e);
        }
    }
    pub fn take_error(&mut self) -> Option<Error> {
        self.error.take()
    }
    pub fn wu(&self) -> &WorkUnit {
        self.current_wu.as_ref().expect("GpuBuilder::wu called outside a lower()")
    }

    /// Register a **source input**: decode `View`, geometry, fetched buffer.
    /// Called by a Source leaf's `lower`; binds this node to a source slot.
    pub fn input(&mut self, view: View, region: RegionParams, buf: Arc<crate::backend::gpu::buffer::GpuBuffer>) -> &mut Self {
        let slot = self.input_views.len();
        if let Some(k) = self.cur_node {
            self.source_of.insert(k, slot);
        }
        region.push_into(&mut self.params, &format!("region_in_{slot}"));
        self.input_views.push(view);
        self.source_buffers.push(buf);
        self
    }

    /// Add a kernel step to the current node. Its first kernel reads the node's
    /// graph inputs; any later kernel (intra-node multistep) reads the step
    /// before it. The step's output temp is its own index.
    pub fn kernel(&mut self, entry: &'static str) -> &mut Self {
        let inputs = if !self.cur_started {
            self.cur_started = true;
            self.cur_inputs.clone()
        } else {
            vec![StepInput::Step(self.steps.len() - 1)]
        };
        let params = std::mem::take(&mut self.pending_params);
        self.steps.push(Step { kernel: entry, inputs, params, temp_elem: TempElem::F4 });
        let idx = self.steps.len() - 1;
        if let Some(k) = self.cur_node {
            self.last_step_of.insert(k, idx);
        }
        self
    }

    /// Make the current node a **zero-cost view** of its single input: no
    /// kernel step is added. The node's value becomes its input's value
    /// (a source decode or a prior step's temp), read through component
    /// `channel` (0=x/r .. 3=w/a) and broadcast as `float4(v,v,v,1)` — the
    /// same shape a Gray codec decode produces. A downstream consumer of this
    /// node gets `StepInput::SwizzleSource`/`SwizzleStep`; if this node is
    /// the DAG root, the final encode broadcasts `channel` instead of the
    /// full working pixel (see `output_swizzle`).
    ///
    /// Used by `ExtractBand` — extracting a band is then free: it never adds
    /// a pass, it just changes how the *next* kernel (or the encoder) reads
    /// the value that's already there. Works whether the input is a fresh
    /// source decode or a prior step's temp.
    pub fn alias(&mut self, channel: u32) -> &mut Self {
        let Some(&input) = self.cur_inputs.first() else {
            self.fail(Error::Backend("GpuBuilder::alias: node has no input to alias".into()));
            return self;
        };
        if let Some(k) = self.cur_node {
            self.alias_swizzles.insert(k, (input, channel));
        }
        self.output_swizzle = Some(channel);
        self
    }

    pub fn remap(&mut self, kind: RemapKind, params: RemapParams) -> &mut Self {
        let Some(&input) = self.cur_inputs.first() else {
            self.fail(Error::Backend("GpuBuilder::remap: node has no input to remap".into()));
            return self;
        };
        if let Some(k) = self.cur_node {
            self.alias_remaps.insert(k, (input, kind, params));
        }
        self
    }

    /// A scalar param for the current step. Namespaced by step index so two
    /// steps (or two nodes) using the same name never collide in `ChainParams`.
    pub fn param<T: bytemuck::Pod>(&mut self, name: &str, value: T) -> &mut Self {
        let idx = self.steps.len() - 1;
        let field = format!("s{idx}_{name}");
        self.params.fields.push((field.clone(), "scalar"));
        self.params.bytes.extend_from_slice(bytemuck::bytes_of(&value));
        self.steps[idx].params.push(field);
        self
    }

    /// Register the output, as described by its **Kind** (`GpuView::output`).
    pub fn output(&mut self, wrap: OutputWrap) -> &mut Self {
        if let Some(WorkUnit::Region(r)) = &self.current_wu {
            RegionParams::tight(r.w, r.h).push_into(&mut self.params, "region_out");
        }
        self.output = Some(wrap);
        self
    }

    /// Merge a whole `ParamBlock` (e.g. a reduction's `bin_count`) into the
    /// shared `ChainParams`.
    pub fn param_block(&mut self, block: ParamBlock) -> &mut Self {
        self.pending_params.extend(block.fields.iter().map(|(name, _)| name.clone()));
        self.params.fields.extend(block.fields);
        self.params.bytes.extend_from_slice(&block.bytes);
        self
    }

    /// Merge a `ParamBlock` into `ChainParams` without queuing its fields as
    /// trailing kernel-call args. For fields consumed only by the output
    /// wrapper's ctor (e.g. a reduction's `bin_count`), not by the kernel body.
    pub fn output_params(&mut self, block: ParamBlock) -> &mut Self {
        self.params.fields.extend(block.fields);
        self.params.bytes.extend_from_slice(&block.bytes);
        self
    }

    /// Does the output use the codec sandwich (image) vs a direct write?
    pub fn needs_scratch(&self) -> bool {
        matches!(&self.output, Some(w) if w.arg_buffer == OutBuffer::Scratch)
    }

    /// `float4` working temps to bind: one per step that writes a temp. With a
    /// direct (atomic) output the final step writes the target instead.
    pub fn work_buffer_count(&self) -> usize {
        if self.needs_scratch() {
            self.steps.len()
        } else {
            self.steps.len().saturating_sub(1)
        }
    }
}

/// A Kind's GPU lowering capability: it declares how its data is **decoded** at
/// a shader input and **encoded** at a shader output (its codec sandwich lives
/// here, not in the emitter or the ops). An op only contributes its kernel.
pub trait GpuView: Kind {
    /// Wrapper a kernel reads this Kind through at an input slot (decodes).
    fn input(&self) -> View;
    /// How an output of this Kind is written (see [`OutputWrap`]).
    fn output(&self) -> OutputWrap;
    /// Extra wrapper params beyond the universal per-slot `BufferRegion`
    /// (e.g. a histogram's `bin_count`). Default: none.
    fn params(&self, _wu: &WorkUnit) -> ParamBlock {
        ParamBlock::empty()
    }
}

impl crate::backend::Backend for GpuBackend {
    type Ctx = GpuContext;
    type Payload = GpuBuffer;
    type Builder = GpuBuilder;

    fn materialize(
        ctx: &Arc<Self::Ctx>,
        root: &Arc<Node<Self>>,
        wu: &WorkUnit,
    ) -> Result<crate::buffer::Buffer<Self>, Error> {
        let materializer = materialize::Materializer { ctx, root };
        materializer.execute(wu)
    }
}
