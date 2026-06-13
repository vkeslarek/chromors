pub mod context;
pub mod buffer;
pub mod view;
pub mod emit;
pub mod compile;
pub mod slang;
pub mod pass;

pub use context::*;
pub use buffer::*;
pub use view::*;

use std::collections::HashMap;
use std::sync::Arc;
use crate::backend::Builder;
use crate::error::Error;
use crate::kind::{AnyKind, Kind};
use crate::node::NodeId;
use crate::work_unit::{Region, WorkUnit, WorkUnitFor};

pub struct GpuBackend;

/// Which producer a [`StepInput`] reads from, before any adapter wrapping.
#[derive(Clone, Copy, Debug)]
pub enum BaseInput {
    /// A source leaf's decoded buffer (`in_{i}`, a `CodecRegion`).
    Source(usize),
    /// A prior step's working temp (`work_{j}`, read as an `RWRegion`).
    Step(usize),
}

/// Where a kernel step reads one of its arguments from: a base producer,
/// optionally wrapped in a zero-cost [`ViewAdapter`] (swizzle, remap, …).
/// Produced by [`GpuBuilder::adapt`] for "free" view-only nodes (e.g.
/// `ExtractBand`, `Flip`) that add no kernel step of their own.
#[derive(Clone, Debug)]
pub struct StepInput {
    pub base: BaseInput,
    pub adapter: Option<ViewAdapter>,
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
    /// (e.g. `"RWRegion"`). Also the `R` of any generic `IRegion`-wrapping
    /// [`ViewAdapter`] applied to this temp.
    pub region_wrapper: &'static str,
    /// Bytes per element, used to size the `work_{k}` buffer as
    /// `domain_w * domain_h * byte_size`.
    pub byte_size: u64,
}

impl TempElem {
    /// The image working-pixel temp: `RWStructuredBuffer<float4>`, read via
    /// `RWRegion`.
    pub const F4: TempElem = TempElem {
        buffer_ty: "float4",
        region_wrapper: "RWRegion",
        byte_size: 16,
    };
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
    /// Slang module defining `kernel` (e.g. `"ops.invert"`), imported by the emitter.
    pub module: &'static str,
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

    /// node id → source slot index (set when a Source leaf lowers).
    source_of: HashMap<NodeId, usize>,
    /// node id → its final step index (set as an op's kernels lower).
    last_step_of: HashMap<NodeId, usize>,
    /// node id → its resolved `StepInput`, set by `adapt`. A downstream
    /// consumer resolving this node as an input gets this adapted view
    /// instead of a plain `Source`/`Step`.
    alias_adapters: HashMap<NodeId, StepInput>,
    /// The node currently lowering, and its resolved input edges — so the
    /// node's *first* kernel reads its graph inputs and later kernels chain.
    cur_node: Option<NodeId>,
    cur_inputs: Vec<StepInput>,
    cur_started: bool,

    /// Scalar field names registered via `param_block` since the last
    /// `kernel()` call. Drained into the next step's `params` (its trailing
    /// kernel-call args, in declaration order) — lets ops call `param_block`
    /// before `kernel`, matching the existing convention everywhere.
    pending_params: Vec<String>,

    /// Set by `adapt` when the *current* (possibly terminal) node is a pure
    /// view-adapted alias of its input. If this node turns out to be the
    /// root, the emitter reads/encodes through this adapter directly instead
    /// of the full working pixel. Cleared at the start of every `enter`.
    cur_output_adapter: Option<StepInput>,

    /// Count of distinct adapters resolved so far this pass — used to assign
    /// each adapter a unique `ChainParams` field prefix (`a{n}_...`).
    adapter_count: usize,

    ctx: Arc<GpuContext>,
    error: Option<Error>,
    current_wu: Option<WorkUnit>,

    /// The dispatch domain (`numthreads` grid, in pixels/elements) — set
    /// explicitly via [`GpuBuilder::dispatch`], or defaulted from each
    /// `output()` call's `Region` WorkUnit when not pinned. `None` only for
    /// an Atomic output whose op never set it (falls back to `(1, 1)`).
    dispatch: Option<(u32, u32)>,
    /// Set by [`GpuBuilder::dispatch`] — once true, `output()` no longer
    /// overwrites `dispatch` (a reduction op's explicit input-sized domain
    /// wins over any leaf source's region-derived default).
    dispatch_explicit: bool,
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
            alias_adapters: HashMap::new(),
            cur_node: None,
            cur_inputs: Vec::new(),
            cur_started: false,
            pending_params: Vec::new(),
            cur_output_adapter: None,
            adapter_count: 0,
            ctx,
            error: None,
            current_wu: None,
            dispatch: None,
            dispatch_explicit: false,
        }
    }

    /// Explicitly set the dispatch domain (in pixels/elements). Reduction ops
    /// (histogram, vectorscope) call this with their *input* image's dims —
    /// their own output is `Atomic`-shaped, so `output()` can't derive it.
    pub fn dispatch(&mut self, dims: (u32, u32)) -> &mut Self {
        self.dispatch = Some(dims);
        self.dispatch_explicit = true;
        self
    }

    /// Remove any existing `params` fields with the given names (and their
    /// matching bytes/sizes), preserving the relative order of the rest. Used
    /// by [`GpuBuilder::output`] so a leaf source's `region_out` (pushed by
    /// its `lower` in case it turns out to be the bare DAG root) doesn't
    /// shadow the *actual* root op's `region_out` once one is registered.
    fn remove_fields_named(&mut self, names: &[String]) {
        let old_fields = std::mem::take(&mut self.params.fields);
        let old_sizes = std::mem::take(&mut self.params.field_sizes);
        let old_bytes = std::mem::take(&mut self.params.bytes);
        let mut offset = 0usize;
        for ((name, ty), size) in old_fields.into_iter().zip(old_sizes) {
            if !names.iter().any(|n| n == &name) {
                self.params.fields.push((name, ty));
                self.params.field_sizes.push(size);
                self.params.bytes.extend_from_slice(&old_bytes[offset..offset + size]);
            }
            offset += size;
        }
    }

    /// Materializer hook: announce the node about to be lowered — its
    /// identity, the identities of its input nodes (resolved to source slots or
    /// prior steps), and its resolved WorkUnit.
    pub fn enter(&mut self, node_key: NodeId, input_keys: &[NodeId], wu: &WorkUnit) {
        self.current_wu = Some(wu.clone());
        self.cur_node = Some(node_key);
        self.cur_started = false;
        self.cur_output_adapter = None;
        self.cur_inputs = input_keys
            .iter()
            .map(|k| {
                if let Some(adapted) = self.alias_adapters.get(k) {
                    adapted.clone()
                } else if let Some(&si) = self.source_of.get(k) {
                    StepInput { base: BaseInput::Source(si), adapter: None }
                } else if let Some(&st) = self.last_step_of.get(k) {
                    StepInput { base: BaseInput::Step(st), adapter: None }
                } else {
                    // Pruned/absent input reached a consumer — shouldn't happen
                    // for a live node; fall back to source 0 to stay total.
                    StepInput { base: BaseInput::Source(0), adapter: None }
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

    /// Register a **source input**: decode `View`, slot params, fetched buffer.
    /// Called by a Source leaf's `lower`; binds this node to a source slot.
    /// `slot_params` field names may contain the literal `"{slot}"`, which is
    /// replaced with this input's assigned slot index (e.g. `"region_in_{slot}"`
    /// → `"region_in_0"`).
    pub fn input(&mut self, view: View, slot_params: ParamBlock, buf: Arc<crate::backend::gpu::buffer::GpuBuffer>) -> &mut Self {
        let slot = self.input_views.len();
        if let Some(k) = self.cur_node {
            self.source_of.insert(k, slot);
        }
        for (name, ty) in &slot_params.fields {
            self.params.fields.push((name.replace("{slot}", &slot.to_string()), ty));
        }
        self.params.field_sizes.extend(slot_params.field_sizes.iter().copied());
        self.params.bytes.extend_from_slice(&slot_params.bytes);
        self.input_views.push(view);
        self.source_buffers.push(buf);
        // If this leaf turns out to be the DAG root (no kernel steps at all —
        // a plain `Data::from_source(..).pull(..)`), the encoder reads its
        // decoded value directly through this slot.
        self.cur_output_adapter = Some(StepInput { base: BaseInput::Source(slot), adapter: None });
        self
    }

    /// Add a kernel step to the current node. Its first kernel reads the node's
    /// graph inputs; any later kernel (intra-node multistep) reads the step
    /// before it. The step's output temp is its own index.
    pub fn kernel(&mut self, module: &'static str, entry: &'static str) -> &mut Self {
        self.kernel_with_temp(module, entry, TempElem::F4)
    }

    /// Like [`GpuBuilder::kernel`], but with a non-default working temp element
    /// (e.g. a reduction step writing `uint` bins instead of `float4` pixels).
    pub fn kernel_with_temp(&mut self, module: &'static str, entry: &'static str, temp_elem: TempElem) -> &mut Self {
        let inputs = if !self.cur_started {
            self.cur_started = true;
            self.cur_inputs.clone()
        } else {
            vec![StepInput { base: BaseInput::Step(self.steps.len() - 1), adapter: None }]
        };
        let params = std::mem::take(&mut self.pending_params);
        self.steps.push(Step { module, kernel: entry, inputs, params, temp_elem });
        let idx = self.steps.len() - 1;
        if let Some(k) = self.cur_node {
            self.last_step_of.insert(k, idx);
        }
        self
    }

    /// Make the current node a **zero-cost view** of its single input: no
    /// kernel step is added. The node's value becomes its input's value (a
    /// source decode or a prior step's temp) wrapped in `adapter` (swizzle,
    /// remap, …). A downstream consumer of this node gets this adapted
    /// `StepInput`; if this node is the DAG root, the emitter reads/encodes
    /// through the adapter directly (see `cur_output_adapter`).
    ///
    /// Used by `ExtractBand`/`Flip`/etc. — the view is then free: it never
    /// adds a pass, it just changes how the *next* kernel (or the encoder)
    /// reads the value that's already there. Works whether the input is a
    /// fresh source decode or a prior step's temp.
    ///
    /// If the current input is *already* adapted (chained view-only nodes),
    /// the existing adapter wins — nesting adapters isn't supported, the
    /// first one applied takes precedence.
    pub fn adapt(&mut self, adapter: ViewAdapter) -> &mut Self {
        let Some(input) = self.cur_inputs.first().cloned() else {
            self.fail(Error::Backend("GpuBuilder::adapt: node has no input to adapt".into()));
            return self;
        };
        let final_input = if input.adapter.is_some() {
            input
        } else {
            let n = self.adapter_count;
            self.adapter_count += 1;
            let prefix = format!("a{n}");
            let fields: Vec<(String, &'static str)> = adapter
                .params
                .fields
                .iter()
                .map(|(name, ty)| (name.replace("{p}", &prefix), *ty))
                .collect();
            self.params.fields.extend(fields.iter().cloned());
            self.params.field_sizes.extend(adapter.params.field_sizes.iter().copied());
            self.params.bytes.extend_from_slice(&adapter.params.bytes);
            let resolved = ViewAdapter {
                wrapper: adapter.wrapper,
                ctor: adapter.ctor.replace("{p}", &prefix).into(),
                params: ParamBlock { fields, field_sizes: adapter.params.field_sizes.clone(), bytes: adapter.params.bytes },
                module: adapter.module,
            };
            StepInput { base: input.base, adapter: Some(resolved) }
        };
        if let Some(k) = self.cur_node {
            self.alias_adapters.insert(k, final_input.clone());
        }
        self.cur_output_adapter = Some(final_input);
        self
    }

    /// The current node's value IS its single input's value — no kernel, no
    /// temp, no adapter. A downstream consumer resolving this node gets its
    /// input instead (via `alias_adapters`, the same map `adapt` uses); if
    /// this node is the DAG root, the encoder reads through it too (via
    /// `cur_output_adapter`). Used by [`crate::operation::Reinterpret`] — a
    /// zero-cost typed cast between byte-identical Kinds.
    pub fn forward(&mut self) -> &mut Self {
        let Some(input) = self.cur_inputs.first().cloned() else {
            self.fail(Error::Backend("forward: node has no input".into()));
            return self;
        };
        if let Some(k) = self.cur_node {
            self.alias_adapters.insert(k, input.clone());
        }
        self.cur_output_adapter = Some(input);
        self
    }

    /// A scalar param for the current step. Namespaced by step index so two
    /// steps (or two nodes) using the same name never collide in `ChainParams`.
    pub fn param<T: SlangScalar>(&mut self, name: &str, value: T) -> &mut Self {
        let idx = self.steps.len() - 1;
        let field = format!("s{idx}_{name}");
        self.params.fields.push((field.clone(), T::SLANG_TY));
        self.params.field_sizes.push(std::mem::size_of::<T>());
        self.params.bytes.extend_from_slice(bytemuck::bytes_of(&value));
        self.steps[idx].params.push(field);
        self
    }

    /// Register the output, as described by its **Kind** (`GpuView::output`).
    /// Merges the wrap's own `params` (e.g. a reduction's `bin_count`, or an
    /// image's `region_out`) into the shared `ChainParams`. If no dispatch
    /// domain was set explicitly (via [`GpuBuilder::dispatch`]), defaults it
    /// from the root's `Region` WorkUnit — for image ops the dispatch domain
    /// and the output rect are the same rectangle.
    pub fn output(&mut self, wrap: OutputWrap) -> &mut Self {
        if !self.dispatch_explicit {
            if let Some(r) = Region::typed(self.wu()) {
                self.dispatch = Some((r.w.max(0) as u32, r.h.max(0) as u32));
            }
        }
        // A leaf source's `lower` always calls `output()` (in case it's the
        // bare DAG root); when an op also calls `output()`, drop the
        // source's same-named fields (e.g. `region_out`) so the op's values
        // — not the source's — land at that field's offset in `ChainParams`.
        let names: Vec<String> = wrap.params.fields.iter().map(|(n, _)| n.clone()).collect();
        self.remove_fields_named(&names);
        self.params.fields.extend(wrap.params.fields.iter().cloned());
        self.params.field_sizes.extend(wrap.params.field_sizes.iter().copied());
        self.params.bytes.extend_from_slice(&wrap.params.bytes);
        self.output = Some(wrap);
        self
    }

    /// Merge a whole `ParamBlock` (e.g. a reduction's `bin_count`) into the
    /// shared `ChainParams`.
    pub fn param_block(&mut self, block: ParamBlock) -> &mut Self {
        self.pending_params.extend(block.fields.iter().map(|(name, _)| name.clone()));
        self.params.fields.extend(block.fields);
        self.params.field_sizes.extend(block.field_sizes);
        self.params.bytes.extend_from_slice(&block.bytes);
        self
    }

    /// Does the output use the codec sandwich (image) vs a direct write?
    pub fn needs_scratch(&self) -> bool {
        matches!(&self.output, Some(w) if w.dest == OutBuffer::Scratch)
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
    /// How an output of this Kind is written (see [`OutputWrap`]). `wu` is the
    /// resolved output WorkUnit — Region-shaped Kinds (images) use it to size
    /// their `region_out` geometry.
    fn output(&self, wu: &WorkUnit) -> OutputWrap;
}

impl crate::backend::Backend for GpuBackend {
    type Ctx = GpuContext;
    type Payload = GpuBuffer;
    type Builder = GpuBuilder;

    /// GPU-specific materialization: analyzes the DAG for binding budget
    /// violations, pre-materializes staging cuts in parallel via rayon, and
    /// re-runs with a reduced DAG. Falls through to the standard walk when
    /// no cuts are needed.
    fn materialize(
        ctx: &std::sync::Arc<GpuContext>,
        root: &std::sync::Arc<crate::node::Node<GpuBackend>>,
        wu: &WorkUnit,
    ) -> Result<crate::buffer::Buffer<GpuBackend>, Error> {
        pass::gpu_materialize(ctx, root, wu)
    }
}

impl Builder<GpuBackend> for GpuBuilder {
    fn new(ctx: Arc<GpuContext>) -> Self {
        Self::new(ctx)
    }

    fn enter(&mut self, node: NodeId, inputs: &[NodeId], wu: &WorkUnit) {
        GpuBuilder::enter(self, node, inputs, wu)
    }

    fn finish(mut self, _root: NodeId, spec: Arc<dyn AnyKind>, root_wu: &WorkUnit) -> Result<crate::buffer::Buffer<GpuBackend>, Error> {
        if let Some(e) = self.take_error() {
            return Err(e);
        }

        // Binding budget: the CutFinder (pass::gpu_materialize) should have
        // already split the DAG so this pass fits. If it still doesn't, that's
        // a bug in the CutFinder — fail loudly rather than silently submitting
        // an invalid bind group.
        let n_bindings = pass::binding_count(
            self.steps.len(),
            self.source_buffers.len(),
            self.needs_scratch(),
        );
        if n_bindings > self.ctx.max_storage_buffers as usize {
            return Err(Error::Backend(format!(
                "fused pass requires {} storage buffer bindings but the device \
                 supports at most {} — CutFinder failed to reduce the pass",
                n_bindings, self.ctx.max_storage_buffers
            )));
        }

        let dims = self.dispatch.unwrap_or((1, 1));
        RegionParams::tight(dims.0 as i32, dims.1 as i32).push_into(&mut self.params, "domain");

        let slang = emit::emit_slang(&self, self.ctx.wg_dim);
        let hash = emit::hash_slang(&slang);
        let compiled = compile::compile(self.ctx.as_ref(), &self, slang, hash)?;

        let out_bytes = spec.byte_size(root_wu);
        let payload = compile::dispatch(self.ctx.as_ref(), &compiled, &self, out_bytes, dims)?;

        Ok(crate::buffer::Buffer { payload, spec })
    }
}
