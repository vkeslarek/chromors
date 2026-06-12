# Core Simplification Plan

Status: **proposal — ready for implementation**
Scope: the agnostic core (`node.rs`, `kind.rs`, `work_unit.rs`, `io.rs`, `buffer.rs`,
`operation/mod.rs`, `backend/mod.rs`) and the GPU lowering/emission pipeline
(`backend/gpu/{mod,view,emit,compile,materialize}.rs`). The Vips builder, the Slang FFI
(`slang.rs`), the operation library, and the shader tree are touched only where a core
signature forces them to change.

## Goals and invariants

1. Idiomatic Rust. No magic macros — plain traits, structs, and functions.
2. The core must contain **zero datatype-specific code**. The core speaks `Kind`,
   `WorkUnit`, `ParamBlock`, `View` — never `Image`, never `Region`-matching logic,
   never swizzle/flip/rotate vocabulary.
3. No loss of functionality. Every test in `tests/gpu_probe.rs` and the 42 lib tests
   must stay green.
4. Fewer concepts, each with one owner. Where two layers duplicate a decision today
   (binding order, materialize walk, output element type), one of them becomes the
   single source of truth.

The litmus test for every change below: *could a new datatype (point list, video frame,
1-D mask) be added by writing only a new `data/*.rs` module + shaders, with zero edits
to `backend/gpu/{mod,emit,compile}.rs`?* Today the answer is no in four places; after
this plan the answer is yes.

---

## Part 1 — Inventory of problems

### 1.1 Image vocabulary inside the core (violates invariant 2)

| Where | What leaks | Severity |
|---|---|---|
| `backend/gpu/mod.rs` | `RemapKind` (FlipH/FlipV/Rot90/Tile/…) and `RemapParams` (out_w/out_h/sx/sy/tx/ty) — pure 2-D image geometry — are core types; `StepInput` has 4 image-shaped variants (`SwizzleSource/SwizzleStep/RemapSource/RemapStep`); `GpuBuilder::alias`/`remap` are image-semantics methods | High |
| `backend/gpu/mod.rs` | `GpuBuilder::output()` matches `WorkUnit::Region` to push `region_out`; `GpuBuilder::input()` takes a `RegionParams` argument | High |
| `backend/gpu/materialize.rs` | dispatch grid = `match root_wu { Region(r) => (w,h), _ => (1,1) }` — assumes "output shape == thread grid", which is an *image* property. For `Atomic` outputs this is also a **correctness bug**: a histogram over a 512×512 image dispatches one workgroup (≤32×32 threads) and silently counts ≤1024 pixels | High (bug) |
| `backend/gpu/compile.rs` | work-temp size hardcoded `dims.0 * dims.1 * 16` (float4) — duplicates `TempElem` knowledge and assumes the image temp | Medium |
| `backend/gpu/emit.rs` | hardcoded, ever-growing `import ops.*;` list (the file itself says "a production emitter would carry each kernel's module path"); `TempElem::F4` fallback in the zero-step path; broadcast/`output_swizzle` (band-extraction semantics) implemented inside the emitter | Medium |
| `backend/gpu/mod.rs` | `GpuBuilder::kernel()` hardcodes `temp_elem: TempElem::F4` — every step temp is an image working pixel | Medium |

### 1.2 Duplication (violates invariant 4)

- **`Backend::materialize` is copy-pasted per backend.** GPU and Vips both do:
  demand walk → `GraphWalk::lower` with an `enter` + `node.lower` closure → backend
  finish. Only the builder type and the finish differ.
- **Binding layout is written three times** — `emit.rs` (declarations), `compile.rs`
  `compile()` (BGL entries), `compile.rs` `dispatch()` (BG entries) — all must agree on
  the order *target, params, work_0..k, src_0..n* by hand. One drift = wrong buffer
  bound silently.
- **`OutputWrap` duplicates `View`'s fields** (`arg_type`/`arg_ctor`/`buffer_type` ≡
  `slang`/`ctor`/`buffer_type`). The recently-fixed missing-`buffer_type` bug existed
  *because* of this parallel field set.
- **`Data::pull` and `Data::extract` are byte-identical.**
- The zero-step encode path in `emit.rs` re-implements the per-step input-resolution
  logic (`SwizzleSource`/`RemapSource`/`Source` match) a second time.

### 1.3 Dead code

- `Role` enum and `Binding` struct (+ `View.binding` field, always `{0,0}`, never read).
- `AnyKind::shape()` — never called anywhere. The `Shape` enum is only used to
  implement this dead method (the *typed* shape already lives in
  `Kind::WorkUnit: WorkUnitFor`).
- `ParamBlock::empty()` duplicates `new()`/`Default`.
- `Data::_m: PhantomData<(K, B)>` — both `K` (in `spec: Arc<K>`) and `B` (in
  `root`/`ctx`) already appear in real fields.

### 1.4 Fragile contracts (correctness findings — fix while here)

- **F1.** `GpuBuilder::enter` falls back to `StepInput::Source(0)` when an input node
  resolves to nothing ("fall back to source 0 to stay total") — silent wrong-data.
  Must be an error.
- **F2.** `Operation::demand` must return one entry per `inputs()` element; the demand
  walk `zip`s them and silently drops extras/missing. **`Composite2`, `Join`, and
  `Insert` violate this today**: 2 inputs, `vec![Some(..)]` of length 1 → the second
  input gets no demand → pruned → hits F1's fallback.
- **F3.** `GraphWalk::demand` re-pushes a node's children on *every* visit even when
  the union didn't grow — exponential on dense diamond chains.
- **F4.** `GpuBuilder::param` declares every scalar as `"scalar"` → emitted as `float`.
  Passing a `u32` (e.g. `param("channel", self.channel)`) produces an
  implicit-conversion warning and float-rounded bits on the GPU.
- **F5.** `emit.rs` formats `RemapParams` floats with `{:?}` directly into Slang source
  — `f32::NAN`/`INFINITY` would emit invalid Slang, and params-in-source defeats the
  pipeline cache (every translate offset = a new pipeline).

---

## Part 2 — Proposed design

Seven changes, ordered by leverage. Each section: what, how (with code), why.

### C1. One `materialize` in the core; backends provide a `Builder`

**What.** Delete `Backend::materialize` and both per-backend implementations
(`gpu/materialize.rs`'s `Materializer`, the body in `vips/mod.rs`). The core owns the
walk; a backend only says what happens *per node* and *at the end*.

```rust
// backend/mod.rs
pub trait Backend: Sized + Send + Sync + 'static {
    type Ctx: Send + Sync;
    type Payload: Send + Sync;
    type Builder: Builder<Self>;
}

/// What a backend accumulates during the lower walk and how it finishes.
pub trait Builder<B: Backend>: Sized {
    fn new(ctx: Arc<B::Ctx>) -> Self;
    /// Announce the node about to lower: its id, its input ids, its resolved unit.
    fn enter(&mut self, node: NodeId, inputs: &[NodeId], wu: &WorkUnit);
    /// Run the pass and produce the root's buffer.
    fn finish(self, root: NodeId, spec: Arc<dyn AnyKind>, root_wu: &WorkUnit)
        -> Result<Buffer<B>, Error>;
}
```

```rust
// node.rs — THE engine entry, written once
pub(crate) fn materialize<B: Backend>(
    ctx: &Arc<B::Ctx>,
    root: &Arc<Node<B>>,
    wu: &WorkUnit,
) -> Result<Buffer<B>, Error> {
    let mut walk = GraphWalk::new(root);
    walk.demand(wu);

    let mut builder = B::Builder::new(Arc::clone(ctx));
    walk.lower(|node, node_wu| {
        let inputs: Vec<NodeId> = node.inputs().iter().map(|i| NodeId::of(i.src())).collect();
        builder.enter(NodeId::of(node), &inputs, node_wu);
        node.lower(&mut builder);
    });
    builder.finish(NodeId::of(root), root.output_kind(), wu)
}
```

- `GpuBuilder::finish` = today's `Materializer::execute` steps 3–5 (take_error → emit
  → compile → dispatch → `Buffer`). `gpu/materialize.rs` is deleted.
- `VipsBuilder::finish` = `take(root)` → `Buffer`. `VipsBuilder::enter` ignores
  `inputs` (it resolves inputs lazily via `input(&Arc<Node>)`); that's fine — the
  parameter costs nothing.

**Why.** Removes ~60 duplicated lines and the *possibility* of the two walks drifting.
A third backend (e.g. a pure-CPU Rust backend) becomes "implement `Builder`", not
"copy the walk correctly".

### C2. `NodeId` newtype

```rust
// node.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub fn of<B: Backend>(node: &Arc<Node<B>>) -> Self {
        Self(Arc::as_ptr(node) as *const () as usize)
    }
}
```

Replaces the `Arc::as_ptr(x) as *const () as usize` cast currently scattered across
`node.rs`, `gpu/materialize.rs`, `vips/mod.rs`, and `data/*.rs` hash impls.
`GraphWalk.demands`/`lowered`, `GpuBuilder.source_of`/`last_step_of`/aliases, and
`VipsBuilder.outputs` all key on `NodeId`. Pure mechanical, no behavior change.

### C3. `Data` cleanup

- Delete `_m: PhantomData<(K, B)>` (both parameters appear in real fields).
- Delete `extract` (keep `pull`; identical bodies).
- Add constructors so tests/sources stop building the struct literally:

```rust
impl<K: Kind, B: Backend> Data<K, B> {
    pub fn from_source<S: Source<B, Kind = K>>(source: Arc<S>, ctx: Arc<B::Ctx>) -> Self {
        let spec = source.spec();
        Self { root: Arc::new(Node::Source(source)), ctx, spec }
    }
}
```

`Image2D::open`, `tests/common::vips_to_gpu`, and the raw-backend constructors all
collapse onto this.

### C4. `ViewAdapter` — alias/remap leave the core

**What.** Replace the four image-shaped `StepInput` variants, `RemapKind`,
`RemapParams`, `GpuBuilder::alias`, `GpuBuilder::remap`, `output_swizzle`, and
`TempElem::{components, broadcast}` with one generic, data-driven concept: a zero-cost
Slang wrapper interposed on an edge.

```rust
// backend/gpu/view.rs — core. Knows NOTHING about swizzle/flip/scale.
/// A zero-cost Slang view interposed between a producer and a consumer —
/// no kernel step, no buffer. The core stores and splices strings; the
/// semantics live in the shader struct and the datatype module that builds it.
#[derive(Debug, Clone)]
pub struct ViewAdapter {
    /// Wrapper type template. `{inner}` = the wrapped IRegion type.
    /// e.g. "SwizzleView<{inner}>", "RemapView<{inner}>".
    pub wrapper: Cow<'static, str>,
    /// Ctor template. `{value}` = the wrapped value expr, `{params}` = ChainParams
    /// var, `{p}` = this adapter's unique param-field prefix.
    /// e.g. "{ {value}, {params}[0].{p}_channel }".
    pub ctor: Cow<'static, str>,
    /// Adapter params, field names prefixed with `{p}` (the builder replaces it
    /// with a unique `a{n}` so two adapters never collide).
    pub params: ParamBlock,
    /// Slang module defining the wrapper struct (folded into emitted imports).
    pub module: &'static str,
}
```

```rust
// backend/gpu/mod.rs — StepInput shrinks from 6 variants to a flat pair:
#[derive(Clone, Debug)]
pub enum BaseInput {
    Source(usize),
    Step(usize),
}

#[derive(Clone, Debug)]
pub struct StepInput {
    pub base: BaseInput,
    pub adapter: Option<ViewAdapter>,
}

impl GpuBuilder {
    /// Make the current node a zero-cost adapted view of its single input.
    /// Replaces both `alias()` and `remap()`.
    pub fn adapt(&mut self, adapter: ViewAdapter) -> &mut Self { /* … */ }
}
```

The *constructors* move to the modules that own the semantics:

```rust
// data/image.rs (or operation/bands.rs) — band extraction:
pub fn swizzle_adapter(channel: u32) -> ViewAdapter {
    ViewAdapter {
        wrapper: "SwizzleView<{inner}>".into(),
        ctor: "{ {value}, {params}[0].{p}_channel }".into(),
        params: ParamBlock::scalar("{p}_channel", "uint", channel),
        module: "lib.region",
    }
}

// operation/geometry.rs — RemapKind/RemapParams move HERE (with the ops that
// use them), and the numeric params go through ChainParams, not into source text:
pub fn remap_adapter(kind: RemapKind, p: RemapParams) -> ViewAdapter {
    ViewAdapter {
        wrapper: "RemapView<{inner}>".into(),
        ctor: "{ {value}, {params}[0].{p}_kind, {params}[0].{p}_geo }".into(),
        params: ParamBlock::new()
            .param("{p}_kind", "uint", kind as u32)
            .param("{p}_geo", "RemapGeo", p),  // one POD struct field, std430
        module: "lib.region",
    }
}
```

Call sites change one line: `cx.alias(2)` → `cx.adapt(swizzle_adapter(2))`,
`cx.remap(kind, p)` → `cx.adapt(remap_adapter(kind, p))`.

**Emitter effect.** The 6-arm `StepInput` match in `emit.rs` becomes one function used
by both the step loop and the final-encode path (deleting the duplicated zero-step
match):

```rust
/// The Slang lvalue for reading `input`, declaring an adapter local if needed.
fn read_expr(s: &mut String, input: &StepInput, builder: &GpuBuilder, var: &str) -> String {
    let inner = match input.base {
        BaseInput::Source(i) => format!("in_{i}"),
        BaseInput::Step(j) => { /* declare `{wrapper} r = { work_j, params[0].domain }` */ }
    };
    match &input.adapter {
        None => inner,
        Some(a) => { /* declare `{a.wrapper/{inner}} {var} = {a.ctor expanded}` */ var.into() }
    }
}
```

The root-alias broadcast special case disappears: when the DAG root is adapted, the
encode tail reads through the adapter (`enc.write(idx, adapted.read(idx))`) — the
`SwizzleView` shader already broadcasts `float4(v,v,v,1)`, so behavior is identical and
`TempElem.components`/`broadcast`/`component()`/`broadcast_expr()` are deleted.

**Why.** This is the heart of invariant 2. After C4, adding a new zero-cost view (e.g.
a wraparound sampler, a channel matrix) = one shader struct + one constructor function
in a datatype module. Core untouched. It also fixes **F5** (no more `{:?}`-formatted
floats in source; pipeline cache stops churning per-offset).

### C5. Explicit dispatch domain; output params move to the Kind

**What.** The builder stops inferring the thread grid from `WorkUnit::Region` and
stops pushing `region_out` itself.

```rust
// backend/gpu/mod.rs
impl GpuBuilder {
    /// The pass's thread grid. Set by whoever knows the work domain:
    /// an image op's output, a reduction op's input extent.
    pub fn dispatch(&mut self, dims: (u32, u32)) -> &mut Self { self.dispatch = Some(dims); self }
}
```

- `OutputWrap` gains `params: ParamBlock` (absorbing the just-added
  `GpuBuilder::output_params` mechanism — delete that method). `GpuView::output`
  changes signature to `fn output(&self, wu: &WorkUnit) -> OutputWrap` so the Kind
  computes its own geometry:

```rust
// data/image.rs
impl GpuView for ImageKind {
    fn output(&self, wu: &WorkUnit) -> OutputWrap {
        let r = Region::typed(wu).expect("image output is Region-shaped");
        OutputWrap {
            arg: View::new(/* RWRegion … */),
            dest: OutBuffer::Scratch,
            encode: Some(/* RWCodecRegion … */),
            params: RegionParams::tight(r.w, r.h).into_block("region_out"),
        }
    }
}
```

- Region-shaped ops set the grid alongside the output:
  `cx.output(spec.output(cx.wu())).dispatch((r.w as u32, r.h as u32))` — or, since this
  pairs 1:1, a convenience on the builder that reads `wrap.params`' dims is acceptable;
  prefer the explicit call.
- Reduction ops set it from their input: `HistogramOp::lower` does
  `cx.dispatch((w, h))` with the input dims it already computes in `demand()`.
  **This fixes the `(1,1)` Atomic-dispatch bug** (1.1, row 3).
- `materialize`/`finish` reads `builder.dispatch.unwrap_or((1, 1))`. No `WorkUnit`
  match remains anywhere in `backend/gpu/`.
- The per-step work temps need a geometry struct; that geometry is the *thread grid*,
  not the image. The emitter pushes one emitter-owned `BufferRegion` named `domain`
  derived from `dispatch`, and temps wrap `{ work_j, params[0].domain }`. (`region_out`
  keeps existing only inside `ImageKind`'s own ctor strings.)

**Why.** "Thread grid" is GPU vocabulary, owned by the GPU builder; "output rect" is
image vocabulary, owned by `ImageKind`. Today they're conflated, which is exactly why
Atomic outputs dispatch wrong.

### C6. `GpuBuilder::input` stops taking `RegionParams`; `{slot}` hole

```rust
// before
pub fn input(&mut self, view: View, region: RegionParams, buf: Arc<GpuBuffer>) -> &mut Self
// after
pub fn input(&mut self, view: View, slot_params: ParamBlock, buf: Arc<GpuBuffer>) -> &mut Self
```

The source builds its own geometry block with a `{slot}` hole in field names
(`"region_in_{slot}"`); the builder replaces `{slot}` with the assigned index in both
the field names and the view's ctor. The `{region}` ctor hole is deleted — `ImageKind`'s
input ctor becomes explicit: `"{ {buf}, {params}[0].region_in_{slot} }"`.

**Why.** Same as C5: the core routes opaque param blocks; only `data/image.rs` knows
images carry a `BufferRegion`. A 1-D mask source would push a `range_in_{slot}` field
instead, with zero core changes.

### C7. Emitter hygiene

1. **Kernel module on the step.** `kernel(entry)` →
   `kernel(module: &'static str, entry: &'static str)`, e.g.
   `cx.kernel("ops.invert", "invert_kernel")`. The emitter collects distinct modules
   from steps + adapters + views into the import block. Delete the 18-line hardcoded
   list. (Lib imports `lib.region/io/codecs/pixel` stay — they're the emitter's own
   vocabulary.)
2. **Single binding-layout source of truth.**

```rust
// backend/gpu/emit.rs (or layout.rs)
pub enum Slot<'a> {
    Target,
    Params,
    Work(usize, &'a TempElem),
    Source(usize, &'a View),
}
pub fn slots(builder: &GpuBuilder) -> impl Iterator<Item = Slot<'_>>
```

   `emit_slang` (declarations), `compile` (BGL entries: `read_only` from the variant),
   and `dispatch` (BG entries) all iterate `slots()`. The three hand-synchronized loops
   are deleted.
3. **Typed scalars (fixes F4).**

```rust
pub trait SlangScalar: bytemuck::Pod {
    const SLANG_TY: &'static str;
}
impl SlangScalar for f32 { const SLANG_TY: &'static str = "float"; }
impl SlangScalar for u32 { const SLANG_TY: &'static str = "uint"; }
impl SlangScalar for i32 { const SLANG_TY: &'static str = "int"; }
```

   `GpuBuilder::param<T: SlangScalar>` and `ParamBlock::param<T: SlangScalar>` (drop
   the `slang_type: &'static str` argument) use `T::SLANG_TY`. Delete the
   `"scalar"` → `"float"` remap in `emit_slang`. Existing
   `param("x", self.x /* i32 */)` call sites become *correct automatically*; audit any
   site that deliberately passed an int as float (e.g. cast at the call site).
4. **`OutputWrap` composes `View`** (fixes 1.2, bullet 3):

```rust
pub struct OutputWrap {
    pub arg: View,             // type + ctor + buffer elem of the kernel's out arg
    pub dest: OutBuffer,       // Scratch | Target
    pub encode: Option<View>,  // codec-sandwich close; None ⇒ kernel wrote target
    pub params: ParamBlock,    // Kind-owned output geometry/config (see C5)
}
```

   Target element type = `encode.as_ref().map(|e| &e.buffer_type).unwrap_or(&arg.buffer_type)`
   — and for the `Scratch` case `arg.buffer_type` is the *scratch* elem; the target
   elem under `encode` is the encode view's, same as today.
5. **`TempElem` slims down** to `{ buffer_ty, region_wrapper, byte_size: u64 }`
   (components/broadcast deleted per C4; `byte_size` added so `dispatch()` sizes each
   step's temp as `domain_w * domain_h * elem.byte_size` instead of hardcoding 16 —
   fixes 1.1, row 4). `Step.temp_elem` stays; `kernel()` keeps defaulting to
   `TempElem::F4` *for now* but gains `kernel_with_temp(module, entry, elem)` so a
   non-float4 step is possible without touching the core later.
6. Use `std::fmt::Write` (`write!(s, …)`) instead of `s.push_str(&format!(…))`
   throughout `emit_slang`.

### C8. Contract enforcement + dead-code deletion

- **F1**: `enter`'s unresolved-input fallback becomes
  `self.fail(Error::Backend(format!("input node {k:?} was never lowered")))`.
- **F2**: in `GraphWalk::demand`, assert the op's contract:

```rust
let demands = node.demand_erased(&wu);
debug_assert_eq!(demands.len(), node.inputs().len(),
    "Operation::demand must return one entry per input");
```

  and **fix the three violators** — `Composite2`/`Join`/`Insert` return
  `vec![Some(WorkUnit::Region(out.clone())); 2]` (one per input; correct halo math is
  out of scope here, full-rect per input is the status quo for these ops).
- **F3**: demand walk only re-pushes children when the union grew:

```rust
let grown = match self.demands.entry(k) {
    Entry::Occupied(mut e) => { let u = e.get().union(&wu); let g = /* compare */; e.insert(u); g }
    Entry::Vacant(e) => { e.insert(wu.clone()); true }
};
if !grown { continue; }
```

  (Requires `PartialEq` on `WorkUnit` — derive it.)
- Delete: `Role`, `Binding`, `View.binding`, `AnyKind::shape()`, the `Shape` enum
  (`Kind::WorkUnit` already *is* the typed shape; update `data/*.rs` impls and the
  doc-comments that mention it), `ParamBlock::empty()`, `Data::_m`, `Data::extract`,
  `GpuBuilder::output_params` (absorbed by `OutputWrap.params`),
  `gpu/materialize.rs` (absorbed by C1).

---

## Part 3 — What deliberately stays

- **String-template Slang emission** (`View.ctor` holes, `ParamBlock`). This is the
  no-macro, data-driven answer to JIT codegen; the alternative (a typed Slang AST) is
  a project, not a cleanup. The templates get *fewer* and more uniform here
  (`{buf}/{params}/{value}/{inner}/{p}/{slot}`), not replaced.
- **`work_unit.rs` defining `Region`/`Range`/`Atomic`.** The core may *define* the
  closed shape vocabulary (with `WorkUnit::union` as the one allowed shape switch);
  what it may not do is *match on a specific shape* to make decisions — C5/C6 remove
  the two places that did.
- **`GpuView` / `VipsBand` as per-backend Kind capabilities.** This split is correct
  and is exactly how GPU-only Kinds (`HistogramKind`) are enforced at compile time.
  Optional cosmetic: rename `GpuView` → `GpuRepr` for symmetry; not required.
- **`VipsBuilder`** — already minimal and shape-blind; only gains the `Builder` trait
  impl (C1).
- **`slang.rs`, `context.rs`, `GpuBuffer`** — sound RAII/FFI; untouched.
- **Pointer-identity node keys** — fine for an immutable `Arc` DAG; C2 just names it.

---

## Part 4 — Implementation order

Each step compiles and passes `cargo test --lib` (42 green) +
`cargo test --test gpu_probe` before the next.

1. **C8 dead-code deletion + C3 + C2** (mechanical, zero behavior): delete dead items,
   `NodeId`, `Data` cleanup. Touches many files trivially.
2. **C1** materialize hoist (`Builder` trait, delete `gpu/materialize.rs`, slim
   `vips/mod.rs`).
3. **C7.3** `SlangScalar` (fixes F4) and **C7.4** `OutputWrap{arg, dest, encode, params}`
   — update `ImageKind`/`HistogramKind`/`VectorscopeKind` impls.
4. **C5 + C6** dispatch domain + input params (fixes the Atomic dispatch bug — add a
   `gpu_probe` assertion that a histogram over an image larger than one workgroup
   counts *all* pixels).
5. **C4** `ViewAdapter` (move `RemapKind`/`RemapParams` to `operation/geometry.rs`,
   port `alias`/`remap` call sites, add `RemapGeo` std430 struct to `lib/region.slang`,
   change `RemapView` to read params instead of inline literals).
6. **C7.1/.2/.5/.6** emitter hygiene (kernel module paths — mechanical edit of every
   `cx.kernel("x_kernel")` call site to `cx.kernel("ops.x", "x_kernel")`; `slots()`;
   `TempElem.byte_size`; `fmt::Write`).
7. **C8 F2/F3** demand contract assert + fix `Composite2`/`Join`/`Insert` + worklist
   growth check.

Estimated diff: core shrinks by ~250 lines; `data/`/`operation/` grow by ~80
(constructors + moved types). No public-API regression for the `data/*` user surface
(`Image2D::open/invert/…`, `pull`) except `extract` → `pull` (one name).

## Part 5 — Verification

- `cargo check --workspace && cargo test --lib` — 42/42.
- `cargo test --test gpu_probe` — all current tests, plus two new:
  - `histogram_counts_all_pixels_in_large_image` (image > 1 workgroup; asserts
    `sum(bins) == w*h`) — guards the C5 fix.
  - `remap_pipeline_cache_stable` — two `Translate` remaps with different offsets
    produce the **same** pipeline-cache key (guards C4/F5; inspect via the slang text
    hash, not timing).
- `grep -rn "Region(" src/backend/gpu/` → only `work_unit::Region` *type* mentions in
  comments; zero `WorkUnit::Region` matches.
- `grep -rn "Remap\|Swizzle\|alias" src/backend/gpu/mod.rs` → no hits (all moved).
- `cargo clippy --workspace` clean.

## Out of scope (explicitly)

- Multi-output nodes, standalone graph-builder entry, node-editor APIs.
- Typed Slang AST / replacing string templates.
- The working-space sandwich color-accuracy caveats noted in
  `tests/gpu_probe.rs::bandjoin_reconstruction_close_to_original`.
- Halo-correct `demand()` math for `Composite2`/`Join`/`Insert` (F2 only fixes the
  arity, preserving today's full-rect behavior).
- The Tier2/Tier3 op backlog (`docs/gpu-ops-todo.md`).
