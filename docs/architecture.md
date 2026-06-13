# Chromors Processing Model v2 — Architecture

> Companion to `CLAUDE.md` (the binding contract — read that first for the
> *rules*; this document is the *map*: what each type/trait actually does,
> how the pieces fit at runtime, and how to extend the system without
> breaking the model). Where this document and `CLAUDE.md` disagree,
> `CLAUDE.md` wins — file an issue against this doc.

This document is for someone arriving fresh. It walks the whole stack
bottom-up: agnostic core types → the DAG/materializer → the two backends
(GPU/Slang, libvips) → how to add a datatype or an operation → a full
worked trace of a real pipeline through both backends.

---

## 1. The one-sentence model

A **lazy, multi-backend DAG**. User code builds an immutable tree of
`Arc<Node<B>>` by calling methods like `.blur(3.0)` — nothing runs yet. Only
when you call `.pull(target, work_unit)` does the engine:

1. walk the tree **backwards** from the root, figuring out how much of each
   input is actually needed (the **demand walk**),
2. walk it **forwards** (post-order, deduplicated) handing each node to the
   chosen `Backend`'s `Builder`, which accumulates backend-specific state
   (the **lower walk**),
3. ask the `Builder` to `finish()` — for GPU this JIT-compiles one fused
   Slang shader and dispatches it; for vips this just returns the handle the
   vips operation chain already built,
4. hand the resulting backend-resident `Buffer<B>` to a `Target`, which is
   the *only* way data leaves the model (download to RAM, write to disk, or
   just hand back the resident buffer for a viewport).

Everything in the crate is in service of steps 1–4. The rest of this
document is "where does each piece live, and what does it actually do".

---

## 2. The two halves, concretely

`CLAUDE.md` §2 states the rule (AGNOSTIC vs PER-BACKEND). Concretely, here is
which **files** are which:

| Half | Files |
|---|---|
| **AGNOSTIC** (no Slang, no wgpu, no libvips types) | `src/kind.rs`, `src/work_unit.rs`, `src/operation/mod.rs` (the traits — not the per-op `impl Lower<...>` blocks), `src/io.rs`, `src/node.rs`, `src/buffer.rs`, `src/backend/mod.rs` |
| **PER-BACKEND** | `src/backend/gpu/**` (everything), `src/backend/vips/**` (everything), and every `impl Lower<GpuBackend>` / `impl Lower<VipsBackend>` block scattered through `src/operation/*.rs` and `src/data/*.rs` |
| **Datatypes** (mix of both, but cleanly separated per-file) | `src/data/*.rs` — each file has an agnostic `*Kind` (`impl AnyKind`, `impl Kind`) plus per-backend capability impls (`impl GpuView for *Kind`, `impl VipsBand for *Kind`) side by side |

The litmus test from `CLAUDE.md` §2 is real: open any file, and if a type or
method mentions `View`, `ParamBlock`, `wgpu`, `VipsHandle`, or `gaussblur`,
it is per-backend. If you're writing code that the *materializer* or
`GraphWalk` needs to call, and it mentions any of those — stop, you've
crossed the line.

---

## 3. The agnostic core, type by type

This section is the "what does each trait actually do" reference. Read it
once; everything else in the crate is built on these eight things.

### 3.1 `AnyKind` / `Kind` — `src/kind.rs`

```rust
pub trait AnyKind: Send + Sync + Debug + 'static {
    fn as_any(&self) -> &dyn Any;
    fn byte_size(&self, wu: &WorkUnit) -> u64;
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

pub trait Kind: AnyKind + Clone + Sized {
    type WorkUnit: WorkUnitFor;
}
```

A **Kind** is the *metadata* that tags a piece of data flowing through the
DAG — `ImageKind { format, color_space, width, height }`,
`HistogramKind { bins, bands }`, etc. It is **not** the data itself (the data
lives in a backend's `Payload`, see §3.6), and it is **not** a handle (that's
`Data<K, B>`, §3.5).

- `AnyKind` is the **object-safe, erased** view — `Arc<dyn AnyKind>` is what
  flows through `Node<B>`, `Buffer<B>`, etc., where the concrete Kind isn't
  known. `byte_size(wu)` answers "how many bytes does a `WorkUnit` of this
  shape cost for this Kind" — the GPU backend uses this to size the output
  buffer (`compile::dispatch`'s `out_bytes`).
- `Kind` is the **typed** surface generic op code is written against. Its
  only addition is `type WorkUnit: WorkUnitFor` — *which* of `Region` /
  `Range` / `Atomic` (§3.2) this datatype's operations are sliced by.
- `as_any()` exists so a concrete type can be recovered *if truly needed*
  (it almost never is in agnostic code — see invariant #2, no
  `downcast_ref::<ConcreteKind>()` in the materializer).
- `dyn_hash` feeds a `Hasher` with the Kind's own identity bits (not a
  `Debug` string) — used for cache keys (the `Cached` source adapter
  pattern, and the GPU pipeline cache transitively via op `dyn_hash`).

**`ReinterpretAs<T>`** is the third trait here: a Kind declares
`fn reinterpret_spec(&self) -> T` to say "my payload bytes are also valid as
a `T`". The `T: Kind<WorkUnit = Self::WorkUnit>` bound is a compile-time
guard against reinterpreting e.g. a `Region`-shaped Kind as an
`Atomic`-shaped one. This powers `Data::reinterpret()` (§3.5) and the
`Reinterpret<K,T,B>` operation (§8.4).

### 3.2 `WorkUnit` family — `src/work_unit.rs`

```rust
pub struct Lod(pub u32);                 // mip level; scale_factor() = 1 << lod
pub struct Region { x, y, w, h, lod }     // 2D slice — images
pub struct Range  { start, end }          // 1D slice — e.g. a LUT range
pub struct Atomic;                        // 0D — "the whole thing" (histograms)

pub enum WorkUnit { Region(Region), Range(Range), Atomic }

pub trait WorkUnitFor: Clone + Send + Sync + 'static {
    fn erase(&self) -> WorkUnit;
    fn typed(wu: &WorkUnit) -> Option<Self>;
}
```

A `WorkUnit` answers "how much of this node's output do we need / are we
producing". Every `Kind` picks exactly one shape via `type WorkUnit`. The
**erased** `WorkUnit` enum is what crosses node boundaries in the generic
walk (since two adjacent nodes can have different Kinds, hence different
typed `WorkUnit`s) — `WorkUnitFor::erase`/`typed` convert between the typed
and erased forms at the edges.

`WorkUnit::union` is **the one allowed shape-`match`** (invariant #3 in
`CLAUDE.md`):

```rust
pub fn union(&self, other: &WorkUnit) -> WorkUnit {
    match (self, other) {
        (Region(a), Region(b)) => Region(a.bounding(b)),
        (Range(a), Range(b))   => Range(a.bounding(b)),
        (Atomic, Atomic)       => Atomic,
        _ => self.clone(),
    }
}
```

This is called by `GraphWalk::demand` (§4) every time a node is reached by a
second consumer with a (possibly different) demand — the two demands are
merged into the smallest unit covering both. The actual rectangle/range math
(`Region::bounding`, `Range::bounding`, `Region::expanded`,
`Region::tile_aligned`) lives on the typed structs — `union` just dispatches
to the right one. **Adding a new datatype never adds an arm here** — if your
Kind's `WorkUnit` is `Region`/`Range`/`Atomic`, `union` already handles it;
if you invent a 4th shape, *that's* the one place you'd add an arm (and it
hasn't happened yet — all current datatypes fit these three).

`Region::expanded(amount)` is the halo helper — `Blur::demand` calls
`out.expanded(radius)` to ask its input for `radius` extra pixels on each
side (see §8.3 and §10).

### 3.3 `Operation<B>` / `Lower<B>` / `AnyOperation<B>` — `src/operation/mod.rs`

This is the heart of "what is a node that *computes* something" (as opposed
to a `Source`, §3.4, which is a leaf).

```rust
pub trait Operation<B: Backend>: Lower<B> + 'static + Send + Sync {
    type Output: Kind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>>;
    fn demand(&self, out: &<Self::Output as Kind>::WorkUnit) -> Vec<Option<WorkUnit>>;
    fn output_spec(&self) -> Self::Output;
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

pub trait Lower<B: Backend> {
    fn lower(&self, cx: &mut B::Builder);
}
```

- **`type Output: Kind`** — what this op produces. Often the same Kind as
  the input (`Blur`, `Invert`: `ImageKind → ImageKind`), sometimes a
  different one (`HistogramOp`: `ImageKind → HistogramKind`).
- **`inputs()`** — the op's input edges as `&dyn AnyInput<B>` (object-safe,
  see §3.3.1). Most ops have one input (`vec![&self.input]`); `Bandjoin` has
  N; a constant-only op (none exist yet, but would be) has zero.
- **`demand(out)`** — *the* propagation function. Given "the consumer wants
  this much of *my* output", return "how much do I need from *each of my
  inputs*" — one entry per `inputs()`, `None` meaning "this input isn't
  needed for this region at all" (pruning). This is where halos
  (`Region::expanded`), full-image reductions (`Region::full(...)`), or
  pass-through (`out.clone()`) get expressed. **This is purely geometric
  bookkeeping — no backend, no lowering.**
- **`output_spec()`** — a fresh clone of `Self::Output`, usually derived from
  the input's spec (e.g. `Blur::output_spec` returns the input's `ImageKind`
  unchanged — same format/size; `ExtractBand::output_spec` changes
  `format` to a smaller band count via `with_band_count`).
- **`dyn_hash`** — hash *this op's own parameters* (sigma, channel index,
  …) for the GPU pipeline cache key. (`output_spec().dyn_hash()` is hashed
  too, by the blanket `AnyOperation` impl below — so e.g. two `Blur`s with
  identical sigma but different output formats get different cache entries.)
- **`Lower<B>`** is the *only* place backend-specific code is written. One
  `impl Lower<GpuBackend> for Foo<GpuBackend>` and/or
  `impl Lower<VipsBackend> for Foo<VipsBackend>` per op. `lower` takes
  **only** `&mut B::Builder` — no `WorkUnit` parameter (the builder carries
  the resolved `WorkUnit` internally, see `cx.wu()` in §5.2). This keeps the
  signature identical across backends.

#### 3.3.1 `Input<K,B>` / `AnyInput<B>`

```rust
pub trait AnyInput<B: Backend>: Send + Sync + 'static {
    fn src(&self) -> &Arc<Node<B>>;
    fn spec(&self) -> &dyn AnyKind;
}

pub struct Input<K: Kind, B: Backend> {
    pub src: Arc<Node<B>>,
    pub spec: Arc<K>,
}
```

`Input<K,B>` is a typed edge: a pointer to the upstream `Node<B>` plus a
clone of its output `Kind` (so the op can read e.g. `self.input.spec.format`
without re-walking the graph). `AnyInput<B>` is its object-safe shadow —
`Operation::inputs()` returns `Vec<&dyn AnyInput<B>>` so the walk can treat
heterogeneous-Kind inputs (e.g. `Bandjoin`'s five `Input<ImageKind, B>`s, or
a future cross-Kind op) uniformly.

#### 3.3.2 `AnyOperation<B>` — the erased bridge

```rust
pub trait AnyOperation<B: Backend>: Send + Sync + 'static {
    fn inputs(&self) -> Vec<&dyn AnyInput<B>>;
    fn demand_erased(&self, out: &WorkUnit) -> Vec<Option<WorkUnit>>;
    fn output_kind(&self) -> Arc<dyn AnyKind>;
    fn lower(&self, cx: &mut B::Builder);
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

impl<B: Backend, T: Operation<B>> AnyOperation<B> for T { ... }  // blanket impl
```

`Operation<B>` is generic over `Self::Output: Kind` — it is **not**
object-safe (associated types + the typed `demand` signature). `Node<B>`
needs to store *heterogeneous* ops in one `Arc<dyn ...>`, so it stores
`Arc<dyn AnyOperation<B>>` instead. The blanket `impl<T: Operation<B>>
AnyOperation<B> for T` is the **only** place the typed↔erased conversion
happens:

- `demand_erased` downcasts the erased `WorkUnit` to `T::Output::WorkUnit`
  via `WorkUnitFor::typed(out).expect(...)`, calls the typed `demand`, and
  the results are already erased `Option<WorkUnit>` (each input's `demand`
  return is `WorkUnit`, already erased — only the *output* shape needed
  downcasting).
- `output_kind` wraps `output_spec()` in `Arc<dyn AnyKind>`.
- `dyn_hash` chains the op's own hash *and* its output spec's hash.

**You never write an `impl AnyOperation` by hand.** Implement `Operation<B>`
+ `Lower<B>`, and the bridge is free. `CLAUDE.md` §3/§7 forbids making the
typed traits object-safe or bypassing this bridge — if you find yourself
writing `dyn Operation<B>` anywhere, that's the bug.

### 3.4 `Source<B>` / `AnySource<B>` / `Target<K,B>` — `src/io.rs`

```rust
pub trait Source<B: Backend>: Send + Sync + 'static {
    type Kind: Kind;
    fn spec(&self) -> Arc<Self::Kind>;
    fn fetch(&self, ctx: &B::Ctx, wu: &<Self::Kind as Kind>::WorkUnit) -> Result<Buffer<B>, Error>;
    fn lower(&self, cx: &mut B::Builder);
    fn dyn_hash(&self, state: &mut dyn Hasher);
}
```

A `Source<B>` is a DAG **leaf** — "data enters the model here". Examples:
`FileImageSource` (vips: opens a file), `VipsImageSource` (GPU: pulls a vips
pipeline's result and uploads it), `GpuConstantSource` (GPU: a literal `f32`
array, used for test images / convolution kernels), `RawFileImageSource`
(libraw).

- `spec()` — the Kind this source produces (its dimensions etc. are usually
  known up front, e.g. from the file header).
- `fetch(ctx, wu)` — eagerly produce a `Buffer<B>` for the given typed
  `WorkUnit`. This exists for callers that want a buffer *without* going
  through the lazy graph (and is what `VipsImageSource::lower` calls
  internally to bridge backends — see §10).
- `lower(cx)` — the lazy path: register this source's data with the
  `Builder` (GPU: call `cx.input(view, params, buffer)`; vips: call
  `cx.emit(handle)`).
- `AnySource<B>` is the object-safe mirror, with the same blanket-impl
  pattern as `AnyOperation` (`fetch_erased` downcasts `wu` via
  `WorkUnitFor::typed`).

`Node::Source(Arc<dyn AnySource<B>>)` is the other variant of `Node<B>`
(alongside `Node::Op`, §3.5) — a DAG leaf.

```rust
pub trait Target<K: Kind, B: Backend>: Send + Sync {
    type Out;
    fn extract(&self, buf: &Buffer<B>, wu: &K::WorkUnit, ctx: &B::Ctx) -> Result<Self::Out, Error>;
}
```

A `Target<K,B>` is the **only door out**. `extract` takes the materialized
`Buffer<B>` and produces `Self::Out` — could be `Vec<u8>` (download to RAM,
`RamImageTarget`), `()` (write-to-disk side effect), or even
`Buffer<B>` itself (a "viewport" target that just clones the `Arc` — still
resident, no download, but lets a caller hold onto the result without going
through `Data::push` again). `Data::materialize` is `pub(crate)` precisely so
that **every exit is a `Target`** — see §3.5.

### 3.5 `Node<B>` / `Data<K,B>` / `GraphWalk` — `src/node.rs`

```rust
pub enum Node<B: Backend> {
    Op(Arc<dyn AnyOperation<B>>),
    Source(Arc<dyn AnySource<B>>),
}

pub struct Data<K: Kind, B: Backend> {
    pub root: Arc<Node<B>>,
    pub ctx: Arc<B::Ctx>,
    pub spec: Arc<K>,
}
```

`Node<B>` is one DAG vertex — either a computation (`Op`) or a leaf
(`Source`). It has delegated methods (`output_kind`, `lower`, `inputs`,
`demand_erased`) that dispatch on the `Op`/`Source` variant, so **generic
walk code never matches on `Node::Op` vs `Node::Source`** (invariant #8).

`Data<K,B>` is **the user-facing handle** — `Image2D<B>` is
`type Image2D<B> = Data<ImageKind, B>;` (a type alias, per the FORBIDDEN list
— there is no separate `Image2D` struct). Key methods:

- **`push<Op, K2>(&self, op: Op) -> Data<K2, B>`** — the *only* way to extend
  the DAG. Wraps `op` (which already holds its `Input`s, i.e. `Arc::clone`s
  of upstream nodes) in a new `Arc<Node::Op>`. The old DAG is untouched —
  immutable, structurally shared (invariant #8). This is what `.blur(3.0)`
  etc. call under the hood (see `Image2D::blur` in §8.3).
- **`as_input(&self) -> Input<K,B>`** — wraps `self` as an edge for the
  *next* op's `inputs()`.
- **`materialize(&self, wu) -> Result<Buffer<B>, Error>`** — `pub(crate)`,
  **never call this directly** (and never make it `pub` — invariant #1).
  Runs the full demand+lower walk (§4) and returns the backend-resident
  result.
- **`pull<T: Target<K,B>>(&self, target: &T, wu) -> Result<T::Out, Error>`**
  — the only public way to get a result: `materialize` then
  `target.extract(...)`. This is what every test (`common::poc_materialize`,
  `common::vips_materialize`) and every real exit calls.
- **`reinterpret<T>()` / `reinterpret_with<T>(spec)`** — zero-cost typed
  casts via `ReinterpretAs` (§3.1) / an explicit target spec, pushing a
  `Reinterpret<K,T,B>` node (§8.4).
- **`from_source<S: Source<B, Kind=K>>(source, ctx) -> Self`** — builds a
  fresh `Data` whose root is a `Node::Source`. This is how `Image2D::open`
  (vips) and `VipsImageSource`-backed GPU images are constructed.

#### `GraphWalk<'a, B>` — the only traversal primitive

```rust
pub struct GraphWalk<'a, B: Backend> {
    pub root: &'a Arc<Node<B>>,
    pub demands: HashMap<NodeId, WorkUnit>,
    pub lowered: HashSet<NodeId>,
}
```

`NodeId(usize)` is `Arc::as_ptr(node) as usize` — pointer identity, stable
for the immutable graph's lifetime, and the *only* way nodes are
deduplicated (invariant #8: "every walk must dedup ... and MUST use the
generic `GraphWalk`"). Two methods:

- **`demand(&mut self, root_wu)`** — iterative (explicit `Vec` stack, not
  recursion) reverse walk. Starting from the root's requested `WorkUnit`,
  pushes `(node, wu)` pairs; for each, merges `wu` into
  `self.demands[NodeId::of(node)]` via `WorkUnit::union` (§3.2). If the
  merge *grew* the entry (first visit, or a second consumer's demand wasn't
  already covered), call `node.demand_erased(&wu)` and push each non-`None`
  child with its propagated demand. **Diamonds**: a node reached twice with
  overlapping demands is only re-expanded if the union actually grew —
  otherwise dense diamonds would re-walk their whole upstream subgraph once
  per path.
- **`lower<F>(&mut self, enter_and_lower: F)`** — recursive post-order walk
  *of the same tree*, but now driven by `self.demands` (computed above) and
  `self.lowered` (a `HashSet<NodeId>` — each node lowered **exactly once**,
  invariant #8). For a node with no entry in `demands`, it's pruned —
  skipped entirely. For each input, recurse first (so inputs are always
  lowered before their consumers — `VipsBuilder::input` and
  `GpuBuilder::enter`'s `last_step_of`/`source_of` lookups both rely on
  this), then call `enter_and_lower(node, &resolved_wu)`.

#### `materialize<B>` — ties it together

```rust
pub(crate) fn materialize<B: Backend>(ctx, root, wu) -> Result<Buffer<B>, Error> {
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

Demand walk first (fills `walk.demands`), then lower walk (drains it into the
builder via `enter` + `node.lower(&mut builder)` per node), then
`builder.finish(...)`. This function is the **entire** evaluation engine —
everything backend-specific happens inside `enter`/`lower`/`finish`.

### 3.6 `Buffer<B>` / `Backend` / `Builder<B>` — `src/buffer.rs`, `src/backend/mod.rs`

```rust
pub struct Buffer<B: Backend> {
    pub payload: Arc<B::Payload>,
    pub spec: Arc<dyn AnyKind>,
}

pub trait Backend: Sized + Send + Sync + 'static {
    type Ctx: Send + Sync;       // GpuContext / ()
    type Payload: Send + Sync;   // GpuBuffer / VipsHandle
    type Builder: Builder<Self>; // GpuBuilder / VipsBuilder
}

pub trait Builder<B: Backend>: Sized {
    fn new(ctx: Arc<B::Ctx>) -> Self;
    fn enter(&mut self, node: NodeId, inputs: &[NodeId], wu: &WorkUnit);
    fn finish(self, root: NodeId, spec: Arc<dyn AnyKind>, root_wu: &WorkUnit) -> Result<Buffer<B>, Error>;
}
```

`Buffer<B>` is the *result type* of materialization — a backend payload
(`Arc<GpuBuffer>` for GPU, `Arc<VipsHandle>` for vips) plus the erased Kind
that describes it. `Backend` just names the three associated types per
backend. `Builder<B>` is the **three-method seam** the core walk uses:
`new` (one per `materialize` call), `enter` (once per node, in post-order,
*before* `node.lower(&mut builder)`), `finish` (once, at the very end). Every
backend-specific accumulation strategy (GPU: fused-shader state machine;
vips: node-keyed handle map) lives behind these three calls.

---

## 4. The materialize algorithm, end to end

Putting §3.5/§3.6 together, here's literally what happens on
`data.pull(&target, wu)`:

```
1. materialize(ctx, root, wu):
     a. GraphWalk::demand(wu)
        - stack = [(root, wu)]
        - pop (node, wu); union into demands[node]; if grew:
            - children = node.demand_erased(wu)   // <-- Operation::demand, per op
            - push (child_input.src(), child_wu) for each Some(child_wu)
        - repeat until stack empty
        => demands: NodeId -> WorkUnit   (every node that contributes, with its
           ACCUMULATED required region/range/atomic)

     b. builder = B::Builder::new(ctx)
     c. GraphWalk::lower(|node, wu| { builder.enter(...); node.lower(&mut builder) }):
        - post-order over root, skipping nodes absent from `demands` (pruned),
          visiting each present node exactly once (lowered: HashSet<NodeId>)
        => builder accumulates backend state, one enter+lower call per live node,
           inputs always processed before their consumers

     d. builder.finish(NodeId::of(root), root.output_kind(), wu)
        => Buffer<B>

2. target.extract(&buf, &wu, &ctx) => T::Out
```

Two things to internalize:

- **`demand` only ever sees `WorkUnit`s — never backend state.** It's pure
  geometry. `lower` is where backend state accumulates, but it *also* never
  does geometry — it reads `cx.wu()` (the *already-resolved* `WorkUnit` for
  this node, computed by step (a)) and turns that into builder calls.
- **A diamond** (two ops sharing one upstream node) is demanded twice (union
  merges the requests) but lowered once. On the GPU this means the shared
  node's kernel step is emitted once and both downstream steps read its
  `work_{k}` temp by index (§5.4); on vips it means the shared node's
  `VipsHandle` is computed once and both consumers' `cx.input(...)` look it
  up from `VipsBuilder::outputs`.

---

## 5. The GPU backend (`src/backend/gpu/`)

The GPU backend's whole job: turn a subgraph into **one fused Slang compute
shader** + one `dispatch_workgroups` call. "Fused" means: N op-nodes →
(ideally) N kernel steps in **one** `main()`, sharing bindings, each writing
its own scratch temp that the next step (or the final encode) reads.

### 5.1 `GpuContext` — `context.rs`

Created once, shared via `Arc` across all GPU `Data` handles (`Data::ctx`).
Holds:

- `device` / `queue` — the wgpu handles.
- `pipeline_cache: RwLock<LruCache<u64, Arc<CachedPipelines>>>` — **the only
  engine-owned cache** (invariant #9). Keyed by a `u64` that is the emitted
  Slang text's hash **XOR**ed with a `shader_fingerprint()` (a hash of every
  `.slang` file's path+contents under `shaders/`, memoized in a
  `OnceLock`). The XOR matters: a kernel's *body* never appears in the
  emitted `main()` (it's `import`ed and called by name) — without the
  fingerprint, editing `shaders/ops/gaussian_blur.slang` wouldn't change the
  emitted text's hash, and a stale compiled pipeline would silently survive.
- `max_storage_buffers` / `wg_dim` — device limits. `wg_dim` is 32 normally,
  16 on weaker GPUs (`max_compute_invocations_per_workgroup < 1024`) — this
  is the `numthreads(wg_dim, wg_dim, 1)` in every emitted shader.
- `allocated_bytes` — VRAM accounting (diagnostics only).

`max_storage_buffers` is read but the "split a fused pass when it would
exceed the binding limit" logic (`CutFinder` in the doc comment) is **not
implemented yet** — currently a fused pass's binding count is
`2 + work_buffers + sources` (§5.4) and there's no automatic splitting if
that exceeds the device limit. Something to watch if you fuse a very long
chain with many distinct source images.

### 5.2 `GpuBuilder` — the fused-pass state machine (`mod.rs`)

This is the biggest piece of per-backend state in the crate. One
`GpuBuilder` is built per `materialize` call and accumulates *everything*
needed to emit one shader. Read it as a state machine driven by the lower
walk's `enter` → `node.lower(&mut builder)` → (next node) `enter` → ... loop.

**Fields that persist across the whole pass:**

- `input_views: Vec<View>` — one per **source** node lowered so far (in
  binding order). A `View` (from `view.rs`, §5.3) is the Slang wrapper type +
  ctor expression for decoding that source's buffer.
- `steps: Vec<Step>` — one per **kernel invocation**, in topo (= lowering)
  order. `Step { module, kernel, inputs: Vec<StepInput>, params: Vec<String>,
  temp_elem: TempElem }`.
- `output: Option<OutputWrap>` — set by the **last** `cx.output(...)` call
  (the root's, since lowering is post-order and the root lowers last)
  describes how the final result is written (§5.3).
- `source_buffers: Vec<Arc<GpuBuffer>>` — the actual uploaded/fetched buffers
  for each source, parallel to `input_views`.
- `params: ParamBlock` — the **one** `ChainParams` SSBO's accumulated fields
  + raw bytes (std430 layout). Every scalar any op/source/adapter needs ends
  up here.
- `dispatch: Option<(u32,u32)>` / `dispatch_explicit: bool` — the
  `dispatch_workgroups` grid size, in elements (pixels). Defaults from the
  root's `Region` (`output()` sets it unless `dispatch()` was called
  explicitly first — used by reductions, §5.6).

**Per-node bookkeeping (consulted/updated on every `enter`):**

- `source_of: HashMap<NodeId, usize>` — which source slot a `Source` node
  became.
- `last_step_of: HashMap<NodeId, usize>` — which step index an `Op` node's
  *last* kernel call produced (so a downstream consumer reads `work_{that
  index}`).
- `alias_adapters: HashMap<NodeId, StepInput>` — for zero-cost nodes
  (`adapt`/`forward`, §5.2.3), what a downstream consumer should read
  *instead of* a step/source.

**Per-`enter` transient state** (reset every call):

- `cur_node`, `cur_inputs: Vec<StepInput>` (this node's resolved input edges
  — looked up from `alias_adapters`/`source_of`/`last_step_of` for each
  `input_keys` entry passed to `enter`), `cur_started` (has this node added
  its first kernel step yet — controls whether the *next* `kernel()` call
  reads `cur_inputs` (graph inputs) or the previous step's temp), `cur_output_adapter` (set if this node is a pure view-adapted
  alias — in case it ends up being the DAG root), `current_wu` (this node's
  resolved `WorkUnit`, returned by `cx.wu()`).

#### 5.2.1 The "registration" calls an op's `lower` makes

These are the only things `impl Lower<GpuBackend>` ever calls (besides
reading `cx.wu()` for scaling):

- **`cx.kernel(module, entry)`** (or `kernel_with_temp` for a non-`float4`
  temp, §5.6) — appends a `Step`. First call for this node: its inputs are
  `cur_inputs` (the node's graph inputs, resolved by `enter`). Later calls
  (intra-node multistep, e.g. separable filters): inputs = `[Step(prev)]`
  (this node's own previous step's temp). **This is how fusion happens
  automatically** — `cur_inputs` for node N+1 resolves (via
  `last_step_of`) to node N's last step, so consecutive ops in the DAG just
  chain steps in one shader with zero extra Rust code.
- **`cx.param(name, value)`** — call **after** `kernel()`. Adds a
  `s{step_idx}_{name}` field to `ChainParams` (step-namespaced — see the
  "Bug #1" callout in §5.2.4) and to that step's trailing kernel-call args
  (`params[0].s{idx}_{name}`).
- **`cx.param_block(block: ParamBlock)`** — merges a whole pre-built
  `ParamBlock` (e.g. several related scalars) into `ChainParams`; field names
  are **not** namespaced (caller's responsibility — see §5.2.4).
- **`cx.output(wrap: OutputWrap)`** — registers how this node's value is
  written if it's the root. Always called, even by non-root nodes (later
  `output()` calls from downstream nodes overwrite it — post-order means the
  *last* call, the root's, wins; `remove_fields_named` strips a source
  leaf's placeholder `region_out` etc. so the real root's don't collide).
- **`cx.dispatch(w, h)`** — pin the dispatch grid explicitly (reductions:
  histogram/vectorscope dispatch over the *input* image's size, but their
  *output* is `Atomic`-shaped so `output()` can't infer it).
- **`cx.fail(err)`** — record a fatal error (e.g. wrong `WorkUnit` shape);
  surfaced by `take_error()` in `finish`.

#### 5.2.2 `cx.input(view, slot_params, buf)` — only `Source::lower` calls this

Registers a new source slot: pushes `view` onto `input_views`, `buf` onto
`source_buffers`, merges `slot_params` (with `{slot}` replaced by the new
slot index — e.g. `"region_in_{slot}"` → `"region_in_0"`) into
`ChainParams`, and records `source_of[cur_node] = slot`. Also sets
`cur_output_adapter = Source(slot)` — in case this leaf is the *entire* DAG
(no ops at all, e.g. `Image2D::from_source(...).pull(...)`), the encoder
reads straight from this slot (§5.4's "n_steps == 0" case).

#### 5.2.3 `cx.adapt(adapter)` / `cx.forward()` — zero-cost view nodes

Some ops add **no kernel step** — they just change how the *next* reader sees
the existing value:

- **`forward()`** — "my value IS my input's value, byte for byte". Used by
  `Reinterpret` (§8.4): no Slang, no params, the typed cast is purely a Rust
  type-system fiction; `alias_adapters[cur_node] = cur_inputs[0]`, and if
  this node is the root, `cur_output_adapter` is the same.
- **`adapt(ViewAdapter)`** — "my value is my input's value wrapped in a
  Slang adapter struct" (e.g. `SwizzleView<{inner}>` for `ExtractBand`'s
  single-band case, §8.3's example). Allocates a fresh `ChainParams` prefix
  `a{n}` for the adapter's own params (so two adapters never collide), then
  same `alias_adapters`/`cur_output_adapter` wiring as `forward`.

A downstream node's `enter` resolves its inputs by checking
`alias_adapters` **first** — so a chain of `flip().extract_band(0)` costs
**zero** kernel steps; the eventual real kernel (or the encoder, if the chain
ends here) reads through both adapters via nested `{inner}`/`{value}`
template expansion (see `emit::read_expr`, §5.4).

#### 5.2.4 `ParamBlock` / `ChainParams` — the shared scalar SSBO

`ParamBlock` (`view.rs`) is `{ fields: Vec<(name, slang_ty)>, field_sizes:
Vec<usize>, bytes: Vec<u8> }` — a std430-layout accumulator. **All** of
`GpuBuilder.params` becomes **one** Slang `struct ChainParams { ... }`,
declared once, bound as `StructuredBuffer<ChainParams> params` (binding 1,
always `params[0]`).

⚠️ **The collision rule** (documented inline at `cx.param`, and the subject of
a real bug fixed during Blur's development): two different steps/nodes that
both push a field named e.g. `"sigma"` via `param_block` (un-namespaced) end
up with **two entries in `params.bytes`** but Slang's `struct ChainParams`
declaration **dedupes by name to one field** (`emit_slang`'s
`HashSet<String> seen` over `builder.params.fields`). Result: every field
*after* the duplicate is read from the wrong byte offset — silent
miscompilation (symptom we hit: an `idx >= domain.width` guard read garbage
and discarded every thread, producing an all-zero output).

- **`cx.param(name, value)`** (call after `kernel()`) is **namespaced**
  (`s{step_idx}_{name}`) — always safe between different steps.
- **`cx.param_block(...)`** is **not** namespaced — only use it for fields
  that are provably unique crate-wide (geometry fields like `region_in_0`,
  `region_out`, `domain`, or an op-specific name used by exactly one kernel
  family, e.g. `bandfold_kernel`'s `factor`/`in_bands`/`out_bands`). If two
  ops sharing a fused pass might both want a generically-named scalar
  (`sigma`, `radius`, `gain`, ...), use `cx.param` instead.

`RegionParams { stride, x, y, w, h }` (also in `view.rs`) is the
`BufferRegion` struct every Slang region wrapper indexes with —
`RegionParams::tight(w,h).into_block("region_out")` is the standard way an
image Kind's `output()` describes its output geometry; `"{slot}"` templating
(`"region_in_{slot}"`) is how each source gets its own `region_in_N`.

### 5.3 The View vocabulary — `view.rs`

This file is **pure data** — strings and byte blobs that `emit.rs` splices
verbatim. None of it is interpreted by Rust; it's all Slang source fragments.

- **`View { buffer_type, slang, ctor }`** — describes one buffer slot's
  *Slang-side* shape: `buffer_type` is the raw `StructuredBuffer<T>` element
  type (`"uint"`, `"float4"`, ...); `slang` is the wrapper struct the kernel
  function receives (`"CodecRegion<U8Codec, 0>"`, `"HistogramOut"`, ...);
  `ctor` is a template string with `{buf}`/`{params}`/`{region}`/`{slot}`
  placeholders, expanded by `input_expr`/`init_expr`.
- **`GpuView` trait** (the Kind capability, defined in `mod.rs` but
  conceptually part of this vocabulary):
  ```rust
  pub trait GpuView: Kind {
      fn input(&self) -> View;                       // decode wrapper
      fn output(&self, wu: &WorkUnit) -> OutputWrap;  // encode / direct write
  }
  ```
  **A Kind owns its codec.** `ImageKind::input()` returns a `CodecRegion<U8Codec|U16Codec|F32Codec,
  CH>` (decodes the pixel format to working `float4`); `ImageKind::output()`
  returns the *codec sandwich* (§5.4) — write `float4` to scratch, then
  `RWCodecRegion<...>` re-encodes. `HistogramKind::output()` instead returns
  a **direct** atomic-accumulate wrapper (`HistogramOut`, no scratch, no
  encode — `OutBuffer::Target`). **Ops never decode/encode** — they read/write
  whatever `IRegion`-implementing struct their input/output `View` says, and
  the Kind chose that struct.
- **`OutputWrap { arg, dest, encode, params }`** — `dest: OutBuffer::Scratch`
  (image: kernel writes `RWRegion` scratch, then `encode: Some(View)`
  re-encodes) vs `OutBuffer::Target` (reduction: kernel writes `arg` directly
  into the target buffer, `encode: None`).
- **`ViewAdapter { wrapper, ctor, params, module }`** — the zero-cost wrapper
  template for `cx.adapt(...)` (§5.2.3). `wrapper` has an `{inner}`
  placeholder (the wrapped value's Slang type); `ctor` has `{value}`
  (wrapped value's variable name) and `{params}`. `swizzle_adapter(channel)`
  in `bands.rs` is the canonical example — `SwizzleView<{inner}>`.
- **`TempElem { buffer_ty, region_wrapper, byte_size }`** — describes a
  step's `work_{k}` scratch buffer's element type. `TempElem::F4` (the
  default: `float4` / `RWRegion` / 16 bytes) covers all image ops. A
  reduction step can use `kernel_with_temp` with a different `TempElem` (e.g.
  `uint` bins) — though in practice histogram/vectorscope write directly to
  the target (`OutBuffer::Target`) and never allocate a `work_{k}` at all.
- **`SlangScalar`** (`f32`→`"float"`, `u32`→`"uint"`, `i32`→`"int"`) — lets
  `ParamBlock::param`/`cx.param` infer the Slang field type from the Rust
  type, so you can't accidentally declare `float sigma` and write an `i32`'s
  bytes into it.

### 5.4 `emit.rs` — turning `GpuBuilder` into Slang text

`emit_slang(builder, wg_dim) -> String` is a **pure function of the
builder's final state** — no datatype/op-specific branches, by construction
(every op/Kind already expressed itself as `View`/`OutputWrap`/`Step`/`ParamBlock`
data). Structure of the emitted text:

```text
import lib.region; import lib.io; import lib.codecs; import lib.pixel;  // CORE_MODULES, always
import ops.gaussian_blur;            // referenced_modules(): one per distinct step.module
                                      // + any adapter.module, deduped, CORE_MODULES excluded

struct ChainParams {                 // deduped by field name (the collision hazard, §5.2.4)
    BufferRegion domain;
    BufferRegion region_in_0;
    BufferRegion region_out;
    float s0_sigma;
    int s0_radius;
    ...
};

[[vk::binding(0,0)]] RWStructuredBuffer<uint> target_buffer;          // Slot::Target
[[vk::binding(1,0)]] StructuredBuffer<ChainParams> params;            // Slot::Params
[[vk::binding(2,0)]] RWStructuredBuffer<float4> work_0;               // Slot::Work(k) — one per
                                                                       //   work_buffer_count()
[[vk::binding(3,0)]] StructuredBuffer<uint> src_0;                    // Slot::Source(i) — one per
                                                                       //   input_views

[shader("compute")]
[numthreads(32, 32, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID) {
    uint2 idx = dispatchThreadID.xy;
    if (idx.x >= params[0].domain.width || idx.y >= params[0].domain.height) { return; }

    CodecRegion<U8Codec, 1> in_0 = { src_0, params[0].region_in_0 };   // one per input_views

    // step 0 (e.g. blur_kernel)
    RWRegion out_0 = { work_0, params[0].domain };          // non-final step -> own temp
    blur_kernel(idx, in_0, out_0, params[0].s0_sigma, params[0].s0_radius);

    // (more steps...)

    // final step's encode (image codec sandwich close)
    RWCodecRegion<U8Codec, 1> enc = { target_buffer, params[0].region_out };
    enc.write(idx, out_N.read(idx));
}
```

Key mechanics:

- **`slots(builder)`** (`Slot::Target, Params, Work(0..n), Source(0..m)`) is
  *the single source of truth* for binding indices — `emit_slang` (variable
  declarations), `compile::compile` (bind-group-layout `read_only` flags:
  `Target`/`Work` are RW, `Params`/`Source` are read-only), and
  `compile::dispatch` (actual `BindGroupEntry`s + work-buffer sizing) all
  iterate it identically. **Never hardcode a binding number anywhere else.**
- **`n_work = work_buffer_count()`** = `steps.len()` if the output needs the
  scratch sandwich (`OutBuffer::Scratch`), else `steps.len() - 1` (the last
  step writes the target directly — `OutBuffer::Target`, e.g. histogram).
- **Step inputs are resolved by `read_expr`**: `BaseInput::Source(i)` → the
  predeclared `in_{i}` (no extra decl); `BaseInput::Step(j)` → declares
  `{region_wrapper} r_{s}_{k} = { work_{j}, params[0].domain };` reading the
  prior step's temp. An `adapter` wraps either of these in
  `{wrapper}<{inner}> var = {ctor};` with `{value}`/`{params}` substituted.
- **Each step's output** is its own `work_{s}` temp (`RWRegion out_s = {
  work_s, params[0].domain };`) **unless** it's the final step of a *direct*
  (non-scratch) output, in which case it constructs the *target* wrapper
  directly (`output.arg.init_expr(...)`) — this is how a reduction's last
  step writes straight into `target_buffer` with no intermediate `work_{}`.
- **The codec-sandwich close** (image outputs): after the final step, declare
  `enc = { target_buffer, params[0].region_out }` using `output.encode`'s
  view, and `enc.write(idx, out_N.read(idx))` (or, if the root is a pure
  adapter of the final temp — e.g. `.flip()` immediately before `.pull()` —
  read through `cur_output_adapter` instead).
- **Zero-step pass** (`n_steps == 0`, e.g. a bare opened image, or an image
  with only zero-cost adapter ops applied): no step loop ran at all; encode
  reads directly from `cur_output_adapter` (an adapted source) or
  `cur_inputs[0]` (a plain source) — same `read_expr` machinery.

`hash_slang(slang) -> u64` is `DefaultHasher` over the text — the raw
half of the pipeline-cache key (XORed with `shader_fingerprint()`, §5.1).

**Diamonds, concretely**: if node X feeds both node Y and node Z, X is
lowered once (one `Step`, say index 2) — `last_step_of[X] = 2`. When Y and
Z each `enter`, their `cur_inputs` both resolve to `StepInput { base:
Step(2), .. }`. `emit_slang` therefore emits **two** `r_{s}_{k}` declarations
(one in Y's step, one in Z's step) both reading `work_2` — the *computation*
of X happened once, but each reader re-reads the temp. This is intentional
and cheap (a buffer read, not recomputation).

### 5.5 `compile.rs` / `slang.rs` — JIT, cache, dispatch

`compile(ctx, builder, slang, hash)`:

1. `hash ^= shader_fingerprint()` (§5.1).
2. Pipeline cache hit → return cached `(bgls, pipeline)` immediately.
3. Miss → `compile_spirv` (FFI into `SlangCompiler::compile_ir`, `slang.rs`):
   - Disk-cached by hash at `{OUT_DIR}/{hash:016x}.opt.spv` (checked before
     *and* after acquiring the global Slang lock — Slang's C++ API is **not**
     reentrant, so all `createSession`/`loadModuleFromSource`/`link`/
     `getTargetCode` calls across the whole process share **one
     `std::sync::Mutex<Option<SlangState>>`**). Each call gets a unique
     module name (`m{call_id:016x}`) — the disk cache is the real
     dedup layer, not Slang's session dictionary.
   - SPIR-V is written to disk *before* releasing the lock (so concurrent
     callers' double-check finds it and doesn't re-register the same module
     name), then `spirv-opt` (performance passes + strip debug info) runs
     **outside** the lock and overwrites the disk cache entry.
4. Build the bind-group layout from `emit::slots(builder)` (read-only flags
   per `Slot` variant, as above), the pipeline layout (one BGL, group 0), and
   the compute pipeline (`entry_point = "main"` always).
5. Cache `(bgls, pipelines)` in `ctx.pipeline_cache` under `hash`.

`dispatch(ctx, pass, builder, out_bytes, dims)`:

- Allocates the output buffer (`out_bytes.max(16)`, sized by
  `AnyKind::byte_size(root_wu)` — computed back in `GpuBuilder::finish`).
- Uploads `builder.params.bytes` as the `ChainParams` SSBO (binding 1).
- Allocates one `work_{k}` buffer per `Slot::Work`, sized
  `dims.0 * dims.1 * elem.byte_size` (so `(2*radius+1)²`-window kernels like
  `blur_kernel` don't need larger temps — they read the *source*, not a
  temp, see §10).
- Builds the bind group (`bg_entries`) by iterating `slots()` again,
  resolving `Slot::Source(i)` to `builder.source_buffers[i]`.
- `dispatch_workgroups((dims.0+wg-1)/wg, (dims.1+wg-1)/wg, 1)` — one
  `dispatch_workgroups` call, ever, per `materialize`. **This is why a
  fused multi-step pass cannot have a step read a *neighbor thread's*
  temp from an earlier step** — there's no barrier between workgroups
  within one dispatch. (This is exactly the bug that motivated Blur's
  single-pass-2D-kernel design, §10.)

### 5.6 Reductions (`HistogramKind`/`VectorscopeKind`) — the `Atomic` pattern

These are the canonical "GPU-only, `Atomic`-shaped" Kinds (`CLAUDE.md`'s
invariant #6 example — no `VipsBand` impl, so `Data<HistogramKind,
VipsBackend>` doesn't compile). Their `Lower<GpuBackend>`:

```rust
fn lower(&self, cx: &mut GpuBuilder) {
    let (w, h) = self.input.spec.dims();
    cx.dispatch((w.max(0) as u32, h.max(0) as u32));   // dispatch = INPUT's size
    cx.kernel("ops.histogram", "histogram_kernel").param("channel", self.channel);
    cx.output(self.output_spec().output(cx.wu()));     // OutBuffer::Target, no encode
}
```

`cx.dispatch(...)` must be called **before** `cx.output(...)` (which would
otherwise default `dispatch` from the `Atomic` output's — nonexistent —
`Region`); `dispatch_explicit = true` then prevents `output()` from
overwriting it. The kernel does `InterlockedAdd` directly into
`target_buffer` (one thread per input pixel, `HistogramOut` wrapper) — no
`work_{k}` temp, no encode step.

### 5.7 Pass splitting and demand tiling — `pass.rs`

Two GPU-specific mechanisms ensure a fused pass stays within hardware limits.
Both live entirely in `backend/gpu/pass.rs` — the agnostic core (`node.rs`,
`work_unit.rs`) is untouched. `Backend::materialize` provides the seam:
`GpuBackend` overrides it to route through `pass::gpu_materialize`, while
other backends inherit the default (`node::materialize`).

#### 5.7.1 CutFinder — binding budget enforcement (BFS)

Pipeline: **DAG → BFS analysis → find cuts → parallel pre-materialize →
rebuild DAG with wrappers → standard materialize on reduced DAG.**

A fused pass needs `2 + W + S` storage buffer bindings (`target` + `params` +
`W` work temps + `S` source inputs). When the full DAG would exceed
`ctx.max_storage_buffers`, `pass::find_cuts` walks the DAG **breadth-first**
from the root, grouping nodes by BFS level:

```
Level 0: root (Op7)
Level 1: Op5, Op6          ← independent at this level
Level 2: Op1, Op2, Op3, Op4
Level 3: S1, S2, S3, S4, S5, S6, S7, S8
```

For each candidate cut depth (shallow first), it computes what the remaining
pass would look like if everything at that depth were pre-materialized and
replaced by sources. The **shallowest** cut that fits is chosen — this
maximizes the **width** of independent sub-trees at the cut level, which
maximizes rayon parallelism.

Cut subgraphs are pre-materialized **in parallel** via `rayon::par_iter`.
Each result is wrapped as a `BufferImageSource` (a `Source<GpuBackend>` backed
by the pre-materialized `GpuBuffer`). The DAG is then **rebuilt** with
lightweight wrapper nodes:

- **`StubInput`** — implements `AnyInput<GpuBackend>`, holds a rebuilt child
  node + its Kind. Points `inputs()` at new (possibly cut-replaced) children.
- **`RebuiltOp`** — implements `AnyOperation<GpuBackend>`, wraps the original
  op. Delegates `lower`/`demand_erased`/`output_kind`/`dyn_hash` unchanged;
  only overrides `inputs()` with `StubInput`s. `GraphWalk` traverses the
  rebuilt tree naturally — no modifications to the walker.

Unchanged subtrees share the original `Arc<Node>` (structural sharing).
The rebuild is recursive — if a cut subgraph still exceeds limits, it cuts
again.

`GpuBuilder::finish` retains a binding-count assertion as a safety net: if
the CutFinder failed to reduce the pass, it fails loudly rather than
submitting an invalid bind group.

#### 5.7.2 Demand tiling (buffer-size enforcement)

`WorkUnit::split(max_bytes, calc_bytes)` (in `work_unit.rs`) is **pure
geometry** — no GPU types. It bisects `Region` along the longest axis and
`Range` at the midpoint, repeatedly, until every tile's byte cost (evaluated
by the caller-provided `calc_bytes` closure) fits under `max_bytes`. `Atomic`
work units cannot be subdivided and return `Err(InvalidWorkUnit)`.

---

## 6. The libvips backend (`src/backend/vips/`)

Vastly simpler than GPU, because **libvips already has a lazy, demand-driven
pipeline of its own** — `VipsImage` operations are themselves lazy
(`vips_image_new_from_file` doesn't decode; `vips_gaussblur` doesn't compute;
nothing runs until a sink like `vips_image_write_to_memory` pulls). The
Chromors vips backend's job is just to **wire our DAG's edges to vips's**.

### 6.1 `VipsHandle` / `VipsBuilder`

```rust
pub struct VipsHandle { pub(crate) ptr: *mut ffi::VipsImage }  // refcounted GObject wrapper

pub struct VipsBuilder {
    outputs: HashMap<NodeId, VipsHandle>,
    current: Option<NodeId>,
    current_wu: Option<WorkUnit>,
}
```

`VipsHandle` is `Clone` (bumps the GObject refcount) and `Drop`s (unrefs) —
a thin RAII wrapper, `unsafe impl Send + Sync` (vips images are
thread-safe-ish by convention here).

`VipsBuilder` is a **node-keyed handle map** — the vips analogue of
`GpuBuilder`'s `source_of`/`last_step_of`, but trivial because there's no
fusion to plan: each node just produces one `VipsHandle` and consumers look
it up by `NodeId`.

- **`enter(node, _inputs, wu)`** — `current = Some(node)`, `current_wu =
  Some(wu)`. (`_inputs` unused — vips doesn't need topology, only `outputs`
  lookups, and post-order guarantees inputs are already in `outputs`.)
- **`cx.input(src: &Arc<Node<VipsBackend>>) -> VipsHandle`** — looks up
  `outputs[NodeId::of(src)]`, `.clone()`s it (bumps refcount, since the op
  will hand the `ptr` to a vips operation that may take ownership semantics).
- **`cx.emit(handle: VipsHandle)`** — `outputs[current] = handle`. Every
  `impl Lower<VipsBackend>` ends with exactly one `cx.emit(...)`.
- **`cx.wu()`** — the resolved `WorkUnit`, mostly unused by vips ops (vips
  computes the whole lazy pipeline and lets *its own* demand-driven region
  system figure out tiling — our `WorkUnit` mainly matters for the *GPU*
  side and for `byte_size`/target extraction sizes).
- **`finish(root, spec, _root_wu)`** — `outputs.remove(root)` (must be
  `Some`, or it's a bug — "root node produced no handle") wrapped in
  `Buffer { payload: Arc::new(handle), spec }`.

### 6.2 `VipsBand` — the per-backend Kind capability

```rust
pub trait VipsBand: Kind {
    fn band_format(&self) -> i32;  // VipsBandFormat enum value
}
```

Symmetric to `GpuView`. `ImageKind::band_format()` maps `PixelFormat` to
`VIPS_FORMAT_UCHAR`/`USHORT`/`FLOAT` by bytes-per-sample.
`HistogramKind::band_format()` returns `VIPS_FORMAT_UINT` (vips histograms
are `bins × 1` images, `bands` bands, `uint`). A Kind with no `impl VipsBand`
(e.g. a hypothetical GPU-only point-list Kind) makes `Data<ThatKind,
VipsBackend>` a compile error — `Image2D<B>`-style ergonomic methods that
require `VipsBand` simply don't exist for it. This is invariant #6: **no
runtime "unsupported backend" error, ever**.

### 6.3 A typical `impl Lower<VipsBackend>`

The overwhelming majority look like `Invert`:

```rust
impl Lower<VipsBackend> for Invert<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"invert\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}
```

`VipsGObject` (`gobject.rs`) is a thin builder over `g_object_new` +
`g_object_set` (via `set_image`/`set_double`/`set_int`/...) + `vips_cache_operation_build`-style
`.run()`. Multi-input ops (`Bandjoin`) call `cx.input(...)` per `Input`, build
a `Vec<*mut VipsImage>`, and call the corresponding `ffi::vips_*` function
directly (when there's no convenient one-shot `VipsGObject` form).

### 6.4 CPU custom regions — `custom.rs` / `working.rs`

For operations with **no libvips equivalent**, the backend can register a
`VipsCustomOperation` — `custom.rs` provides `CustomRegion` (a raw
row-pointer view into a vips region's memory for a given rect), and
`working.rs` (`RegionView<P>` / `RegionViewMut<P>`) wraps it with **typed,
SIMD-friendly pixel decode/encode** via the `Pixel` trait
(`src/pixel/*`). `RegionProcessor::process<P: Pixel>(&self, src: &RegionView<P>,
dst: &mut RegionViewMut<P>)` is written **once, generically** over the
working pixel type `W` (via `.get::<W>()`/`.set::<W>()`), and
`execute_processor` dispatches the *physical* format `P` via
`dispatch_format!` (a macro matching `PixelFormat` → concrete `Rgba<u8>` /
`Rgb<f32>` / etc. — this is **the one place** a closed match over pixel
*storage* formats is acceptable, since it's CPU SIMD dispatch, not a
datatype-extensibility axis `CLAUDE.md` cares about). `RegionViewMut::set`
auto-preserves alpha from a `source` region if the working pixel type `W`
has none (`Rgb` written back into an `Rgba8` image keeps the original A).

---

## 7. Adding a datatype — recipe + what each piece is for

(`CLAUDE.md` §5 is the authoritative checklist; this is the "why" for each
step, with the canonical examples.)

1. **`struct FooKind { ...metadata... }`** + `impl AnyKind` (`byte_size`,
   `dyn_hash`) + `impl Kind { type WorkUnit = Region|Range|Atomic; }`. This is
   the *agnostic* identity of your datatype — no backend types allowed here.
   Look at `ImageKind` (Region-shaped, full colorimetric metadata) vs
   `HistogramKind` (Atomic-shaped, just `{bins, bands}`) for how minimal this
   can be — `HistogramKind` has **zero** color/format fields because (per
   `docs/new-datatypes.md`) "if the payload must not pass through the color
   pipeline, it is not an Image".
2. **`impl GpuView for FooKind`** (if GPU-supported) — `input()` (how a
   kernel decodes a `FooKind` buffer) and `output(wu)` (how a kernel's result
   becomes a `FooKind` buffer: sandwich-with-encode, or direct write).
   **and/or `impl VipsBand for FooKind`** (if vips-supported) —
   `band_format()`. A Kind needs *at least one* of these to be usable; **it's
   fine to have only one** — that's how `HistogramKind` is GPU-only
   *by the type system*.
3. **`pub type Foo<B = ...> = Data<FooKind, B>;`** — the user-facing alias.
   Never a hand-written struct (FORBIDDEN list).
4. **Operations producing it**: `struct SomeOp { input: Input<...>, ... }` +
   `impl Operation<B> for SomeOp` (the structural/geometric half — `inputs`,
   `demand`, `output_spec`, `dyn_hash`) + one `impl Lower<EachBackend> for
   SomeOp` per supported backend. Cross-Kind ops (input Kind ≠ output Kind,
   e.g. `HistogramOp: ImageKind → HistogramKind`) are completely ordinary —
   `Operation::Output` is just a different `Kind` than `self.input.spec`'s.
5. **Ergonomic methods**: `impl<B> Foo<B> { pub fn frobnicate(&self, ...) ->
   Self { self.push(SomeOp { input: self.as_input(), ... }) } }`. In-crate,
   plain inherent impls — no extension-trait ceremony needed.
6. **Slang** (GPU only, §8) — new kernel(s) in `shaders/ops/your_thing.slang`
   + any new `IRegion`-implementing wrapper struct in `shaders/lib/region.slang`
   if your `View`/`OutputWrap` needs one that doesn't exist yet.

**Zero edits required** to `kind.rs`, `operation/mod.rs`, `io.rs`, `node.rs`,
`backend/*/mod.rs`'s core builder logic, or `work_unit.rs`'s `union` (unless
you invent a genuinely new *shape*, which none of the 6 current datatypes
have needed — `ImageKind`/`Mask2DKind` use `Region`, `HistogramKind`/
`VectorscopeKind`/`Fft2DKind`(?)/`LutKind`(?) use `Atomic` or `Range` as
appropriate).

---

## 8. Adding an operation — recipe + four worked examples

### 8.1 The skeleton (from `CLAUDE.md` §6, annotated)

```rust
pub struct Foo<B: Backend> { input: Input<InKind, B>, /* params */ }

impl<B: Backend> Operation<B> for Foo<B> where Foo<B>: Lower<B> {
    type Output = OutKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &<OutKind as Kind>::WorkUnit) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]   // pass-through; expand for halos
    }
    fn output_spec(&self) -> OutKind { /* derive from self.input.spec */ }
    fn dyn_hash(&self, s: &mut dyn Hasher) { /* hash own params */ }
}

impl Lower<GpuBackend> for Foo<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.kernel("ops.foo", "foo_kernel");
        cx.param("k", self.k);                       // AFTER kernel()
        cx.output(self.output_spec().output(cx.wu()));
    }
}
impl Lower<VipsBackend> for Foo<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let h = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"vips_op_name\0").unwrap();
        op.set_image("in", h.ptr);
        cx.emit(op.run().unwrap());
    }
}
```

The `where Foo<B>: Lower<B>` bound on the generic `Operation<B> for Foo<B>`
impl is the standard pattern — it lets `Operation`'s structural half be
written **once, generically**, while `Lower` is implemented per-backend
separately. If `Foo` only supports GPU, simply don't write
`impl Lower<VipsBackend> for Foo<VipsBackend>` — then `Foo<VipsBackend>:
Operation<VipsBackend>` fails to hold (no blanket impl, since the `where`
bound isn't satisfied), and `Data<_, VipsBackend>::push(Foo { ... })` is a
compile error. Same mechanism as invariant #6, applied to operations instead
of Kinds.

### 8.2 Pointwise, no halo — `Invert` (`edge.rs`)

The simplest possible op: `demand` is pure pass-through
(`vec![Some(WorkUnit::Region(out.clone()))]`), `output_spec` returns the
input's spec unchanged (same format/size), `dyn_hash` is a no-op (no
parameters). GPU lowering is one line: `cx.kernel("ops.invert",
"invert_kernel"); cx.output(...)`. Vips lowering is the one-liner
`VipsGObject::new(b"invert\0")`. This is the template for ~80% of pointwise
ops (arithmetic, gamma, exposure, …).

### 8.3 Zero-cost view — `ExtractBand` (`bands.rs`)

When `count` is `None`/`1` (single-band extract), `Lower<GpuBackend>` calls
`cx.adapt(swizzle_adapter(self.band as u32))` **instead of** `cx.kernel(...)`
— no shader code runs for this node; it's purely a different read-expression
for whatever comes next (§5.2.3/§5.4). `swizzle_adapter`:

```rust
pub fn swizzle_adapter(channel: u32) -> ViewAdapter {
    ViewAdapter {
        wrapper: "SwizzleView<{inner}>".into(),
        ctor: "{ {value}, {params}[0].{p}_channel }".into(),
        params: ParamBlock::scalar("{p}_channel", channel),
        module: "lib.region",
    }
}
```

When `count > 1`, it falls back to a real kernel
(`extract_band_range_kernel`) with `param_block`-pushed `band`/`count`. Note
the **field-order contract** comment on `Bandfold`/`Bandunfold`/`Bandjoin`:
when a kernel takes multiple `param_block`-pushed scalars, the kernel
function's parameter order must match the `ParamBlock::new().param(...).param(...)`
call order **exactly** — `emit_slang` appends `params[0].{name}` for each
declared field in declaration order, positionally.

### 8.4 Zero-cost typed cast — `Reinterpret<K,T,B>` (`reinterpret.rs`)

The generic mechanism behind `Data::reinterpret()`/`reinterpret_with()`
(§3.5). `demand` debug-asserts byte-size equality between `K` and `T` for the
requested `WorkUnit` (`ReinterpretAs::reinterpret_spec`'s "contract", checked
at runtime in debug builds only — invariant: "byte-identical payload is the
impl's contract, not a runtime check"). `Lower<GpuBackend>` is just
`cx.forward(); cx.output(self.spec.output(cx.wu()))` — **no kernel**, the
node's value *is* its input's value, just tagged with a different `Kind` for
downstream type-checking. `Lower<VipsBackend>` is `cx.input(...)` then
immediately `cx.emit(...)` — handle passthrough.

> Note: `reinterpret.rs`'s GPU `lower` currently has a stray
> `eprintln!("DEBUG Reinterpret::lower called")` — leftover debug output, not
> part of the design; harmless but worth deleting next time you're in that
> file.

### 8.5 Halo + single-pass kernel — `Blur` (`filters.rs`)

The most instructive example, because it shows **`demand` and `lower` working
together** to implement a Gaussian blur *without* the separable-pass
architectural trap (§10 has the full story of why it's single-pass).

```rust
fn gauss_radius(sigma: f32) -> i32 { /* matches vips_gaussmat's min_ampl=0.2 mask */ }

impl<B: Backend> Operation<B> for Blur<B> where Blur<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let halo = gauss_radius(self.sigma / out.lod.scale_factor() as f32);
        vec![Some(WorkUnit::Region(out.expanded(halo)))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) { state.write_u32(self.sigma.to_bits()); }
}

impl Lower<GpuBackend> for Blur<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let scale = if let WorkUnit::Region(r) = &wu { r.lod.scale_factor() as f32 } else { 1.0 };
        let sigma = self.sigma / scale;
        cx.kernel("ops.gaussian_blur", "blur_kernel");
        cx.param("sigma", sigma);
        cx.param("radius", gauss_radius(sigma));
        cx.output(self.output_spec().output(cx.wu()));
    }
}
```

- `demand` asks upstream for `out.expanded(radius)` — `radius` extra pixels
  on every side, so the kernel can read `(2*radius+1)²` neighbors without
  going out of bounds *of the demanded region* (the shader itself still
  `read_clamped`s at the true image edges).
- `radius`/`sigma` are **rescaled by `out.lod.scale_factor()`** — at LOD 1
  (half-res), a "3-pixel-sigma" blur on the full-res image should look like a
  1.5-pixel-sigma blur on the half-res buffer.
- `Lower<VipsBackend>` is the one-liner `gaussblur` with `sigma` — vips
  computes its own mask radius the same way (`gauss_radius` was derived
  *from* vips's `vips_gaussmat` formula precisely so the two backends agree).

---

## 9. Adding a backend

Per invariant #5: a new `impl Backend` (its `Ctx`/`Payload`/`Builder`
associated types) + that `Builder`'s `impl Builder<NewBackend>` (the 3-method
seam, §3.6) + a new per-backend capability trait (the `NewBackend` analogue
of `GpuView`/`VipsBand`) that Kinds opt into. Every *existing* op gains the
new backend by adding **one** `impl Lower<NewBackend> for ExistingOp<NewBackend>`
each — `Operation<B>`'s generic structural impl (`where ExistingOp<B>:
Lower<B>`) needs no changes; it was written generically over `B` from the
start. `src/backend/raw/` (LibRaw) and `src/backend/vello/` (vector graphics)
are partial examples of this pattern already in the tree — `RawBackend` only
implements `Source`/`Target` for `ImageKind` (it's a *capture* backend, not a
*processing* one: `RawFileImageSource` + `RamImageTarget`, no `Lower<RawBackend>`
for any op), showing that a backend doesn't have to support the full
operation surface to be useful.

---

## 10. Worked trace: `img.invert().blur(3.0)` on GPU

Putting §4/§5 together for a concrete two-op pipeline.

**Build time** (no computation): `img: Image2D<GpuBackend>` (root =
`Node::Source(VipsImageSource{...})`, say 200×200 `Rgb8`).
`img.invert()` → `Data::push(Invert { input: img.as_input() })` → new root
`Node::Op(Invert)`, `spec` unchanged (`ImageKind` same format/size).
`.blur(3.0)` → `Data::push(Blur { input: <invert>.as_input(), sigma: 3.0 })`
→ new root `Node::Op(Blur)`. **Nothing has executed.**

**`pull(&RamImageTarget, Region::full((200,200), Lod(0)))`:**

1. **Demand walk**, stack starts at `(Blur, Region{0,0,200,200,lod0})`:
   - `Blur.demand_erased(region)` → `radius = gauss_radius(3.0) = 5` →
     `[Some(Region{-5,-5,210,210,lod0})]` → push `(Invert, expanded_region)`.
   - `Invert.demand_erased(expanded_region)` → pass-through →
     `[Some(expanded_region)]` → push `(Source, expanded_region)`.
   - Source has no inputs (`Node::inputs()` returns `[]` for `Source`).
   - `demands = { Blur: {0,0,200,200}, Invert: {-5,-5,210,210}, Source: {-5,-5,210,210} }`.

2. **Lower walk**, post-order from `Blur`:
   - Recurse into `Invert`, recurse into `Source`. `Source` has no children
     → `enter(Source, [], {-5,-5,210,210})` then `Source.lower(&mut
     builder)`: `VipsImageSource::lower` materializes the *upstream vips
     pipeline* for that region (a **separate**, nested `materialize::<VipsBackend>`
     call — this is the cross-backend bridge, §3.4), uploads the bytes, calls
     `cx.input(ImageKind::input(), RegionParams::tight(210,210).into_block("region_in_{slot}"), buf)`.
     This sets `input_views = [CodecRegion<U8Codec,1>]`, `source_of[Source] = 0`,
     `cur_output_adapter = Source(0)`.
   - Back to `Invert`: `enter(Invert, [Source], {-5,-5,210,210})` resolves
     `cur_inputs = [Source(0)]` (via `source_of`). `Invert.lower`:
     `cx.kernel("ops.invert", "invert_kernel")` → `Step{0, inputs:[Source(0)]}`,
     `last_step_of[Invert] = 0`. `cx.output(ImageKind::output(...))` — sets
     `output` (will be overwritten by `Blur`'s call, since `Blur` lowers
     last) and `region_out`/`domain` params for *this* 210×210 region (later
     stripped by `remove_fields_named` when `Blur` calls `output()` again).
   - Finally `Blur`: `enter(Blur, [Invert], {0,0,200,200})` resolves
     `cur_inputs = [Step(0)]` (via `last_step_of[Invert] = 0`). `Blur.lower`:
     `cx.kernel("ops.gaussian_blur", "blur_kernel")` → `Step{1, inputs:[Step(0)]}`.
     `cx.param("sigma", 3.0)` → field `s1_sigma`. `cx.param("radius", 5)` →
     field `s1_radius`. `cx.output(ImageKind::output({0,0,200,200}))` — sets
     the **final** `output` and `region_out` = `{stride:200,x:0,y:0,w:200,h:200}`,
     and (since not `dispatch_explicit`) `dispatch = (200, 200)`.

3. **`finish`**: pushes `domain = RegionParams::tight(200,200)`. `emit_slang`
   produces roughly:
   ```text
   import lib.region; import lib.io; import lib.codecs; import lib.pixel;
   import ops.invert; import ops.gaussian_blur;

   struct ChainParams {
       BufferRegion domain;
       BufferRegion region_in_0;
       BufferRegion region_out;
       float s1_sigma;
       int s1_radius;
   };
   [[vk::binding(0,0)]] RWStructuredBuffer<uint> target_buffer;
   [[vk::binding(1,0)]] StructuredBuffer<ChainParams> params;
   [[vk::binding(2,0)]] RWStructuredBuffer<float4> work_0;   // Invert's temp (work_buffer_count=2: scratch)
   [[vk::binding(3,0)]] StructuredBuffer<uint> src_0;

   void main(...) {
       ...
       CodecRegion<U8Codec,1> in_0 = { src_0, params[0].region_in_0 };

       RWRegion out_0 = { work_0, params[0].domain };
       invert_kernel(idx, in_0, out_0);                       // step 0: reads source, writes work_0

       RWRegion out_1 = { work_1, params[0].domain };  // <- actually the final step writes
                                                         //    target directly if no scratch needed,
                                                         //    but image output IS scratch, so:
       blur_kernel(idx, r_1_0 /* = work_0 read */, out_1, params[0].s1_sigma, params[0].s1_radius);

       RWCodecRegion<U8Codec,1> enc = { target_buffer, params[0].region_out };
       enc.write(idx, out_1.read(idx));
   }
   ```
   **One** `dispatch_workgroups` call, **one** pipeline, **two** fused
   kernels. `blur_kernel<R: IRegion>(idx, input: R, output, sigma, radius)`
   reads `input.read_clamped(idx + (x,y))` for `x,y ∈ [-radius,radius]`,
   where `R` is whatever `read_expr` resolved this step's input to.

   - If `input` is `BaseInput::Source` (a decode buffer, fully uploaded
     before the dispatch starts), every thread can safely read every other
     thread's source pixel — **no hazard**, regardless of workgroup
     ordering. This is the case in this trace: `blur_kernel`'s `input` is
     `Invert`'s `work_0`... which is **not** a source.
   - That's the crux of the remaining gap: when `Blur` is fused **after**
     another op (as here, after `Invert`), `blur_kernel`'s `(2*radius+1)²`
     neighbor reads target `work_0` — written by `invert_kernel`, a
     *different* thread/workgroup, in the *same* dispatch. wgpu/Vulkan gives
     no ordering or visibility guarantee across workgroups within one
     dispatch, so a neighboring workgroup's `work_0` write may not be visible
     yet when this thread reads it. The single-pass kernel fixes the
     **separable H/V** hazard (§ design note in `gaussian_blur.slang`) but
     **does not fully fix** the "blur fused after a prior step" case — that
     requires either splitting into multiple dispatches with a barrier
     between them, or restricting fusion so a halo-reading op is always
     first (reads only `Source`). **Not yet implemented** — `CLAUDE.md`'s
     "Multistep / fusion" guidance applies; treat any new op whose kernel
     does neighbor-reads of a *prior step's* `work_{k}` (not a `Source`) as
     suspect until a real fix lands. The cross-backend tests currently pass
     because they don't exercise `invert().blur(...)` specifically — this
     trace is illustrative of the *general* fusion mechanism, not a verified
     pipeline.

4. **`extract`**: `RamImageTarget::extract` calls
   `buf.payload.read_to_cpu(ctx)` — GPU→staging→CPU copy, `Vec<u8>` of
   200×200×3 bytes.

---

## 11. THE INVARIANTS (non-negotiable) and FORBIDDEN patterns

This is the **normative rule set** — `CLAUDE.md` §4/§6 summarize it, but this
section is the canonical full text. If `CLAUDE.md` and this section ever
diverge on wording, treat it as a doc-sync bug to fix (not a real
disagreement — `CLAUDE.md`'s short forms are meant to be exactly these rules,
condensed).

### 11.1 The ten invariants

1. **Data is always backend-resident.** No `residency` field, no host
   `Buffer` variant. The ONLY way to host is `Target::extract`.
   `Data::materialize` is `pub(crate)` — never make it public, never
   `.download()` outside a `Target`. → §3.5/§3.6, §6.1 (`finish` returns
   `Buffer<B>`, never raw bytes).

2. **The materializer is type-blind.** It walks `Arc<dyn AnyKind>` and
   **must never** read a `View`/`ParamBlock` from a Kind, nor downcast to a
   concrete Kind. Views/params are **injected by the node inside `lower`**
   (concrete-type site). You cannot cross-cast `dyn AnyKind` → `dyn GpuView`
   — do not try. → §4 (`materialize` is generic over `B`, never matches
   concrete Kinds), §5.2.1 (`lower` is where `View`/`ParamBlock` appear).

3. **One closed enum only: `Shape` (Region/Range/Atomic).** It is
   per-*shape* (3 materialize strategies), NOT per-datatype. The single
   allowed `match` over shapes lives in `work_unit.rs` (`WorkUnit::union`).
   Adding a datatype must add ZERO match arms anywhere. Adding a Slang
   wrapper type must add ZERO Rust enum variants. → §3.2.

4. **Adding a datatype is additive: one new file in `src/data/`.** No
   central enum, no `emit.rs` match, no edit to `AnyKind`/`Operation`/
   `Backend`. → §7.

5. **Adding a backend is additive.** A new `impl Backend` + its `Builder` +
   the per-backend capability trait (`GpuView`-equivalent). Existing ops
   gain it by writing one `Lower<NewBackend>` each; the structural
   `Operation<B>` impl is untouched. → §9.

6. **A Kind only supports a backend if it implements that backend's
   capability.** `HistogramKind` has `GpuView` and not `VipsBand` ⇒
   `Data<HistogramKind, VipsBackend>` does not compile. This is the intended
   mechanism — never add a runtime "unsupported backend" error. → §6.2,
   §7 step 2, §8.1.

7. **Region/dimensional math lives on the typed `WorkUnit`s**
   (`Region::bounding`, `Region::tile_aligned`), exposed to the generic
   engine via `WorkUnit::union` (the 3-arm shape switch). The materializer
   calls `wu.union(other)`; it does not do rect math itself. → §3.2, §4.

8. **The DAG is immutable and arena-free.** `push` wraps a NEW `Arc<Node>`.
   No `Mutex`, no `NodeId` arena, no central `Graph`. Diamonds dedup by
   `Arc::as_ptr`. Every walk must dedup (a `HashSet<NodeId>`) and MUST use
   the generic `GraphWalk<'a, B>` object from `src/node.rs`, which owns the
   transient traversal state (`demands` map, `lowered` set). Loose
   traversal functions are strictly forbidden. `Node<B>` provides delegated
   methods (`lower()`, `output_kind()`, `inputs()`, `demand_erased()`) so
   you NEVER `match` on `Node::Op` vs `Node::Source` during materialization.
   → §3.5, §4.

9. **The only engine-owned cache is the GPU pipeline cache** (keyed by
   IR-text hash XOR the `.slang` source fingerprint), with LRU. The
   fingerprint is mandatory: a kernel body never appears in the emitted
   `main()`, so without it an edited shader keeps a stale cached pipeline.
   No data/tile cache in the engine — that's a caller-side `Cached` source
   adapter (interactive=yes, batch=no). → §5.1, §5.5.

10. **Color/format conversion is an `Operation`, never an implicit fusion
    step.** All GPU codecs live in Slang; Rust only picks the wrapper
    string. → §5.3, §8.

### 11.2 FORBIDDEN — delete on sight, never write

- ❌ `trait Operation<Input> { fn execute(&self, input) }` (eager, old
  chromors). We use lazy `Operation<B>` + `Lower<B>`. There is no `execute`.
- ❌ A backend-generic `Image2D<B>` *handle struct* of its own. The handle is
  `Data<ImageKind, B>`; `Image2D<B>` is only a `type` alias of it. → §3.5.
- ❌ `view`/`params`/`Role`/`View`/`ParamBlock`/`TempSpec` on `AnyKind`,
  `Kind`, `Operation`, `WorkUnit`, or anything in the AGNOSTIC column of §2.
- ❌ The materializer (or any agnostic code) calling `GpuView`/`VipsBand`, or
  `downcast_ref::<ConcreteKind>()` to pick a view/codec. → invariant #2.
- ❌ A central `match`/enum over datatypes (`ValueKind`, `InputEncoder`,
  `OutputDecoder`, `WriteMode` as closed Rust enums that grow per datatype).
  Slang wrapper choice is a string from `GpuView::input`/`output`, not a Rust
  enum. → invariant #3.
- ❌ A persistent `Graph` struct, `NodeId` arena, or `Arc<Mutex<Graph>>`. →
  invariant #8.
- ❌ An engine-owned tile/region/value cache. Only the pipeline (shader)
  cache. → invariant #9.
- ❌ Making `Data::materialize` public, or downloading outside a `Target`. →
  invariant #1.
- ❌ A walk without pointer-dedup (causes duplicated lowering on diamonds). →
  invariant #8.
- ❌ A `wu` parameter on `Lower::lower`. → §3.3, §5.2.

### 11.3 Common pitfalls, by symptom

- Writing a new `Kind`? No `View`/`ParamBlock`/`Role` fields on it — those go
  in `impl GpuView for YourKind` (§5.3), a separate impl block.
- Writing the materializer / a generic walk? Never `downcast_ref` a `dyn
  AnyKind` to read a view/codec — the *node's* `lower` injects that at the
  concrete-type site (§5.2.1/§6.3).
- Writing `Operation::demand`? Pure `Region`/`Range`/`Atomic` arithmetic, no
  `cx`, no backend types.
- Writing `Lower::lower`? Signature is `fn lower(&self, cx: &mut B::Builder)`
  — no `wu` parameter (use `cx.wu()`).
- Two ops in the same fused pass both want a field called `"sigma"`/`"gain"`/
  etc.? Use `cx.param` (step-namespaced), not `cx.param_block` (§5.2.4).
- Tempted to add a `Graph`/`NodeId` arena, a data cache, or a
  "backend not supported" runtime error? All three are FORBIDDEN — use
  `GraphWalk` (already exists), the `Cached` source adapter pattern (caller
  side), and a missing trait impl (compile-time), respectively.
