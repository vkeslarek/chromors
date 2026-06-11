# Chromors POC — Processing Model v2 — RULES (read before touching anything)

This crate is a **clean-room prototype of one processing model**. It has ONE
architecture. There is no "old way". If you find code that looks like the list
in **§7 FORBIDDEN**, it is a bug to delete, not a pattern to copy.

> The full design rationale is `../docs/processing-model-v2.md`. This file is
> the **binding contract**. Where prose and this file disagree, this file wins.

---

## 1. What the model is (one sentence)

A **lazy, multi-backend DAG**: operations build an immutable `Arc<Node<B>>`
graph; nothing runs until a `Target` pulls a `WorkUnit`; then the chosen
`Backend` (`GpuBackend` = Slang JIT fusion, `VipsBackend` = libvips) walks the
graph **type-blind** and lowers each node into its builder.

---

## 2. The two halves — THE central rule

Every piece of code is in exactly one of two halves. **Never mix them.**

| AGNOSTIC (knows no backend, no Slang, no libvips) | PER-BACKEND (one impl per backend it supports) |
|---|---|
| `AnyKind`, `Kind` — metadata + shape + byte_size + dyn_hash | `GpuView` / `VipsBand` — Kind's lowering capability |
| `Operation<B>` — `inputs` / `demand` / `output_spec` / `dyn_hash` | `Lower<B>` — the execution (`lower`) |
| `WorkUnit` / `Shape` / `Region` / `Range` / `Atomic` | `GpuBuilder` / `VipsBuilder`, `View`, `ParamBlock`, `Role` |
| `Source` / `Target` traits, `Node<B>`, `Data<K,B>`, `Buffer<B>` | `GpuBackend` / `VipsBackend`, `GpuContext`, `GpuBuffer`, `VipsHandle` |

**Litmus test:** does the type/method mention Slang, `View`, `ParamBlock`,
wgpu, or libvips? Then it is **per-backend** and must NOT appear on `AnyKind`,
`Kind`, `Operation<B>`, `WorkUnit`, `Source`, `Target`, or the materializer.

---

## 3. The traits — exact roles (do not add/remove methods without updating this file)

```
AnyKind        : object-safe metadata. as_any / shape / byte_size / dyn_hash. NOTHING GPU.
Kind: AnyKind  : + associated type WorkUnit. NOTHING GPU.
GpuView: Kind  : input()->View (decode wrapper), output()->OutputWrap (how its
                 result lands: codec sandwich vs direct write), params(&WorkUnit)
                 ->ParamBlock (extra wrapper params, e.g. bin_count; default none).
                 The Kind owns its codec — ops never decode/encode. (backend::gpu)
VipsBand: Kind : band_format()->i32.                              (lives in backend::vips)

Operation<B>: Lower<B>   : inputs() -> Vec<&dyn AnyInput<B>>
                           demand(out) -> Vec<Option<WorkUnit>>   (None = prune that input this region)
                           output_spec() -> Self::Output
                           dyn_hash(state)                          (op's own config, backend-independent)
Lower<B>                 : lower(&self, &mut B::Builder)            (THE only per-backend op code)

Source<B>  : type Kind; spec(); fetch(ctx, wu) -> Buffer<B>; lower(&mut B::Builder); dyn_hash()
Target<K,B>: type Out; extract(&Buffer<B>, wu, ctx) -> Out         (the ONLY exit from a backend)

Backend    : type Ctx; type Payload; type Builder; materialize(ctx, root, wu) -> Buffer<B>
Data<K,B>  : the user handle. root: Arc<Node<B>>, ctx, spec: Arc<K>. push/as_input/materialize(internal)/extract.
Buffer<B>  : payload: Arc<B::Payload> + spec: Arc<dyn AnyKind>. Backend-resident. download/extract only via Target.
```

`AnyOperation<B>` / `AnySource<B>` are the **object-safe erased mirrors** of
`Operation<B>` / `Source<B>` (the typed traits are not object-safe). They carry
`inputs` / `demand_erased` / `output_kind` / `lower` / `dyn_hash`. A **blanket
impl bridges** every typed op/source to its erased form. `Node<B>` stores
`Arc<dyn AnyOperation<B>>` / `Arc<dyn AnySource<B>>`. **Do not** make the typed
traits object-safe; do not bypass the bridge.

---

## 4. THE INVARIANTS (non-negotiable)

1. **Data is always backend-resident.** No `residency` field, no host `Buffer`
   variant. The ONLY way to host is `Target::extract`. `Data::materialize` is
   `pub(crate)` — never make it public, never `.download()` outside a `Target`.

2. **The materializer is type-blind.** It walks `Arc<dyn AnyKind>` and **must
   never** read a `View`/`ParamBlock` from a Kind, nor downcast to a concrete
   Kind. Views/params are **injected by the node inside `lower`** (concrete-type
   site). You cannot cross-cast `dyn AnyKind` → `dyn GpuView` — do not try.

3. **One closed enum only: `Shape` (Region/Range/Atomic).** It is per-*shape*
   (3 materialize strategies), NOT per-datatype. The single allowed `match` over
   shapes lives in `work_unit.rs` (`WorkUnit::union`). Adding a datatype must
   add ZERO match arms anywhere. Adding a Slang wrapper type must add ZERO Rust
   enum variants.

4. **Adding a datatype is additive: one new file in `src/data/`.** No central
   enum, no `emit.rs` match, no edit to `AnyKind`/`Operation`/`Backend`.

5. **Adding a backend is additive.** A new `impl Backend` + its `Builder` + the
   per-backend capability trait (`GpuView`-equivalent). Existing ops gain it by
   writing one `Lower<NewBackend>` each; the structural `Operation<B>` impl is
   untouched.

6. **A Kind only supports a backend if it implements that backend's
   capability.** `HistogramKind` has `GpuView` and not `VipsBand` ⇒
   `Data<HistogramKind, VipsBackend>` does not compile. This is the intended
   mechanism — never add a runtime "unsupported backend" error.

7. **Region/dimensional math lives on the typed `WorkUnit`s** (`Region::bounding`,
   `Region::tile_aligned`), exposed to the generic engine via `WorkUnit::union`
   (the 3-arm shape switch). The materializer calls `wu.union(other)`; it does
   not do rect math itself.

8. **The DAG is immutable and arena-free.** `push` wraps a NEW `Arc<Node>`. No
   `Mutex`, no `NodeId`, no central `Graph`. Diamonds dedup by `Arc::as_ptr`.
   Every walk must dedup (a `HashSet<usize>` of node pointers) and MUST use
   the generic `GraphWalk<'a, B>` object from `src/node.rs`, which owns the transient
   traversal state (`demands` map, `lowered` set). Loose traversal functions
   are strictly forbidden. Furthermore, `Node<B>` provides delegated methods
   (`lower()`, `output_kind()`, `inputs()`, `demand_erased()`) so that you NEVER
   need to `match` on `Node::Op` vs `Node::Source` during materialization.

9. **The only engine-owned cache is the GPU pipeline cache (keyed by IR-text
   hash XOR the `.slang` source fingerprint), with LRU.** The fingerprint is
   mandatory: a kernel body never appears in the emitted `main()`, so without it
   an edited shader keeps a stale cached pipeline. No data/tile cache in the
   engine — that's a caller-side `Cached` source adapter (interactive=yes,
   batch=no).

10. **Color/format conversion is an `Operation`, never an implicit fusion
    step.** All GPU codecs live in Slang; Rust only picks the wrapper string.

---

## 5. How to add a datatype (the recipe)

Create `src/data/<name>.rs`, add it to `src/data/mod.rs`. Inside:

1. `struct <Name>Kind { …metadata… }` + `impl AnyKind` (shape/byte_size/dyn_hash) + `impl Kind` (WorkUnit).
2. `impl GpuView for <Name>Kind` and/or `impl VipsBand for <Name>Kind` — only the backends it supports.
3. `pub type <Name><B = …> = Data<<Name>Kind, B>;`
4. Operations producing it: `struct <Op> { input: Input<…, B>, … }` + `impl Operation<B>` (structure) + one `impl Lower<EachBackend>`.
5. Ergonomic methods: `impl <Name><B> { pub fn … }` (in-crate ⇒ plain inherent impls are fine).
6. The Slang kernel + wrapper (GPU) — see §8.

See `src/data/image.rs` (multi-backend), `src/data/histogram.rs` /
`src/data/vectorscope.rs` (GPU-only, `Atomic`-shaped) as the canonical examples.

---

## 6. How to add an operation

```rust
pub struct Foo<B: Backend> { input: Input<InKind, B>, /* params */ }

impl<B: Backend> Operation<B> for Foo<B> where Foo<B>: Lower<B> {
    type Output = OutKind;                                  // may differ from input Kind
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &<OutKind as Kind>::WorkUnit) -> Vec<Option<WorkUnit>> { /* halo / prune */ }
    fn output_spec(&self) -> OutKind { /* derive from self.input.spec */ }
    fn dyn_hash(&self, s: &mut dyn Hasher) { /* hash own params */ }
}

impl Lower<GpuBackend> for Foo<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        // The op contributes ONLY its kernel step(s) + scalar params. Inputs
        // are registered by the Source leaf; decode/encode come from the Kind.
        cx.kernel("foo_main").param("k", self.k);
        cx.output(self.output_spec().output());   // Kind decides sandwich vs direct
    }
}
impl Lower<VipsBackend> for Foo<VipsBackend> { fn lower(&self, cx: &mut VipsBuilder) { /* build vips op, cx.emit */ } }
```

`Lower::lower` takes **only** `&mut B::Builder`. The node's resolved `WorkUnit`
is carried by the builder (`GpuBuilder::wu()` / set via `enter`) — **do NOT add
a `wu` parameter to `lower`** (it must stay backend-neutral; vips ignores wu).

**Multistep / fusion.** Each `cx.kernel(...)` adds a step to the fused pass.
Step 0 reads the decoded sources; step `i>0` reads step `i-1`'s working buffer
(the emitter ping-pongs two `float4` scratch buffers). So a separable op assembles
its passes — e.g. blur is `run_h` then `run_v`, a method that just calls
`cx.kernel("blur_h_kernel")` then `cx.kernel("blur_v_kernel")` — and fusion
across ops (`invert` then `blur`) falls out of the post-order lower for free.
The op never allocates buffers or names bindings; it only adds kernels + the
Kind's `output()`.

---

## 7. FORBIDDEN — delete on sight, never write

- ❌ `trait Operation<Input> { fn execute(&self, input) }` (eager, old chromors).
  We use lazy `Operation<B>` + `Lower<B>`. There is no `execute`.
- ❌ A backend-generic `Image2D<B>` *handle struct* of its own. The handle is
  `Data<ImageKind, B>`; `Image2D<B>` is only a `type` alias of it.
- ❌ `view`/`params`/`Role`/`View`/`ParamBlock`/`TempSpec` on `AnyKind`,
  `Kind`, `Operation`, `WorkUnit`, or anything in §2-AGNOSTIC.
- ❌ The materializer (or any agnostic code) calling `GpuView`/`VipsBand`, or
  `downcast_ref::<ConcreteKind>()` to pick a view/codec.
- ❌ A central `match` / enum over datatypes (`ValueKind`, `InputEncoder`,
  `OutputDecoder`, `WriteMode` as closed Rust enums that grow per datatype).
  Slang wrapper choice is a string from `GpuView::view`, not a Rust enum.
- ❌ A persistent `Graph` struct, `NodeId` arena, or `Arc<Mutex<Graph>>`.
- ❌ An engine-owned tile/region/value cache. Only the pipeline (shader) cache.
- ❌ Making `Data::materialize` public, or downloading outside a `Target`.
- ❌ A walk without pointer-dedup (causes duplicated lowering on diamonds).
- ❌ A `wu` parameter on `Lower::lower`.

---

## 8. Slang code (incoming)

Slang shaders live in `shaders/` (compiled by `backend::gpu::slang` via FFI to
SPIR-V, cached by IR hash). Rules:

- The **working-space sandwich** is mandatory: every kernel decodes inputs to
  the working representation, processes, encodes the output — via the generic
  Slang `Codec<Format, ColorSpace>` / `WorkingView` library, parameterised by
  the strings `GpuView::view` returns. Color conversion is shader-side only.
- A new datatype's Slang wrapper (e.g. `HistogramOut<N>`, `PointListView<N>`)
  is real new GPU code — but it is the ONLY thing added; no Rust enum/match.
- Entry-point names are what `lower` passes to `GpuBuilder::kernel(...)`.
- Do not put image-processing logic in the materializer or the viewport; the
  shader does the math, Rust only orchestrates buffers + params.

---

## 9. File map

```
src/
  kind.rs            AnyKind, Kind                          (agnostic)
  work_unit.rs       Shape, WorkUnit, Region/Range/Atomic, Lod, union/bounding/tile_aligned
  operation.rs       Operation<B>, Lower<B>, Input, AnyInput, AnyOperation (erased bridge)
  io.rs              Source<B>, Target<K,B>, AnySource (erased bridge)
  node.rs            Node<B>, Data<K,B>                      (immutable DAG handle)
  buffer.rs          Buffer<B>                               (backend-resident)
  backend/mod.rs     Backend trait
  backend/gpu/       GpuBackend, GpuBuilder, GpuView, GpuContext (pipeline cache), GpuBuffer,
                     view.rs (View/ParamBlock/Role/Binding/TempSpec — GPU vocabulary),
                     materialize.rs (demand_walk + lower_walk, type-blind),
                     emit.rs / compile.rs / slang.rs (Slang JIT → SPIR-V → pipeline)
  backend/vips/      VipsBackend, VipsBuilder (node-keyed handle map), VipsBand,
                     mod.rs (lower_walk, deduped); gobject/source/target/… = FFI plumbing
  data/              concrete datatypes: image.rs, histogram.rs, vectorscope.rs
  color/ pixel/      color science + pixel formats (agnostic metadata used by Kinds)
tests/
  smoke.rs           GPU: ImageKind+Blur+ImageSource type-check end to end
  vips_smoke.rs      Vips: SAME generic traits, different Lower
```

---

## 10. Verification (run before claiming done)

```
cargo build --lib        # MUST be 0 errors
cargo test --test smoke --test vips_smoke   # MUST pass (compile-proofs of the model)
```

If `cargo test` (full) shows errors only in `color/*` / `pixel/*` / generated
`ffi.rs`, those are unrelated engine-port leftovers — note them, do not let them
mask a real regression in the model code above.
