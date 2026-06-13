# Chromors POC — Processing Model v2 — RULES (read before touching anything)

This crate is a **clean-room prototype of one processing model**. It has ONE
architecture. There is no "old way". If you find code that looks like the list
in **§7 FORBIDDEN**, it is a bug to delete, not a pattern to copy.

> **`docs/architecture.md` is the full reference**: every type, every trait,
> the DAG/demand/lower walk, GPU kernel fusion + emit/compile/dispatch, the
> vips backend, and step-by-step recipes for adding a datatype/operation/
> backend, with worked examples. **Read it first if you're new.**
>
> This file (`CLAUDE.md`) is the **compact binding contract**: the
> non-negotiable rules, distilled. It does not re-explain *why* or *how* —
> that's `architecture.md`. Where the two disagree on a RULE (not an
> explanation), this file wins.

---

## 1. What the model is (one sentence)

A **lazy, multi-backend DAG**: operations build an immutable `Arc<Node<B>>`
graph; nothing runs until a `Target` pulls a `WorkUnit`; then the chosen
`Backend` (`GpuBackend` = Slang JIT fusion, `VipsBackend` = libvips) walks the
graph **type-blind** and lowers each node into its builder.

→ `docs/architecture.md` §1 for the full mental model and §10 for an
end-to-end worked trace.

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

→ `docs/architecture.md` §2 for the file-by-file mapping of this table.

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
`Operation<B>` / `Source<B>` (the typed traits are not object-safe), bridged
by a blanket impl. `Node<B>` stores `Arc<dyn AnyOperation<B>>` /
`Arc<dyn AnySource<B>>`. **Do not** make the typed traits object-safe; do not
bypass the bridge.

→ `docs/architecture.md` §3 for what each trait/type does and how to use it
(one subsection per item above).

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
   Every walk must dedup and MUST use the generic `GraphWalk<'a, B>` object
   from `src/node.rs` (owns `demands` map + `lowered` set). Loose traversal
   functions are strictly forbidden. `Node<B>` provides delegated methods
   (`lower()`, `output_kind()`, `inputs()`, `demand_erased()`) so you NEVER
   `match` on `Node::Op` vs `Node::Source` during materialization.

9. **The only engine-owned cache is the GPU pipeline cache (keyed by IR-text
   hash XOR the `.slang` source fingerprint), with LRU.** The fingerprint is
   mandatory: a kernel body never appears in the emitted `main()`, so without it
   an edited shader keeps a stale cached pipeline. No data/tile cache in the
   engine — that's a caller-side `Cached` source adapter (interactive=yes,
   batch=no).

10. **Color/format conversion is an `Operation`, never an implicit fusion
    step.** All GPU codecs live in Slang; Rust only picks the wrapper string.

→ `docs/architecture.md` §3.5/§4 (#1,#2,#8), §3.2 (#3,#7), §7/§8 (#4,#5,#6),
§5.1 (#9), §8/§5.3 (#10) for the mechanics behind each invariant, and §11 for
a one-page cheat sheet linking invariants to FORBIDDEN patterns.

---

## 5. Adding a datatype / operation / backend — quick recipe

Full recipes with worked examples (`ImageKind`/`HistogramKind`,
`Invert`/`ExtractBand`/`Blur`/`Reinterpret`) are in `docs/architecture.md`
§7–§9. The compact version:

- **Datatype** (`docs/architecture.md` §7): one new `src/data/<name>.rs` —
  `<Name>Kind` (`impl AnyKind` + `impl Kind { type WorkUnit = ... }`),
  `impl GpuView`/`impl VipsBand` for whichever backends it supports,
  `pub type <Name><B> = Data<<Name>Kind, B>`, producing operations, ergonomic
  inherent methods, Slang kernel(s) if GPU.

- **Operation** (`docs/architecture.md` §8):
  ```rust
  pub struct Foo<B: Backend> { input: Input<InKind, B>, /* params */ }

  impl<B: Backend> Operation<B> for Foo<B> where Foo<B>: Lower<B> {
      type Output = OutKind;
      fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
      fn demand(&self, out: &<OutKind as Kind>::WorkUnit) -> Vec<Option<WorkUnit>> { /* halo / prune */ }
      fn output_spec(&self) -> OutKind { /* derive from self.input.spec */ }
      fn dyn_hash(&self, s: &mut dyn Hasher) { /* hash own params */ }
  }
  impl Lower<GpuBackend> for Foo<GpuBackend> {
      fn lower(&self, cx: &mut GpuBuilder) {
          cx.kernel("ops.foo", "foo_kernel").param("k", self.k);
          cx.output(self.output_spec().output(cx.wu()));
      }
  }
  impl Lower<VipsBackend> for Foo<VipsBackend> { fn lower(&self, cx: &mut VipsBuilder) { /* build vips op, cx.emit */ } }
  ```
  `lower` takes **only** `&mut B::Builder` (no `wu` param — use `cx.wu()`).
  Fusion is automatic: consecutive ops' kernel steps chain in one shader via
  post-order lowering — no extra Rust needed. Two steps in the same fused pass
  must not both `cx.param_block(...)` a same-named field (use `cx.param`,
  which is step-namespaced) — see `docs/architecture.md` §5.2.4.

- **Backend** (`docs/architecture.md` §9): new `impl Backend` + `Builder` +
  capability trait; every existing op gains it via one
  `impl Lower<NewBackend>` each (the generic `Operation<B>` impl is untouched).

---

## 6. FORBIDDEN — delete on sight, never write

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

## 7. Slang code

Slang shaders live in `shaders/` (compiled by `backend::gpu::slang` via FFI to
SPIR-V, cached by IR hash). Rules:

- The **working-space sandwich** is mandatory: every kernel decodes inputs to
  the working representation, processes, encodes the output — via the generic
  Slang `Codec<Format, ColorSpace>` / `IRegion` library (`shaders/lib/region.slang`),
  parameterised by the strings `GpuView::view` returns. Color conversion is
  shader-side only.
- A new datatype's Slang wrapper (e.g. `HistogramOut<N>`, `PointListView<N>`)
  is real new GPU code — but it is the ONLY thing added; no Rust enum/match.
- Entry-point names are what `lower` passes to `GpuBuilder::kernel(...)`.
- Do not put image-processing logic in the materializer or the viewport; the
  shader does the math, Rust only orchestrates buffers + params.

→ `docs/architecture.md` §5.3/§5.4 for the `View`/`ParamBlock`/`ViewAdapter`
vocabulary and how it becomes emitted Slang text.

---

## 8. File map

```
src/
  kind.rs            AnyKind, Kind                          (agnostic)
  work_unit.rs       Shape, WorkUnit, Region/Range/Atomic, Lod, union/bounding/tile_aligned
  operation/mod.rs   Operation<B>, Lower<B>, Input, AnyInput, AnyOperation (erased bridge)
  io.rs              Source<B>, Target<K,B>, AnySource (erased bridge)
  node.rs            Node<B>, Data<K,B>, GraphWalk           (immutable DAG handle + walk)
  buffer.rs          Buffer<B>                               (backend-resident)
  backend/mod.rs     Backend trait
  backend/gpu/       GpuBackend, GpuBuilder, GpuView, GpuContext (pipeline cache), GpuBuffer,
                     view.rs (View/ParamBlock/ViewAdapter/OutputWrap — GPU vocabulary),
                     emit.rs (builder -> Slang text) / compile.rs (cache+dispatch) / slang.rs (JIT)
  backend/vips/      VipsBackend, VipsBuilder (node-keyed handle map), VipsBand,
                     mod.rs; gobject/source/target/working = FFI + CPU custom-region plumbing
  data/              concrete datatypes: image.rs, histogram.rs, vectorscope.rs, ...
  color/ pixel/      color science + pixel formats (agnostic metadata used by Kinds)
docs/
  architecture.md    full reference: types/traits/algorithms/recipes (read this first)
tests/
  smoke.rs           GPU: ImageKind+Blur+ImageSource type-check end to end
  vips_smoke.rs      Vips: SAME generic traits, different Lower
```

---

## 9. Verification (run before claiming done)

```
cargo build --lib        # MUST be 0 errors
cargo test --test smoke --test vips_smoke   # MUST pass (compile-proofs of the model)
```

If `cargo test` (full) shows errors only in `color/*` / `pixel/*` / generated
`ffi.rs`, those are unrelated engine-port leftovers — note them, do not let them
mask a real regression in the model code above.
