# Staging Foundation — Design Proposal (for review before coding)

> **Status:** PROPOSAL. No engine code written yet. This documents exactly what
> would change in the core so it can be evaluated before any edit.
>
> **Goal:** make **data-dependent / cross-domain ops** (histogram
> equalize/cumulative/normalize/match, FFT, and any future "reduce then consume"
> op) expressible as **lazy `Operation`s**, without weakening any `CLAUDE.md`
> invariant.

---

## 1. The problem (why this is needed)

Some libvips ops are **data-dependent**: the result of a *reduction* must be
fully computed before a later stage can read it.

- `hist_equal` = `hist_find` (reduce image → 256-bin histogram) → `hist_cum` →
  `hist_norm` → `maplut` (apply the resulting LUT to the image).
- The histogram must be **completely accumulated** (every pixel) before the CDF
  that consumes it is correct. That is a **dispatch barrier**.

The current GPU materializer emits **one fused compute pass per `pull`**
(`emit_slang` → one shader, one dispatch domain). A reduction and its consumer
**cannot** share that pass:

1. They have different dispatch domains (image pixels vs 256 bins vs image
   pixels again).
2. A consumer reading the histogram in the same dispatch races the accumulation.

`pass.rs` already splits a pass into several **only** when the binding budget is
exceeded — it does not cut at reduction boundaries, and **no op today consumes a
reduction (`HistogramKind`) or a computed LUT mid-graph** (LUTs come from
constant sources). So these ops have no lazy expression and were left CPU-only.

---

## 2. The idea: a lazy staging boundary

Introduce an explicit **materialization barrier** the DAG can contain:

```
stage(D) : Data<K,B> -> Data<K,B>
```

`stage` returns a new pipeline tip whose **root is a Source**. When the
downstream pass lowers and reaches that source, the source **materializes its
upstream sub-DAG into a `Buffer<B>`** (its own pass/dispatch) and **injects that
buffer back as a normal decoded input** of the downstream pass.

That is exactly a **hard pass boundary**: upstream collapses to one buffer
(computed in full — barrier satisfied), downstream fuses fresh from it.

This is **the same mechanism already proven twice in the codebase**:

- `src/cache.rs` `CacheSource` — my cache boundary: lower-time
  `materialize(upstream)` → `cx.input(...)`. (Currently constrained to
  `Kind<WorkUnit = Region> + GpuView`.)
- `src/backend/gpu/pass.rs` `StagingSource` — the CutFinder's internal
  image-only "pre-dispatched subgraph injected as a leaf".

**Staging generalizes both to any `GpuView` Kind** (Region / Range / Atomic), so
a `HistogramKind` reduction can be staged and then consumed by a CDF→LUT op.

> Laziness is preserved: `stage()` does **not** compute at graph-build time. The
> materialize happens inside the source's `lower`, i.e. during the consuming
> `pull`, exactly like `CacheSource`/`VipsImageSource` already do.

---

## 3. Exact core surface touched

Three changes. **Nothing in the materializer, the demand/lower walk, `emit.rs`,
`compile.rs`, `node.rs`, or `work_unit.rs`.**

### 3.1 `GpuView` gains ONE defaulted method (the only trait change)

File: `src/backend/gpu/mod.rs` (the `GpuView` trait, `CLAUDE.md` §3 per-backend
column).

```rust
pub trait GpuView: Kind {
    fn input(&self) -> View;
    fn output(&self, wu: &WorkUnit) -> OutputWrap;

    /// NEW. Slot params for re-injecting a *materialized* value of this Kind as
    /// a decoded source input (what a staging/cache source pushes alongside
    /// `input()`). Default covers Region/Range Kinds (tight geometry); Atomic
    /// Kinds (histogram/vectorscope) override to supply their own (`bin_count`).
    fn source_params(&self, wu: &WorkUnit) -> ParamBlock {
        let (w, h) = match wu {
            WorkUnit::Region(r) => (r.w, r.h),
            WorkUnit::Range(rg) => (rg.end - rg.start, 1),
            WorkUnit::Atomic => panic!(
                "GpuView::source_params: an Atomic-shaped Kind must override source_params"
            ),
        };
        RegionParams::tight(w, h).into_block("region_in_{slot}")
    }
}
```

Why this is the right shape: it just **lifts the param block that each source's
`lower` already builds today** into the Kind (per `CLAUDE.md`: the Kind owns its
codec/decode vocabulary, not the ops/materializer). It is GPU vocabulary on a
per-backend trait — stays out of the AGNOSTIC half.

**Overrides needed (2):**
- `HistogramKind` (`src/data/histogram.rs`): `ParamBlock::scalar("bin_count", self.bins)`
- `VectorscopeKind` (`src/data/vectorscope.rs`): `ParamBlock::scalar("bin_count", self.grid * self.grid)`

**No change** to `ImageKind`, `LutKind`, `Mask2dKind` (default fits). Existing
sources can later be refactored to call `spec.source_params(wu)` instead of
hand-rolling `RegionParams::tight(...)`, but that's optional cleanup, not
required by this change.

### 3.2 New file `src/stage.rs` (additive, no edits elsewhere except `lib.rs`)

```rust
/// A lazy materialization barrier: its `lower` materializes the upstream
/// sub-DAG into a buffer and injects it as a decoded source for the consuming
/// pass. Generic over any GpuView Kind (Region/Range/Atomic).
pub struct StageSource<K: Kind, B: Backend> {
    upstream: Data<K, B>,
}

impl<K: GpuView> Source<GpuBackend> for StageSource<K, GpuBackend> {
    type Kind = K;
    fn spec(&self) -> Arc<K> { self.upstream.spec.clone() }

    fn fetch(&self, _ctx, wu: &K::WorkUnit) -> Result<Buffer<GpuBackend>, Error> {
        self.upstream.materialize(wu.clone())   // pub(crate), same as CacheSource
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let typed = K::WorkUnit::typed(&wu).expect("stage: wu shape");
        match self.upstream.materialize(typed) {
            Ok(buf) => cx.input(
                self.spec().input(),
                self.spec().source_params(&wu),   // ← the new method
                buf.payload,
            ),
            Err(e) => cx.fail(e),
        }
    }
    fn dyn_hash(&self, s) { s.write_usize(NodeId::of(&self.upstream.root).0); }
}

impl<K: Kind, B: Backend> Data<K, B> {
    /// Insert a materialization barrier: the upstream is computed in full and
    /// re-injected as a source for whatever consumes the returned tip. Lazy —
    /// nothing runs until pulled.
    pub fn stage(&self) -> Self
    where StageSource<K, B>: Source<B, Kind = K> {
        Data::from_source(Arc::new(StageSource { upstream: self.clone() }), self.ctx.clone())
    }
}
```

This is `CacheSource` minus the store, plus `source_params` in place of the
hard-coded `RegionParams::tight`. (Open question §6: should `stage` and the
cache share one `BoundarySource` with an optional store? Leaning yes — unify to
avoid two near-identical sources.)

### 3.3 `lib.rs`: `pub mod stage;`

That's the entire core footprint.

---

## 4. How the histogram ops build on it (no further core changes)

All in `src/operation/stats.rs` (+ small kernels in `shaders/`), as ordinary
`Operation`s — **consumers** of staging, not core changes:

1. **`HistogramFind` (GPU)** — already have the reduction (`HistogramOp` →
   `histogram_kernel`). Wire `HistogramFind<GpuBackend>: Lower` to produce a
   `HistogramKind` (multi-band variant: one `bin_count`-strip per band).
2. **`HistogramKind → LutKind` CDF op** (`EqualizeLut`): a new op whose input is
   a **staged** histogram. Kernel: 256 threads, entry `i` = `Σ bins[0..=i] /
   total`, written as a 256-entry `LutKind`. Dispatch domain = 256 (its own
   pass, fed by the staged histogram).
3. **`equalize()`** = `img.histogram(256, ch).stage()` → `EqualizeLut` →
   `.stage()` → `maplut(img, lut)`. Two barriers, three passes, fully lazy.
4. **`HistogramCumulative` / `HistogramNormalize`** = the same CDF/scale kernels
   exposed as standalone `HistogramKind → HistogramKind` ops.

FFT later reuses the identical pattern (each radix pass is staged).

---

## 5. Invariant check (`CLAUDE.md` §4)

| Invariant | Effect |
|---|---|
| 1. Data backend-resident; only exit is `Target` | ✅ `stage` materializes to a `Buffer<B>` and re-injects; never downloads. `materialize` stays `pub(crate)`. |
| 2. Materializer type-blind, no `View`/downcast | ✅ Untouched. `source_params` is read **inside** the source's `lower` (concrete-type site), exactly where `input()`/`output()` already are. |
| 3. One closed shape-enum, matched once | ⚠️ The `source_params` **default** matches `Region/Range/Atomic`. This is a *shape* switch (like `WorkUnit::union`), not a datatype switch — adding a new datatype adds **zero** arms. Acceptable, but it is a second shape-match; §6 asks whether to push it onto `WorkUnitFor` instead. |
| 4. New datatype = one file, no central enum | ✅ Atomic Kinds override `source_params` in their own file. |
| 6. Backend support via capability trait | ✅ `source_params` lives on `GpuView`; a Kind without `GpuView` can't be staged on GPU — compile error, not runtime. |
| 8. Immutable arena-free DAG, pointer-dedup walk | ✅ `stage` = one more `Arc<Node::Source>`; no mutation. |
| 9. Only engine cache = pipeline cache | ✅ `stage` adds no cache (the separate `cache.rs` store is opt-in and already accepted). |
| 10. Color convert is an Operation | ✅ Unaffected. |

The one yellow flag is invariant 3 (a second shape-match in the default). §6
offers an alternative that removes it.

---

## 6. Open questions for you

1. **Unify `stage` + `cache`?** They're the same boundary; cache = staged +
   memoized. Propose one `BoundarySource<K>` with `Option<Arc<RegionCache>>`;
   `stage()` = no store, `cache()` = with store. Removes duplication. OK?
2. **Where does `source_params` live?** On `GpuView` (proposed, default does a
   shape-match) **or** make the *geometry* part a method on `WorkUnitFor`
   (`Region/Range/Atomic` each return their own block) so there's no shape-match
   in `GpuView` at all (invariant 3 stays pristine), with `GpuView` only
   overriding when it needs extra fields (`bin_count`). Slightly more plumbing,
   cleaner against §4.3. Preference?
3. **Multi-band histogram representation.** `HistogramKind { bins, bands }`
   already supports bands; the multi-band reduction kernel needs per-band
   `InterlockedAdd` into a `bins*bands` buffer. Confirm we want per-band
   equalize (vips default) vs luma-only first.
4. **Eager vs lazy `stage` cost.** Each `stage` is a real extra dispatch +
   intermediate buffer. For the viewport's per-tile pulls that's fine (256-bin
   histogram is tiny), but `equalize` over a full image computes the histogram
   per pull. Do we want `stage` results memoized via the cache by default for
   data-dependent ops? (Ties to Q1.)

---

## 7. What I will NOT touch

`node.rs`, `work_unit.rs` (unless Q6.2 chosen), `emit.rs`, `compile.rs`,
`pass.rs`, the materializer, any AGNOSTIC type. Scope is: one defaulted
`GpuView` method + 2 overrides + one new `stage.rs` + `lib.rs` line. Everything
else (the histogram ops, kernels, FFT) is additive `Operation`/Slang code that
*consumes* this foundation.
```
