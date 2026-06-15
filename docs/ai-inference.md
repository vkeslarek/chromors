# AI Inference — Burn Backend (generic) + Smart Mask (YOLO Segmentation)

> **Status: PROPOSAL.** Like `docs/staging-foundation.md`, this is written for
> discussion *before* any code lands. The headline result: **this needs ZERO
> changes to the engine core** (`kind.rs`, `operation/mod.rs`, `node.rs`,
> `io.rs`, `backend/*`, `work_unit.rs`, `pass.rs`, `emit.rs`, `compile.rs`). It
> is 100% additive — new `Source` impls plus a new per-backend module, built
> entirely on primitives that already exist (`Data::stage`, `GpuContext::{device,queue}`,
> `GpuBuffer::buffer()`, `read_to_cpu`).

---

## 0. What this is — two separate things

1. **The Burn backend** (`src/ai/` core: `device.rs`, model-loading
   conventions, the "inference `Source`" pattern, §§1-10 below) — generic
   plumbing for running **any** [Burn](https://burn.dev) model inside
   chromors, on either chromors `Backend` (GPU via Burn `Wgpu`, sharing
   chromors' `wgpu::Device`; CPU/Vips via Burn `NdArray`), with zero engine-core
   changes.

2. **Smart Mask** (YOLO-seg segmentation → `Mask2D<B>`, §§11-20) — the
   **first application** built on (1). Lives in its own files
   (`src/ai/smart_mask.rs`, `src/ai/yolo_seg.rs`, `src/ai/decode.rs`). A future
   model (depth estimation, matting, classification, super-resolution, ...) is
   a **sibling** module next to `smart_mask.rs` — written by following the
   Part 1 §9 recipe — not a modification to it.

Read Part 1 to understand what "the Burn backend" actually is and what's
reusable. Read Part 2 for the concrete worked example, and to implement smart
masks specifically.

---

# PART 1 — The Burn backend (generic)

## 1. Goals & non-goals

### 1.1 Goals (v1)

- Run an arbitrary Burn model — `Model<BurnB>`, either codegen'd from ONNX
  (`burn-import`) or hand-written — inside chromors, on **both** chromors
  backends:
  - `GpuBackend` → Burn's `Wgpu` backend, **sharing** chromors'
    `wgpu::Device`/`Queue` (`GpuContext`) — best-effort (§4).
  - `VipsBackend` → Burn's `NdArray` (CPU) backend.
- Express every model invocation as an ordinary lazy `Source<B>` producing
  *some* existing `Kind` (`Mask2D`, `Image2D`, `Histogram`, whatever the model
  naturally produces) — composes with `.cache()`, fuses downstream like any
  other leaf, same contract as `BoundarySource`/`GpuConstantLutSource`.
- Preprocessing (resize/convert to the model's expected input tensor) is
  **ordinary chromors ops + `.stage()`** — never bespoke pixel-shuffling code
  outside the `Source`.
- Model loading/codegen happens **once per process** (build-time codegen +
  runtime `OnceLock` cache), with a documented hand-written fallback (RULE
  AI2) when ONNX import doesn't cover a model's ops.
- Everything lives behind `feature = "ai"`. `cargo build --lib` (no features)
  is unaffected — the regression check for "zero core impact."

### 1.2 Non-goals (v1)

- **Not** a new `Backend` (no `AiBackend: Backend` impl). Burn lives entirely
  inside `Source` impls' `fetch`/`lower` — leaves, exactly like
  `GpuConstantLutSource` or `BoundarySource`.
- **Not** zero-copy GPU↔Burn tensor transfer in v1. One GPU→CPU→GPU roundtrip
  via `read_to_cpu` for the (small, preprocessed) model-input tensor. Each
  model's `Source` can independently upgrade to zero-copy later (§4.2/§10
  AI1) — it's an internal `fetch` detail, not part of the shared pattern's
  interface.
- **Not** training/fine-tuning — fixed pre-trained checkpoints only.
- **Output decode is always model-specific.** Part 1 deliberately does **not**
  try to abstract "interpret the model's output tensors into a `Kind`" — that
  logic is irreducibly per-model (§6, §9 of the recipe). What Part 1
  generalizes is everything *around* that: device, loading, caching, the
  `Source` shape, staging.

---

## 2. Where this sits (two-halves placement)

Per `CLAUDE.md` §2, classify every new *core-of-Part-1* piece:

| Piece | Half | Why |
|---|---|---|
| `src/ai/device.rs` (`GpuBurn`/`CpuBurn` aliases, `gpu_burn_device`/`cpu_burn_device`) | **PER-BACKEND** (new) | Burn device types, picked per chromors `Backend`. Lives in `src/ai/`, never touches `AnyKind`/`Operation<B>`/etc. |
| Model-loading/cache pattern (`OnceLock<Model<BurnB>>` per model type, §5) | **PER-BACKEND** (pattern, not a type) | Each model module (e.g. `smart_mask.rs`) declares its own statics following this pattern. |
| "Inference `Source`" shape (§6) | **PER-BACKEND** (pattern) | Each model gets its own `XyzSource<B>: Source<GpuBackend> + Source<VipsBackend>` following this shape. |
| Output `Kind` (`Mask2DKind`, `ImageKind`, `HistogramKind`, ...) | **AGNOSTIC** (existing, unchanged) | A model's output is *just another value of an existing Kind* — Part 1 adds no new Kinds. |
| Per-model params struct (e.g. `SmartMaskParams`) | **AGNOSTIC** (new per model, tiny config struct) | No backend types — lives next to the model's module, re-exported from `src/ai/mod.rs`. |

**Litmus test (CLAUDE.md §2):** does anything mention `burn`, `Wgpu`,
`wgpu::Buffer`, or ONNX on an existing `*Kind`, `AnyKind`, `Operation<B>`,
`WorkUnit`, or the materializer? **No.** Burn types appear *only* inside
`src/ai/*` per-backend `Source` impls — exactly where `wgpu::Buffer`/
`VipsHandle` already appear for every other `Source`.

---

## 3. The central idea: inference is a `Source`, built on `.stage()`

A `Source<B>` is — per `docs/staging-foundation.md` — already "a hard
materialization point: the engine fetches a real `Buffer<B>` there before the
downstream pass fuses from it." Running a Burn model is the **same kind of
cut**, just with a different executor than `B::Builder`'s Slang dispatch.

Generic shape (no model specifics — this is the pattern every `XyzSource<B>`
in Part 2+ follows):

```
input: Data<SomeKind, B>
   │
   │  (ordinary chromors ops — fused pass #1, or vips pipeline)
   ▼
preprocessed = <ops that produce the model's expected input shape/format>
   │
   │  .stage()  ← BoundarySource, NO STORE (docs/staging-foundation.md)
   ▼
staged: Data<SomeKind, B>   (root = BoundarySource; pulling it = pass #2,
                              materializes a tight model-input-shaped buffer)
   │
   │  XyzSource::fetch/lower:
   │    1. staged.materialize(full_region) → Buffer<B>   (pass #2 runs HERE)
   │    2. extract bytes → Burn tensor (Path A: read_to_cpu)
   │    3. model.forward(tensor) → raw output tensor(s)   [Burn's own
   │                                                         dispatch graph —
   │                                                         same wgpu device
   │                                                         for GPU]
   │    4. decode_xyz(raw outputs, params) → CPU buffer in the OUTPUT Kind's
   │                                          layout                (MODEL-SPECIFIC)
   │    5. upload → new GpuBuffer/VipsHandle
   ▼
output: Data<OutputKind, B>   (whatever Kind the model naturally produces)
```

**Three dispatches total** (chromors pass #1 if preprocessing fuses, chromors
pass #2 = the staging materialize, Burn's own dispatch graph for inference) —
and for `GpuBackend`, **all three run on the same `wgpu::Device`** (§4), so
even with the v1 CPU roundtrip, data only leaves the GPU for the small
preprocessed model-input tensor, not the full-resolution input.

This is the mechanism the user specifically wanted ("dois dispatches sem o
dado sair da GPU") extended to a third, heterogeneous executor — and it falls
out of `.stage()` for free. **Steps 1, 2, 3 (decode aside), 5 are generic** —
the only model-specific code is step 4 and the preprocessing that produces
`staged`.

---

## 4. Burn backend selection & device sharing

### 4.1 The two instantiations

```rust
// src/ai/device.rs  (new file, feature = "ai")

/// GPU: Burn's wgpu backend, generic precision.
pub type GpuBurn = burn::backend::Wgpu<f32, i32>;

/// CPU: Burn's ndarray backend.
pub type CpuBurn = burn::backend::NdArray<f32>;
```

Every model wrapper (`YoloSegModel<BurnB>` in Part 2, and any future
`XyzModel<BurnB>`) is generic over `BurnB: burn::tensor::backend::Backend`;
chromors instantiates it as `XyzModel<GpuBurn>` inside `Source<GpuBackend>` and
`XyzModel<CpuBurn>` inside `Source<VipsBackend>`. **This instantiation is
identical for every model** — only the model type itself changes.

### 4.2 Sharing the wgpu device (GPU path)

`GpuContext` (`src/backend/gpu/context.rs`) already exposes:

```rust
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    ...
}
```

`burn-wgpu` supports constructing its runtime from an **existing**
`wgpu::Device`/`Queue` (a `WgpuSetup`/`WgpuDevice` "existing" variant — see
`burn_wgpu::WgpuDevice`/`init_device` in the version pinned in `Cargo.toml`;
the exact constructor name has moved between Burn releases, so **check the
docs for the pinned version at implementation time** — this is the one place
in this doc where an exact signature isn't nailed down, because it's a
fast-moving external API).

**RULE AI1 — device sharing is best-effort, not required for correctness.**
If, at implementation time, the pinned `burn-wgpu` version's
existing-device API is awkward/unavailable, **fall back to a Burn-owned wgpu
device** (`burn::backend::wgpu::WgpuDevice::default()`, its own
adapter/device/queue). Correctness (model output) is unaffected — only the
"stays on one device" property is lost, and v1's CPU roundtrip already crosses
the device boundary anyway via host memory. Wire it as a single function,
`src/ai/device.rs::gpu_burn_device(ctx: &GpuContext) -> <GpuBurn as
Backend>::Device`, shared by **every** model — upgradeable later without
touching call sites.

### 4.3 CPU device

```rust
pub fn cpu_burn_device() -> <CpuBurn as burn::tensor::backend::Backend>::Device {
    Default::default()
}
```

Trivial — `NdArray`'s device is a unit type. Shared by every model.

---

## 5. Model loading & caching conventions

### 5.1 Codegen (build-time)

Each model gets a **feature-gated** build step using `burn-import`/`burn-onnx`'s
`ModelGen` (generates backend-generic Rust code + a `Model<B>::new(device) ->
Self` + `Model<B>::forward(...)`):

```rust
// build.rs, appended, only when feature = "ai":
#[cfg(feature = "ai")]
{
    burn_import::onnx::ModelGen::new()
        .input("models/<model-file>.onnx")
        .out_dir("ai/") // -> OUT_DIR/ai/<model_file>.rs
        .run_from_script();
}
```

```rust
// src/ai/model_gen.rs — one `include!` per model
#[cfg(feature = "ai")]
pub mod <model_module> {
    include!(concat!(env!("OUT_DIR"), "/ai/<model_file>.rs"));
}
```

This generates `<model_module>::Model<B: Backend>` with weights baked in as
constant tensors. **RULE AI2:** if the pinned `burn-import` version cannot
import a given ONNX graph (operator coverage gaps are common for
detection/segmentation heads), the fallback is a **hand-written `Model<B>`
struct** (a handful of `Conv2d`/`Linear`/etc. Burn modules) with weights loaded
at runtime via `burn::record` (`.mpk`/safetensors) — **same downstream
interface** (`Model::new(device)` + `Model::forward(input) -> <output
tensors>`), so the `Source` pattern (§6) is unaffected by which path is used.
Do not let ONNX import issues block the `Source`/staging plumbing for a new
model — stub `Model::forward` with a fixed-size zero tensor first, wire
everything else, swap the real model in last.

**RULE AI3:** for every model, write a tiny `tests/ai/<model>_shapes.rs` that
runs `forward` on a zero tensor and asserts the output tensor shapes — if the
real export's shapes differ from what the decode step (§9-style) assumes, this
test catches it immediately, before any image-dependent debugging.

### 5.2 Model wrapper convention

Each model gets a thin `XyzModel<B: Backend>` wrapper (concrete example:
`YoloSegModel<BurnB>`, Part 2 §13) — `{ model: model_gen::<module>::Model<B>,
device: B::Device }`, with `new(device)` and `forward(...)` matching the
model's actual tensor signature. This wrapper is where model-specific shape
constants (input size, output layout) live — **never** in the `Source` or the
generic plumbing.

### 5.3 Model cache (load once)

Loading/codegen happens once per process per `(BurnB, device)` pair. This is a
**compiled-artifact cache**, the same category as the GPU pipeline cache
(`CLAUDE.md` invariant 9 explicitly permits "the only engine-owned cache is the
GPU pipeline cache" — a model-weights cache is the AI-module's analogue,
**not** a data/tile cache; it lives in `src/ai/`, not the engine core).
Pattern, one pair of statics per model:

```rust
// src/ai/mod.rs (or co-located with the model's module)
use std::sync::OnceLock;

static GPU_MODEL: OnceLock<XyzModel<GpuBurn>> = OnceLock::new();
static CPU_MODEL: OnceLock<XyzModel<CpuBurn>> = OnceLock::new();

pub(crate) fn gpu_model(ctx: &GpuContext) -> &'static XyzModel<GpuBurn> {
    GPU_MODEL.get_or_init(|| XyzModel::new(device::gpu_burn_device(ctx)))
}
pub(crate) fn cpu_model() -> &'static XyzModel<CpuBurn> {
    CPU_MODEL.get_or_init(|| XyzModel::new(device::cpu_burn_device()))
}
```

(If multiple `GpuContext`s with different devices ever coexist, key by
`Arc::as_ptr(ctx)` in a small map instead of `OnceLock` — out of scope for v1,
the engine currently assumes one GPU context. Applies to every model
equally.)

---

## 6. The inference `Source` pattern (generic shape)

Every model's `XyzSource<B>` follows this shape. Concrete instantiation: Part
2 §14 (`SmartMaskSource`).

### 6.1 Struct

```rust
// src/ai/xyz.rs
pub struct XyzSource<B: Backend> {
    /// The staged, preprocessed model-input data (§3).
    staged_input: Data<InputKind, B>,
    /// Whatever extra info `decode_xyz` needs to build the output spec
    /// (e.g. output dimensions) — model-specific, usually tiny.
    output_spec_info: ...,
    params: XyzParams,
}
```

`spec()` returns the fixed output `Kind` value — independent of `wu`, same
contract as `GpuConstantMaskSource`/`GpuConstantLutSource`.

### 6.2 `fetch` — Path A (v1: CPU roundtrip for the small tensor)

```rust
impl Source<GpuBackend> for XyzSource<GpuBackend> {
    type Kind = OutputKind;
    fn spec(&self) -> Arc<OutputKind> { ... }

    fn fetch(&self, ctx: &GpuContext, wu: &<OutputKind as Kind>::WorkUnit)
        -> Result<Buffer<GpuBackend>, Error>
    {
        // 1. Materialize the staged, model-input-shaped buffer (pass #2, §3).
        let input_buf = self.staged_input.materialize(model_input_region())?;

        // 2. GPU -> CPU (small). `GpuBuffer::read_to_cpu` already exists
        //    (used by every `RawTarget`/`RawLutTarget`).
        let bytes: Vec<u8> = input_buf.payload.read_to_cpu(ctx)?;

        // 3. -> Burn tensor, model's expected shape/layout (helper, §6.4).
        let model = crate::ai::gpu_model(ctx); // §5.3
        let input = to_tensor::<GpuBurn>(&bytes, &model.device);

        // 4. Inference — Burn's own dispatch graph, shared (or Burn-owned,
        //    §4.2) wgpu device.
        let raw_outputs = model.forward(input);

        // 5. Decode -> output buffer in OutputKind's layout. MODEL-SPECIFIC —
        //    this is the one step Part 1 does not generalize.
        let decoded: Vec<u8 /* or f32, depends on OutputKind */> =
            decode_xyz(raw_outputs, &self.params, &self.output_spec_info);

        // 6. CPU -> GPU: upload as the OutputKind's buffer.
        upload_buffer::<OutputKind>(ctx, &decoded, &self.output_spec_info)
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        // Same contract as GpuConstantLutSource/GpuConstantMaskSource: a leaf
        // that is ALSO this pull's root calls fetch + cx.output directly (no
        // kernel step, if OutputKind::output is a raw direct write).
        let wu = cx.wu().clone();
        match self.fetch(cx.ctx().as_ref(), &<OutputKind as Kind>::WorkUnit::typed(&wu).expect("wu")) {
            Ok(buf) => {
                cx.input(self.spec().input(), self.spec().source_params(&wu), buf.payload);
                cx.output(self.spec().output(&wu));
            }
            Err(e) => cx.fail(e),
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(NodeId::of(&self.staged_input.root).0 as u64);
        self.params.dyn_hash(state);
    }
}
```

This is **structurally identical** to `GpuConstantLutSource::lower` /
`GpuConstantMaskSource::lower` / `BoundarySource::lower` — a leaf that is its
own root, fetches a buffer, and feeds it through `cx.input` + `cx.output`. If
the model's output is ever consumed by a downstream op in the *same* pass (not
just pulled as a root), `cx.input` alone (no `cx.output`) is the right call —
same branch every other constant source already handles.

### 6.3 `Source<VipsBackend>` — CPU lowering

Same `fetch` shape, different steps 2/6:

- **Step 2** (extraction): `staged_input.materialize(...)` gives a
  `Buffer<VipsBackend>` (`VipsHandle`). Use `vips_image_write_to_memory`
  (already used by `RawLutTarget`'s Vips impl, `src/data/lut.rs`) to get raw
  bytes.
- **Step 4**: `crate::ai::cpu_model().forward(input)`.
- **Step 6** (`upload_buffer` for Vips): build the output as a `VipsHandle` via
  `vips_image_new_from_memory_copy` — same pattern as
  `VipsConstantMaskSource::fetch` in `src/data/mask2d.rs`.

`lower` for `VipsBackend` mirrors `VipsConstantMaskSource::lower`: `fetch` +
`cx.emit((*buf.payload).clone())`.

**No new Vips ops, no new FFI calls beyond what `lut.rs`/`mask2d.rs` already
use — for any model.**

### 6.4 Tensor conversion helpers

Pixel-buffer ↔ tensor layout conversion (e.g. `hwc_u8_to_nchw_f32`, Part 2
§13's concrete version) is plain Rust, generic over the Burn backend
(`B: burn::tensor::backend::Backend`), and **shared across models that share
an input convention** (e.g. "RGB U8 HWC → NCHW f32 [0,1]" is common to most
vision CNNs). Put these in `src/ai/decode.rs` (or `src/ai/tensor.rs` if it
grows) so new models reuse them instead of re-deriving the transpose math.

---

## 7. Feature flag & dependencies

`burn` + `burn-import`/`burn-onnx` are heavy (ML framework, codegen, large
build artifacts). **Gate everything behind a Cargo feature**, default off:

```toml
# Cargo.toml
[features]
default = []
ai = ["burn", "burn-import"]

[dependencies]
burn = { version = "0.x", optional = true, default-features = false, features = ["wgpu", "ndarray"] }

[build-dependencies]
burn-import = { version = "0.x", optional = true }
```

`src/ai/` module is `#[cfg(feature = "ai")]` end to end (the `mod ai;`
declaration in `lib.rs` itself is `#[cfg(feature = "ai")]`). `cargo build --lib`
(CLAUDE.md §9 verification) **without** `--features ai` must remain
unaffected — this is the regression check for "zero core impact," and it
holds regardless of how many models live under `src/ai/`.

---

## 8. `src/ai/` core file map (generic — shared by every model)

```
Cargo.toml              + [features] ai, optional burn/burn-import deps
build.rs                + ModelGen call(s), #[cfg(feature = "ai")]
src/lib.rs               + #[cfg(feature = "ai")] pub mod ai;
src/ai/
  mod.rs                 re-exports, model OnceLocks (§5.3, one pair per model)
  device.rs              GpuBurn/CpuBurn aliases, gpu_burn_device/cpu_burn_device (§4)
  model_gen.rs           include!(OUT_DIR codegen), one `pub mod <model>` per model (§5.1)
  decode.rs              shared tensor<->buffer helpers (hwc_u8_to_nchw_f32 etc, §6.4)

  ── per model, siblings: ──
  smart_mask.rs          (Part 2) SmartMaskSource<B>, Image2D<B>::smart_mask
  yolo_seg.rs            (Part 2) YoloSegModel<B>, constants, decode_mask
  <future_model>.rs       <FutureModel>Source<B>, ergonomic method
  <future_model>_model.rs <FutureModel>Model<B>, constants, decode_<future_model>

tests/
  ai/
    <model>_shapes.rs    RULE AI3: model.forward shape assertions (zero input), one per model
    <model>.rs           e2e test for that model (Part 2 §19 is the smart_mask example)
```

---

## 9. Recipe — adding a new model (the actual "plug in a model" mechanism)

This is what "Burn as a generic backend" cashes out to: per model, do this.
**Nothing below touches the engine core.**

1. Export/obtain an ONNX (or Burn-native) checkpoint. Put it under `models/`,
   document its env-var override (§18 pattern).
2. `ModelGen` it (§5.1) → `Model<B>` with `forward`. If import fails, fall back
   to hand-written `Model<B>` + `burn::record` (RULE AI2) — same downstream
   interface.
3. Write a thin `XyzModel<B>` wrapper (§5.2) with the model's input
   size/output-shape constants. Write `tests/ai/<model>_shapes.rs` (RULE AI3)
   *before* step 4.
4. Write `decode_xyz(...)` mapping the model's raw output tensors to **an
   existing `Kind`'s** buffer layout (`Mask2DKind`, `ImageKind`,
   `HistogramKind`, whatever the model naturally produces). This is the one
   genuinely model-specific piece of logic — everything else in §6 is
   boilerplate.
5. New `XyzSource<B>` (copy §6.2/§6.3 verbatim, swap the model type + step
   4/6 decode/upload calls).
6. New ergonomic method on whichever `Data<K,B>` type is the natural input
   (e.g. `Image2D<B>::xyz(...)`), following §16 AI5 (document `.cache()`).

Steps 1, 2, 5(structure), 6(structure) are **copy-paste-rename**. Step 3 is
~10 lines of constants. **Step 4 is the actual work**, and it is irreducibly
model-specific — no framework abstracts "what does this model's output mean."

---

## 10. Hard rules recap (Part 1, generic — apply to every model)

- **AI1**: GPU device sharing (`burn-wgpu` existing-device API) is
  best-effort; a Burn-owned device is an acceptable fallback. Isolated behind
  `src/ai/device.rs::gpu_burn_device`, shared by all models.
- **AI2**: ONNX import may not cover all of a model's ops; hand-written
  `Model<B>` + `burn::record` is the fallback, same `forward` interface. Don't
  block plumbing on this — stub `forward` first.
- **AI3**: shape-assertion test against a zero tensor, written *before* wiring
  real decode logic, per model.
- **AI5**: inference is **not memoized** — `.stage()` has no store
  (`docs/staging-foundation.md`), so every pull of a model's output re-runs
  preprocessing + inference + decode. Document `.cache()` as the required
  pattern for reuse, for every model's ergonomic method, with a test proving
  the cache hit (Part 2 §19 is the worked example).
- **Zero core changes.** `XyzSource<B>` is "just a `Source`" — same category
  as `GpuConstantLutSource`/`GpuConstantMaskSource`/`BoundarySource`. If at any
  point implementing a model requires touching `kind.rs`, `operation/mod.rs`,
  `node.rs`, `io.rs`, `pass.rs`, `emit.rs`, or `compile.rs` — **stop and write
  a doc**: that means the *model's output* needs a new `Kind` (a normal,
  allowed extension, CLAUDE.md §5 "Adding a datatype"), not that the AI
  plumbing needs a special case.

---

# PART 2 — Smart Mask (YOLO Segmentation): first application

Everything below is a concrete instantiation of Part 1's pattern:
`XyzSource<B>` → `SmartMaskSource<B>`, `XyzModel<B>` → `YoloSegModel<B>`,
`XyzParams` → `SmartMaskParams`, output `Kind` → `Mask2DKind` (existing,
already `GpuView` + `VipsBand`).

## 11. Goals (smart-mask specific)

- `img.smart_mask(&params) -> Mask2D<B>` — one combined `f32` mask (`[0,1]`),
  same `(width, height)` as `img`, where `1.0` = "inside a detected instance
  of a selected class".
- Model = YOLOv8-seg (or compatible) imported via `burn-onnx`/`burn-import`
  codegen (Part 1 §5.1), generic over the Burn backend
  (`YoloSegModel<BurnB>`).
- **Not** per-instance masks (no band-per-instance v1). All matching instances
  are unioned (max) into one mask. Multi-instance output is a v2 extension
  (§20).
- **Not** letterboxed preprocessing. v1 resizes (stretches) to the model's
  fixed input size; aspect-ratio distortion is accepted (documented
  limitation, §20).
- **Not** NMS/box decode as GPU kernels — runs on a small CPU-side tensor
  (hundreds of rows), not worth a Slang kernel.

---

## 12. Preprocessing (ordinary chromors ops)

YOLOv8 training preprocessing: RGB, `[0,1]` range (no per-channel mean/std),
square input, letterboxed (v1: stretched, §11).

```rust
// inside Image2D<B>::smart_mask (§16)
use crate::pixel::{PixelLayout, Storage, ColorModel, AlphaState};
use crate::color::ColorSpace;

const MODEL_LAYOUT: PixelLayout = PixelLayout {
    storage: Storage::U8,
    model: ColorModel::Rgb,
    alpha: AlphaState::None,
    color_space: ColorSpace::SRGB, // gamma-encoded sRGB, matches typical training data
};

let (w, h) = self.spec.dims(); // existing helper, see ImageKind
let (mw, mh) = yolo_seg::MODEL_INPUT; // (640, 640)
let sx = mw as f64 / w as f64;
let sy = mh as f64 / h as f64;

let preprocessed = self
    .convert(MODEL_LAYOUT)              // native color management (existing op)
    .resize(sx, None, Some(sy), None);  // stretch to exactly 640x640 (existing op)

let staged = preprocessed.stage();     // BoundarySource, no store (Part 1 §3)
```

**RULE AI4:** `convert` happens **before** `resize` so the resize kernel
operates on the model's storage format (avoids resizing in linear light and
then converting — cheaper and matches what `vips`-style pipelines do; either
order is *correct* per native color management, this order is *conventional*).
This is smart-mask-specific guidance — other models with different input
conventions follow whatever their own training preprocessing requires, using
the same `<ops>.stage()` shape (Part 1 §3).

`.stage()` guarantees `SmartMaskSource::fetch` (§14) sees a tightly-packed
`640 x 640 x 3` `U8` buffer, regardless of the original image's size/format —
**all model-input-shape logic lives in `SmartMaskSource`/`YoloSegModel`, never
in the generic plumbing (Part 1)**.

---

## 13. `YoloSegModel<BurnB>` — model wrapper (Part 1 §5.2 instance)

### 13.1 Codegen (Part 1 §5.1 instance)

```rust
// build.rs, appended, only when feature = "ai":
#[cfg(feature = "ai")]
{
    burn_import::onnx::ModelGen::new()
        .input("models/yolov8n-seg.onnx")
        .out_dir("ai/") // -> OUT_DIR/ai/yolov8n_seg.rs
        .run_from_script();
}
```

```rust
// src/ai/model_gen.rs
#[cfg(feature = "ai")]
pub mod yolov8n_seg {
    include!(concat!(env!("OUT_DIR"), "/ai/yolov8n_seg.rs"));
}
```

### 13.2 Wrapper

```rust
// src/ai/yolo_seg.rs
use burn::tensor::{Tensor, backend::Backend};

/// Fixed by the chosen YOLOv8-seg export. v1: YOLOv8n-seg @ 640x640, 80 COCO
/// classes, 32 mask coefficients/prototypes.
pub const MODEL_INPUT: (u32, u32) = (640, 640);
pub const NUM_CLASSES: usize = 80;
pub const NUM_MASK_COEFFS: usize = 32;
pub const PROTO_SIZE: (u32, u32) = (160, 160);

pub struct YoloSegModel<B: Backend> {
    model: model_gen::yolov8n_seg::Model<B>,
    device: B::Device,
}

impl<B: Backend> YoloSegModel<B> {
    pub fn new(device: B::Device) -> Self {
        Self {
            model: model_gen::yolov8n_seg::Model::new(&device),
            device,
        }
    }

    /// `input`: `[1, 3, 640, 640]`, f32, channel order matching the ONNX
    /// export's training preprocessing (§12 — typically RGB, [0,1], no
    /// mean/std normalization for YOLOv8).
    /// Returns `(detections, proto)`:
    ///   - `detections`: `[1, 4 + NUM_CLASSES + NUM_MASK_COEFFS, N]`
    ///     (box xywh + per-class scores + mask coeffs, N anchors, e.g. 8400)
    ///   - `proto`: `[1, NUM_MASK_COEFFS, 160, 160]` mask prototypes
    pub fn forward(&self, input: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 4>) {
        self.model.forward(input)
    }
}
```

> The exact output tensor shapes/order depend on the specific YOLOv8-seg ONNX
> export. The numbers above are the standard `ultralytics` export layout.
> **RULE AI3 (instance):** `tests/ai/yolo_seg_shapes.rs` runs `forward` on a
> zero tensor and asserts the output shapes — if they differ, §15's decode
> indices need updating, nothing else.

### 13.3 Model cache (Part 1 §5.3 instance)

```rust
// src/ai/mod.rs
use std::sync::OnceLock;

static GPU_MODEL: OnceLock<YoloSegModel<GpuBurn>> = OnceLock::new();
static CPU_MODEL: OnceLock<YoloSegModel<CpuBurn>> = OnceLock::new();

pub(crate) fn gpu_model(ctx: &GpuContext) -> &'static YoloSegModel<GpuBurn> {
    GPU_MODEL.get_or_init(|| YoloSegModel::new(device::gpu_burn_device(ctx)))
}
pub(crate) fn cpu_model() -> &'static YoloSegModel<CpuBurn> {
    CPU_MODEL.get_or_init(|| YoloSegModel::new(device::cpu_burn_device()))
}
```

---

## 14. `SmartMaskSource<B>` (Part 1 §6 instance)

### 14.1 Struct

```rust
// src/ai/smart_mask.rs
pub struct SmartMaskSource<B: Backend> {
    /// The staged, preprocessed `640x640` model-input image (§12).
    staged_input: Data<ImageKind, B>,
    /// Original image dims — the mask is upsampled to this size (§15.5).
    output_dims: (i32, i32),
    params: SmartMaskParams,
}
```

`spec()` returns `Arc::new(Mask2DKind::new(output_dims.0, output_dims.1))` —
fixed, independent of `wu` (`Region`-shaped Kind, same contract as
`GpuConstantMaskSource`).

### 14.2 `fetch` — GPU (Path A: CPU roundtrip for the small tensor)

```rust
impl Source<GpuBackend> for SmartMaskSource<GpuBackend> {
    type Kind = Mask2DKind;
    fn spec(&self) -> Arc<Mask2DKind> { ... }

    fn fetch(&self, ctx: &GpuContext, wu: &Region) -> Result<Buffer<GpuBackend>, Error> {
        // 1. Materialize the staged 640x640x3 U8 buffer (pass #2, Part 1 §3).
        let model_region = Region::full(yolo_seg::MODEL_INPUT_I32, Lod(0));
        let input_buf = self.staged_input.materialize(model_region)?;

        // 2. GPU -> CPU (small: 640*640*3 = 1.2 MB).
        let bytes: Vec<u8> = input_buf.payload.read_to_cpu(ctx)?; // U8 RGB, row-major

        // 3. -> Burn tensor [1,3,640,640], f32 in [0,1], NCHW (HWC->CHW
        //    transpose happens here, in plain Rust — see §14.4).
        let model = crate::ai::gpu_model(ctx);
        let input = hwc_u8_to_nchw_f32::<GpuBurn>(&bytes, yolo_seg::MODEL_INPUT, &model.device);

        // 4. Inference — Burn's own dispatch graph, on the shared (or
        //    Burn-owned, Part 1 §4.2) wgpu device.
        let (detections, proto) = model.forward(input);

        // 5. Decode -> f32 mask at `output_dims` (§15). Runs on CPU (small
        //    tensors). MODEL-SPECIFIC — the one non-generic step.
        let mask_f32: Vec<f32> = decode_mask(detections, proto, &self.params, self.output_dims);

        // 6. CPU -> GPU: upload as the Mask2DKind output buffer.
        upload_mask_buffer(ctx, &mask_f32, self.output_dims)
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        // Same contract as GpuConstantLutSource/GpuConstantMaskSource (Part 1
        // §6.2): a leaf that is ALSO this pull's root calls fetch + cx.output
        // directly (no kernel step — Mask2DKind::output is a raw direct
        // write, no codec).
        let wu = cx.wu().clone();
        match self.fetch(cx.ctx().as_ref(), &Region::typed(&wu).expect("Region wu")) {
            Ok(buf) => {
                let geom = RegionParams::tight(self.output_dims.0, self.output_dims.1);
                cx.input(self.spec().input(), geom.into_block("region_in_{slot}"), buf.payload);
                cx.output(self.spec().output(&wu));
            }
            Err(e) => cx.fail(e),
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(NodeId::of(&self.staged_input.root).0 as u64);
        self.params.dyn_hash(state);
    }
}
```

If `SmartMask` is ever consumed by a downstream op in the *same* pass (not
just pulled as a root), `cx.input` alone (no `cx.output`) is the right call —
same branch every other constant source already handles (Part 1 §6.2).

### 14.3 `Source<VipsBackend>` (Part 1 §6.3 instance)

Same `fetch` shape, different steps 2/6:

- **Step 2** (pixel extraction): `staged_input.materialize(...)` gives a
  `Buffer<VipsBackend>` (`VipsHandle`). Use `vips_image_write_to_memory`
  (already used by `RawLutTarget`'s Vips impl, `src/data/lut.rs`) to get raw
  `U8 RGB` bytes — same `hwc_u8_to_nchw_f32::<CpuBurn>` helper (generic over
  `B`, §14.4).
- **Step 4**: `crate::ai::cpu_model().forward(input)`.
- **Step 6** (`upload_mask_buffer` for Vips): build the mask as a 1-band
  `VIPS_FORMAT_FLOAT` image via `vips_image_new_from_memory_copy` — same
  pattern as `VipsConstantMaskSource::fetch` in `src/data/mask2d.rs` (copy that
  function's body, source = `mask_f32`, `bands=1`, format=`FLOAT`).

`lower` for `VipsBackend` mirrors `VipsConstantMaskSource::lower`: `fetch` +
`cx.emit((*buf.payload).clone())`.

**No new Vips ops, no new FFI calls beyond what `lut.rs`/`mask2d.rs` already
use.**

### 14.4 `hwc_u8_to_nchw_f32` (Part 1 §6.4 instance — shared helper)

```rust
// src/ai/decode.rs — shared by any model with this input convention
fn hwc_u8_to_nchw_f32<B: burn::tensor::backend::Backend>(
    rgb: &[u8], (w, h): (u32, u32), device: &B::Device,
) -> Tensor<B, 4> {
    let (w, h) = (w as usize, h as usize);
    let mut chw = vec![0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                chw[c * w * h + y * w + x] = rgb[(y * w + x) * 3 + c] as f32 / 255.0;
            }
        }
    }
    Tensor::<B, 1>::from_floats(chw.as_slice(), device).reshape([1, 3, h, w])
}
```

---

## 15. Postprocessing: detections + proto → one `f32` mask (`decode_mask`)

This is plain Rust over small tensors (`detections`: `[1, 4+80+32, 8400]` ≈
2.8M f32 = 11 MB; `proto`: `[1,32,160,160]` ≈ 0.8M f32 = 3.3 MB). Runs on CPU
regardless of backend (download both tensors via `Tensor::into_data()`).

### 15.1 Filter by class + confidence

For each of the 8400 anchors: `class_score[c] = detections[4+c, a]` for
`c in params.class_filter` (or all 80 if `None`); keep anchor if
`max(class_score) >= params.confidence_threshold`.

### 15.2 NMS (per kept anchor, box = `detections[0..4, a]`, xywh)

Standard greedy IoU-based NMS with `params.iou_threshold`. v1: cap at
`params.max_detections` (default 100) to bound cost.

### 15.3 Per-instance mask: `sigmoid(coeffs · proto)`

For each kept detection, `coeffs = detections[4+80 .. 4+80+32, a]` (32
floats). Mask logits `[160,160] = Σ_k coeffs[k] * proto[k,:,:]` — a
`[1,32] @ [32, 160*160]` matmul (`Tensor::matmul`, on whichever Burn backend —
GPU or CPU, this is the one piece of postproc worth keeping on-device since
it's a real matmul). Apply `sigmoid`, threshold at `0.5` → binary `[160,160]`.

### 15.4 Crop to box + union

Each instance mask is valid only inside its detection box (scaled from
640-space to 160-space, i.e. `/4`). Zero it outside the box. Union (`max`)
across all kept instances → one `[160,160]` mask.

### 15.5 Upsample `160x160 -> output_dims` (bilinear, CPU)

`output_dims` = the **original** image's `(width, height)` (§14.1), not
640×640 — this is where the model-resolution mask is mapped back to the
caller's coordinate space, so `Mask2D<B>` is directly usable against `img`
(`img.maplut(...)`-style composition, opacity masking, etc. all assume
matching dims). Plain nearest/bilinear loop over `output_dims` pixels — at
typical photo resolutions (≤ 50MP) this is microseconds, not worth a kernel.

```rust
fn decode_mask<B: Backend>(
    detections: Tensor<B, 3>, proto: Tensor<B, 4>,
    params: &SmartMaskParams, output_dims: (i32, i32),
) -> Vec<f32> {
    // §15.1-15.4 -> Vec<f32> mask_160x160 (0.0 / 1.0)
    // §15.5 -> bilinear-resample to output_dims, return row-major f32
}
```

---

## 16. `SmartMaskParams` + ergonomic constructor

```rust
// src/ai/mod.rs — AGNOSTIC (no Burn/wgpu/vips types)
#[derive(Clone, Debug, PartialEq)]
pub struct SmartMaskParams {
    /// COCO class ids to include. `None` = all 80 classes.
    pub class_filter: Option<Vec<u32>>,
    pub confidence_threshold: f32,   // default 0.25
    pub iou_threshold: f32,          // default 0.45
    pub max_detections: usize,       // default 100
}
impl Default for SmartMaskParams {
    fn default() -> Self {
        Self { class_filter: None, confidence_threshold: 0.25, iou_threshold: 0.45, max_detections: 100 }
    }
}
impl SmartMaskParams {
    pub(crate) fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        if let Some(c) = &self.class_filter { for v in c { state.write_u32(*v); } }
        state.write_u32(self.confidence_threshold.to_bits());
        state.write_u32(self.iou_threshold.to_bits());
        state.write_usize(self.max_detections);
    }
}
```

```rust
// src/ai/smart_mask.rs — per-backend, where-clause gated like every op
impl Image2D<GpuBackend> {
    pub fn smart_mask(&self, params: SmartMaskParams) -> Mask2D<GpuBackend> {
        let staged = /* §12 */;
        let src = SmartMaskSource {
            staged_input: staged,
            output_dims: self.spec.dims(),
            params,
        };
        Data::from_source(Arc::new(src), self.ctx.clone())
    }
}
// + identical impl for Image2D<VipsBackend> using CpuBurn / VipsHandle paths (§14.3)
```

Usage:

```rust
let mask = photo.smart_mask(SmartMaskParams {
    class_filter: Some(vec![0]), // COCO class 0 = "person"
    ..Default::default()
});
let portrait_blur = photo.blur(8.0);
// composite photo-over-portrait_blur using `mask` as alpha — existing ops
```

---

## 17. Caching & cost (Part 1 §10 AI5, applied)

Inference is **expensive** (tens of ms on GPU, hundreds on CPU) and `.stage()`
is **not memoized** (Part 1 §3/§10 — `docs/staging-foundation.md`) — every pull
of `mask` re-runs preprocessing + inference + postprocessing.

**RULE AI5 (instance):** document prominently (rustdoc on `smart_mask`) that
callers needing the mask across multiple pulls (the common case — a mask is
usually reused for several output regions/exports) **must** call `.cache()`:

```rust
let mask = photo.smart_mask(params).cache();
let region1 = mask.handle(); // first pull: runs inference, caches whole 640x640->orig mask
let region2 = mask.handle(); // later pulls: cache hit, no re-inference
```

Because `Mask2DKind` is `Region`-shaped and the source always produces the
*entire* `output_dims` mask regardless of `wu` (§14.1 `spec()` is
`wu`-independent... well, `fetch`/`lower` currently demand exactly `wu`, but
since the source always computes the full mask internally and `cx.input`
binds the *whole* tight buffer via `region_in_{slot}` — any `wu` within
`output_dims` reads from the same full-mask buffer). One `.cache()` boundary
at `content = NodeId::of(smart_mask_root)` covers every region/tile request —
exactly the "layer-stack" use case `cache.rs`'s doc comment describes.

---

## 18. File map additions (smart-mask specific, on top of Part 1 §8)

```
models/
  yolov8n-seg.onnx       (NOT committed if large — document download/path;
                          path configurable via env var or `SmartMaskParams`-
                          adjacent config, e.g. CHROMORS_YOLO_SEG_ONNX)
src/ai/
  smart_mask.rs          SmartMaskSource<B> (Source<GpuBackend> + Source<VipsBackend>),
                         Image2D<B>::smart_mask ergonomic methods
  yolo_seg.rs            YoloSegModel<B>, MODEL_INPUT/NUM_CLASSES/etc constants, decode_mask
tests/ai/
  yolo_seg_shapes.rs     RULE AI3: model.forward shape assertions (zero input)
  smart_mask.rs          e2e: smart_mask on a fixture image, both backends,
                          assert mask dims == image dims, assert cache hit on
                          second pull (RULE AI5)
```

---

## 19. Tests

- `tests/ai/yolo_seg_shapes.rs` (RULE AI3): `YoloSegModel::<NdArray<f32>>::new(default())
  .forward(zeros([1,3,640,640]))` → assert output shapes match §13.2 constants.
  CPU-only, fast, no fixture image needed — catches ONNX-import shape drift
  immediately.
- `tests/ai/smart_mask.rs`:
  - Load a fixture photo containing a known object (e.g. a person).
  - `photo.smart_mask(SmartMaskParams{class_filter: Some(vec![0]), ..default()})`
    on both `GpuBackend` and `VipsBackend`.
  - Assert `mask.width() == photo.width() && mask.height() == photo.height()`.
  - Assert the mask is not all-zero and not all-one (sanity — model actually
    ran and found *something*, without depending on exact pixel values, which
    are model-version-sensitive).
  - `.cache()` test: call `RegionCache::stats()` before/after two pulls,
    assert `hits` increments on the second pull (RULE AI5) — same pattern as
    `tests/gpu/cache.rs`.
  - All `#[cfg(feature = "ai")]`, `#[ignore]`-by-default if the ONNX file
    isn't present in CI (check via `std::path::Path::exists` + early-return
    skip with a printed message — do not fail CI on a missing multi-MB model
    file).

---

## 20. Open questions / risks (for discussion, not blockers)

1. **ONNX operator coverage** (AI2) — the biggest unknown. Worth a 30-minute
   spike importing a real `yolov8n-seg.onnx` with the pinned `burn-import`
   version *before* committing to the architecture above, just to know which
   fallback path (codegen vs hand-written) is needed. The architecture above
   is identical either way.
2. **Letterboxing** (§11) — v2. Requires the preprocessing chain to pad
   (constant-color border) rather than stretch; `decode_mask`'s §15.5 upsample
   then needs to account for the padding offset/scale. Self-contained change
   inside `smart_mask`/`decode.rs`, no architectural impact.
3. **Multi-instance masks** (§11) — v2. Would change `Mask2DKind` usage to
   `bands = N` (it currently only supports `width x height`, 1 implicit band —
   confirm whether `Mask2DKind` needs a `bands` field added, which *would* be
   a small AGNOSTIC-side change, or whether N separate `Mask2D` pulls
   (N separate `SmartMaskSource`s sharing a `.cache()`'d detection-decode
   step) is cleaner). Flag for a follow-up doc if pursued.
4. **Zero-copy GPU tensor handoff** (§14.2 Path A → v2) — once AI1's
   device-sharing is confirmed working, investigate wrapping
   `GpuBuffer::buffer()` (`Arc<wgpu::Buffer>`) directly as a Burn/CubeCL tensor
   handle on the Wgpu runtime, skipping `read_to_cpu`. Purely an optimization
   inside `SmartMaskSource::fetch` step 2-3; no interface change. Applies
   equally to any future model (Part 1 §1.2).
5. **Model file distribution** — `models/yolov8n-seg.onnx` is several MB.
   Decide: commit via Git LFS, download-on-build script, or
   user-supplied-path-only (document a `CHROMORS_YOLO_SEG_ONNX` env var, error
   clearly if unset when `feature = "ai"` + `smart_mask` is called). Not an
   architecture question — pick whichever fits the repo's existing
   asset-handling conventions. Future models follow the same pattern with
   their own env var.
6. **Second model as validation** — once smart_mask ships, picking a second,
   structurally different model (e.g. a classifier producing a `Histogram`-ish
   `Kind`, or a depth model producing `ImageKind`) and running it through Part
   1 §9's recipe is the real test of "generic backend" — it should mostly
   exercise steps 1/2/5/6 (copy-paste) plus step 3/4 (new constants + new
   decode), with zero changes to Part 1 §§3-8.
