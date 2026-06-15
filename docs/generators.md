# Chromors Generators — Detailed Implementation Refinement

> **Audience:** an implementer (human or AI) who will write the code with **zero
> room for interpretation**. Every type, trait, coordinate convention, kernel
> signature, and algorithm is spelled out. Where a decision could go two ways,
> this doc picks one and states it as a RULE.
>
> **Prerequisite reading:** `CLAUDE.md` (the engine contract) and
> `docs/ARCHITECTURE.md`, especially §3 (traits), §5 (GPU emit/compile), §7
> (Slang). This document adds **generators** — zero-input procedural sources —
> without changing any engine invariant. A generator is an ordinary
> `Source<B>`; it just lowers to a *kernel that writes from coordinates* instead
> of an *uploaded buffer*.

---

## 0. What a generator is (one sentence)

A **generator** is a graph leaf (`Source<B>`) whose value at `(x, y)` for a
requested `Region` at any `Lod` is a **pure function of the coordinate and a
fixed parameter set** — no image inputs, no CPU upload: on GPU it emits a Slang
compute kernel that writes its region directly in VRAM and fuses with whatever
consumes it.

Examples: constant fill, linear/radial gradients, coordinate ramps (`xyz`),
test patterns (zone plate, sines, eye chart), and **tiling-correct noise**
(gauss, perlin, worley).

---

## 1. Goals & non-goals

### 1.1 Goals
1. Generators are **`Source` leaves**, lowered like everything else — no new
   node kind, no central enum, no materializer change.
2. **GPU-native**: a generator emits a Slang kernel that writes its tile in
   VRAM and **participates in fusion** (`gaussnoise → add → blur` collapses).
3. **Region/LOD native**: pulling a `Region` at `Lod(n)` computes only that
   downsampled region — generators are resolution-independent infinite-canvas
   sources that cost only what you pull.
4. **Tiling-correct**: `pull(A)` then `pull(B)` reconstructs `pull(A ∪ B)`
   exactly, including noise (counter-based, coordinate-keyed RNG).
5. **CPU parity**: the same generator produces byte-comparable output on the
   `VipsBackend` via a native coordinate-keyed evaluation (NOT a sequential CPU
   RNG), so previews and headless renders agree.
6. **Idiomatic & generic**: one `Generator` trait + two blanket `Source` impls;
   each concrete generator is ~30 lines (params + kernel name + CPU eval).

### 1.2 Non-goals (v1)
- Generators that take image inputs (those are **operations**, not generators).
- Vector/SVG sources (that is the viewport's Vello path).
- Frequency-domain mask builders (`mask_*`) — reserved §12.4.
- Output Kinds other than `ImageKind` (the trait is written so a future
  `PointListKind` generator slots in, but v1 is image-only).

---

## 2. Where generators sit in the model

```
                       ┌─ Source::lower (GPU)  → cx.kernel(...) + cx.output(...)
   Generator (a leaf) ─┤
                       └─ Source::lower (Vips) → render region in Rust → emit vips image
```

A normal source (`VipsImageSource`) lowers by **fetching + uploading a buffer**
and calling `cx.input(view, geom, buffer)` — registering a *source slot* the
emitter decodes. A generator instead calls `cx.kernel(...)`, which registers a
*kernel step* writing a `work_{k}` temp. The downstream consumer therefore
reads the generator as `BaseInput::Step(k)` (a fused temp), not as an uploaded
source buffer. **That single difference is the whole feature.**

> **RULE GEN0.** A generator's `lower` never calls `cx.input` and never uploads
> a buffer on the GPU path. It calls `cx.kernel` (one step, zero read inputs)
> then `cx.output`.

> **RULE GEN1.** A generator **always** calls `cx.output(self.spec().output(cx.wu().clone()))`
> at the end of `lower` (unlike `VipsImageSource`, whose missing root-output is a
> known bug). This makes a bare generator (`gradient(...).pull(...)`) a valid
> root: its single kernel step writes `work_0`, and the codec sandwich encodes
> it to the target.

---

## 3. The coordinate model (READ THIS TWICE — it is the core)

The emitter calls every kernel as:

```
kernel(idx, <read args...>, out_var, <scalar params...>)
```

where `idx = SV_DispatchThreadID.xy`, ranging `0 .. domain.width/height`, and
`domain` is the **dispatch domain = the requested region's `(w, h)`**. `idx` is
therefore **tile-local** (0-based within the pulled region) — it is NOT an
absolute image coordinate. Existing pointwise ops don't care, but a generator
must know *where in the infinite canvas* it is.

The resolved `Region` (absolute origin + size + LOD) is available to the
generator's `lower` via `cx.wu()`. The generator pushes the bits the kernel
needs as scalar params, and the kernel reconstructs the absolute coordinate.

### 3.1 Canonical coordinate reconstruction (every generator kernel uses this)

The generator's `lower` pushes, **in this exact order, immediately after
`cx.kernel`** (before any generator-specific params):

```rust
let r: Region = Region::typed(cx.wu()).expect("generator: Region WorkUnit");
cx.kernel(MODULE, ENTRY);
cx.param("gen_ox", r.x);                       // i32  region origin x (LOD space)
cx.param("gen_oy", r.y);                       // i32  region origin y (LOD space)
cx.param("gen_lod", r.lod.0 as i32);           // i32  LOD level
cx.param("gen_fw", self.spec().width);         // i32  full-res canvas width
cx.param("gen_fh", self.spec().height);        // i32  full-res canvas height
// ... then generator-specific params, in the same order as the kernel sig ...
cx.output(self.spec().output(cx.wu().clone()));
```

The kernel's first five scalar params are therefore always
`int gen_ox, int gen_oy, int gen_lod, int gen_fw, int gen_fh`, followed by the
generator's own params. The Slang helper in §9.2 turns these into coordinates.

### 3.2 The coordinate math (identical in Slang and Rust)

```text
local      = int2(idx)                         // 0 .. (w-1, h-1) within the tile
lod_coord  = int2(gen_ox, gen_oy) + local      // absolute coordinate in LOD space
scale      = 1 << gen_lod                       // 2^lod
full_px    = (float2(lod_coord) + 0.5) * scale  // full-res pixel CENTER
uv         = full_px / float2(gen_fw, gen_fh)    // normalized [0,1) over canvas
```

A generator evaluates its pure function at `full_px` (absolute pixels) or `uv`
(normalized) — its choice, documented per generator. The **RNG key** (§7) is
`lod_coord` (the integer LOD-space grid), mixed with `gen_lod` and the seed.

### 3.3 Why this is tiling-correct (the proof, do not deviate)

- Within one LOD, `lod_coord = origin + local` depends only on the absolute
  position, never on how the canvas was split. Pixel `(100, 50)` gets the same
  `lod_coord` whether it arrived in tile `A` (origin 0) at local `(100,50)` or
  tile `B` (origin 96) at local `(4,50)`. So any function of `lod_coord` —
  including a coordinate-keyed RNG — yields identical results regardless of
  tiling: `pull(A) ∪ pull(B) == pull(A ∪ B)`.
- Across LODs the sampling grid changes by design (a coarser preview samples a
  sparser set of full-res centers); this is correct downsampling, not an
  inconsistency.

> **RULE GEN2.** Generators must derive all spatial values from `lod_coord` /
> `full_px` / `uv` as defined above. **Never** use raw `idx` for anything but
> the local write position (`out.write(idx, value)`). Using `idx` as a world
> coordinate silently breaks tiling and LOD.

---

## 4. The generic core (`src/data/generator.rs`)

One new file. It defines the `Generator` trait and two **blanket** `Source`
impls (GPU + Vips) so each concrete generator implements `Generator` once and
gets both backends for free — mirroring how an `Operation` implements
`Lower<GpuBackend>` and `Lower<VipsBackend>`.

### 4.1 The trait

```rust
use std::hash::Hasher;
use std::sync::Arc;
use crate::data::image::ImageKind;
use crate::backend::gpu::GpuBuilder;
use crate::backend::vips::VipsBuilder;
use crate::work_unit::Region;
use crate::error::Error;

/// A zero-input procedural image source. Implement this once; the blanket impls
/// in §4.2 give you `Source<GpuBackend>` and `Source<VipsBackend>`.
pub trait Generator: Send + Sync + 'static {
    /// Output metadata: pixel layout + full-res extent. Same value on every
    /// backend (agnostic), exactly like an op's `output_spec`.
    fn spec(&self) -> Arc<ImageKind>;

    // ── GPU path ──────────────────────────────────────────────────────────
    /// `(slang_module, entry_point)` of this generator's kernel, e.g.
    /// `("ops.generators", "gradient_kernel")`.
    fn gpu_kernel(&self) -> (&'static str, &'static str);

    /// Push this generator's OWN scalar params (after the 5 canonical ones,
    /// which the blanket impl already pushed) via `cx.param(...)`, in the same
    /// order as the kernel signature's trailing args. Default: no params.
    fn gpu_params(&self, _cx: &mut GpuBuilder) {}

    // ── CPU (Vips) path ───────────────────────────────────────────────────
    /// Render `region` to tightly-packed bytes in this generator's storage
    /// layout (`self.spec().layout`). MUST use the §3.2 coordinate math so CPU
    /// output matches GPU and stays tiling-correct. Called by the Vips blanket
    /// `lower`/`fetch`; the bytes are wrapped as a vips memory image.
    fn render_cpu(&self, region: &Region) -> Result<Vec<u8>, Error>;

    /// Identity for the pipeline cache key (hash every parameter + dims).
    fn dyn_hash(&self, state: &mut dyn Hasher);
}
```

> **RULE GEN3.** `gpu_kernel` math and `render_cpu` math MUST be the same
> function (the doc for each generator gives one formula; both sites implement
> it). A divergence is a correctness bug, not a tolerance issue. Tests in §13
> assert GPU≈CPU within rounding.

### 4.2 Blanket `Source` impls

A newtype wraps the generator so the blanket impls don't conflict with other
`Source` impls in the crate. Constructors (`§10`) build `Data::from_source` over
`GenSource<G>`.

```rust
use crate::io::Source;
use crate::buffer::Buffer;
use crate::backend::Backend;
use crate::backend::gpu::{GpuBackend, GpuContext, view::RegionParams};
use crate::backend::vips::VipsBackend;
use crate::work_unit::WorkUnit;

/// Wrapper that carries a `Generator` as a graph leaf.
pub struct GenSource<G: Generator>(pub G);

impl<G: Generator> Source<GpuBackend> for GenSource<G> {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> { self.0.spec() }

    fn fetch(&self, _ctx: &GpuContext, _wu: &Region) -> Result<Buffer<GpuBackend>, Error> {
        // GPU generators are realized through `lower` (kernel emit), never a
        // standalone upload. `fetch` is unused on the GPU materialize path.
        Err(Error::Backend("generator: use lower(), not fetch(), on GPU".into()))
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let Some(r) = Region::typed(cx.wu()) else {
            cx.fail(Error::InvalidWorkUnit("generator expects a Region".into()));
            return;
        };
        let (module, entry) = self.0.gpu_kernel();
        cx.kernel(module, entry);
        // §3.1 canonical params (order fixed):
        cx.param("gen_ox", r.x);
        cx.param("gen_oy", r.y);
        cx.param("gen_lod", r.lod.0 as i32);
        cx.param("gen_fw", self.0.spec().width);
        cx.param("gen_fh", self.0.spec().height);
        // generator-specific params:
        self.0.gpu_params(cx);
        // codec sandwich output (image encode of the float4 working temp):
        cx.output(self.0.spec().output(cx.wu().clone()));
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) { self.0.dyn_hash(state); }
}

impl<G: Generator> Source<VipsBackend> for GenSource<G> {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> { self.0.spec() }

    fn fetch(&self, _ctx: &(), wu: &Region) -> Result<Buffer<VipsBackend>, Error> {
        let bytes = self.0.render_cpu(wu)?;
        let spec = self.0.spec();
        let handle = crate::backend::vips::image_from_memory(
            &bytes, wu.w, wu.h, spec.layout,
        )?;
        Ok(Buffer { payload: Arc::new(handle), spec })
    }

    fn lower(&self, cx: &mut VipsBuilder) {
        let wu = Region::typed(cx.wu()).expect("generator expects a Region");
        match self.fetch(&(), &wu) {
            Ok(buf) => cx.emit((*buf.payload).clone()),
            Err(e) => { /* VipsBuilder error channel; see §6.3 */ cx.fail(e); }
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) { self.0.dyn_hash(state); }
}
```

> **RULE GEN4.** `image_from_memory(bytes, w, h, layout)` is a new guarded helper
> in `src/backend/vips/mod.rs` (it calls `ensure_init()` then
> `vips_image_new_from_memory_copy`, deriving bands + `VipsBandFormat` from
> `layout` exactly as `RawImageSource::fetch` does today). Do NOT call the raw
> FFI from `generator.rs` — route through this helper (see the vips-init memory).

> **RULE GEN5.** If `VipsBuilder` has no `fail` method yet, add one mirroring
> `GpuBuilder::fail` (store first error, surfaced by `finish`). Check the current
> `VipsBuilder` before assuming; if it already returns `Result` from `emit`,
> propagate instead.

### 4.3 Why a trait + blanket impls (not 2 hand-written `Source` impls each)
Mirrors `Operation`/`Lower`. A concrete generator (`Gradient`) writes one
`impl Generator for Gradient` and is instantly usable on GPU and CPU. No
per-generator `Source` boilerplate; the canonical coordinate-param plumbing
lives in exactly one place (§4.2).

---

## 5. GPU lowering — worked trace

Take `gradient(ctx, 4000, 3000, angle)` consumed by `.blur(8.0)`, pulled at a
tile `Region { x: 1024, y: 512, w: 256, h: 256, lod: 1 }`.

1. **Demand walk.** `blur` demands its input expanded by its halo → the
   generator node's demand is `Region { x: 1024-h, y: 512-h, w: 256+2h, ... }`.
2. **Lower walk (post-order).** Generator lowers first:
   - `cx.kernel("ops.generators", "gradient_kernel")` → `steps[0]`, writes
     `work_0`, **zero read inputs** (`cur_inputs` is empty for a leaf).
   - pushes `gen_ox=1024-h, gen_oy=…, gen_lod=1, gen_fw=4000, gen_fh=3000`,
     then `angle`.
   - `cx.output(ImageKind::output(wu))` → codec sandwich (scratch + encode).
   - `last_step_of[gen] = 0`.
3. `blur` lowers next: its first kernel step reads `BaseInput::Step(0)` →
   `work_0`. **Fusion achieved** — the gradient is computed in-register and read
   by blur; no intermediate buffer, no readback.
4. **Emit.** One shader, bindings `target, params, work_0` (+ blur's temps).
   The generator step emits:
   ```hlsl
   RWRegion out_0 = { work_0, params[0].domain };
   gradient_kernel(idx, out_0,
       params[0].s0_gen_ox, params[0].s0_gen_oy, params[0].s0_gen_lod,
       params[0].s0_gen_fw, params[0].s0_gen_fh, params[0].s0_angle);
   ```
   (blur's step reads `work_0`; the final step's codec encode lands the result.)

> **RULE GEN6.** Nothing about generators is special-cased in `emit.rs` /
> `compile.rs` / the materializer. A zero-read kernel step already emits
> correctly (`read_args` empty → `kernel(idx, out, params...)`). If you find
> yourself editing the emitter for generators, stop — the design is wrong.

---

## 6. CPU lowering — native, tiling-correct

### 6.1 Why not `vips_gaussnoise` / the vips create family
libvips' `gaussnoise`/`perlin` use sequential or block RNGs that are **not**
reconstructable across tiles, and several create ops materialize the full canvas.
Using them would violate goals 3–4. So the canonical CPU path is **native
evaluation** in Rust (`render_cpu`) using the §3.2 math and the §7 RNG mirror.

### 6.2 `render_cpu` contract
- Allocate `w * h * bytes_per_pixel(layout)` bytes.
- For each `local = (lx, ly)` in `0..w × 0..h`, compute `lod_coord/full_px/uv`
  per §3.2 (with `gen_ox=region.x`, etc.), evaluate the generator function to a
  working `[f32; 4]`, then **encode** to the layout's storage (`U8/U16/F16/F32`,
  channel count from `layout`). Reuse `crate::pixel`/`crate::color` encoders so
  CPU encoding matches the Slang codec.
- Parallelize rows with `rayon` (already a dep). The function is embarrassingly
  parallel — no shared RNG state (counter-based, §7).

### 6.3 Optional vips-create shortcut (NON-noise, NON-tiling-sensitive only)
For generators where libvips has an exact, cheap, region-honoring equivalent
(e.g. a constant `black`/`xyz`), you MAY override a default method
`fn vips_shortcut(&self, region) -> Option<VipsHandle>` returning a built create
op, used in place of `render_cpu`. v1 does NOT use this; `render_cpu` is the
single source of truth. Reserved so a later optimization is additive.

---

## 7. Counter-based RNG (`shaders/lib/rng.slang` + `src/generator_rng.rs`)

Deterministic, stateless, coordinate-keyed — the property libvips can't give.

### 7.1 Algorithm: PCG2D (hash) → uniforms
```text
// PCG2D: maps a 2D integer key to two well-mixed uint32s (Jarzynski & Olano 2020)
uint2 pcg2d(uint2 v):
    v = v * 1664525u + 1013904223u
    v.x += v.y * 1664525u;  v.y += v.x * 1664525u
    v ^= v >> 16
    v.x += v.y * 1664525u;  v.y += v.x * 1664525u
    v ^= v >> 16
    return v
```

Helpers (Slang + Rust, identical):
```text
key(lod_coord, lod, seed)  = pcg2d(uint2(lod_coord) ^ uint2(seed, seed*747796405u + lod*2891336453u))
rand_uint2(...)            = pcg2d(key(...))
rand_f32(...)              = (rand_uint2().x >> 8) * (1.0 / 16777216.0)   // [0,1)
rand_f32_2(...)            = float2 from .x and .y the same way
gauss(...)                 = Box–Muller on two rand_f32 (for gaussnoise)
```

> **RULE RNG1.** The RNG is keyed on `lod_coord` (the integer LOD-space grid),
> mixed with `lod` and the generator's `seed`. This makes noise identical for a
> pixel regardless of which tile produced it (tiling-correct), deterministic
> across runs (seedable), and decorrelated per LOD. **Never** advance a running
> counter per thread or per row — there is no sequential state.

> **RULE RNG2.** `shaders/lib/rng.slang` is a new CORE-eligible module imported
> by generator kernels via the normal `import ops.generators;` chain (which
> imports `lib.rng`). The Rust mirror lives in `src/generator_rng.rs` and is the
> ONLY Rust RNG generators use. The two implementations are byte-for-byte the
> same integer math (test in §13.4).

---

## 8. Generator catalog (v1)

Each row: the pure function (working `float4` RGBA in scene-linear unless noted),
params, and which coordinate basis it uses. All clamp/encode via the output
layout. `seed: u32` defaults to 0 where present.

| Generator | Function (per pixel) | Params | Basis |
|---|---|---|---|
| `Constant` | `color` everywhere | `color:[f32;4]` | none |
| `LinearGradient` | `lerp(c0, c1, dot(uv, dir))` clamped | `c0,c1:[f32;4]`, `angle:f32` | `uv` |
| `RadialGradient` | `lerp(c0, c1, len(uv-center)/radius)` | `c0,c1`, `center:[f32;2]`, `radius:f32` | `uv` |
| `Xyz` | `(full_px.x, full_px.y, 0, 1)` (coord ramp) | — | `full_px` |
| `Zone` | `0.5 + 0.5*cos(k*(fx²+fy²))` (zone plate) | `k:f32` | `full_px` centered |
| `Sines` | `0.5+0.5*sin(2π*(fx*hf + fy*vf))` | `hfreq,vfreq:f32` | `uv` |
| `GaussNoise` | `mean + sigma*gauss(key)` per channel | `mean,sigma:f32`, `seed:u32` | `lod_coord`+RNG |
| `Perlin` | classic gradient noise over `uv*freq` | `freq:f32`, `seed:u32` | `uv`+RNG |
| `Worley` | cellular F1 distance over `uv*freq` | `freq:f32`, `seed:u32` | `uv`+RNG |

> **RULE CAT1.** Ship `Constant`, `LinearGradient`, `Xyz`, `GaussNoise` in the
> first milestone (covers fill, gradient, ramp, and the RNG path). The rest are
> additive copies of the same pattern.

---

## 9. Slang kernel authoring contract (`shaders/ops/generators.slang`)

### 9.1 Module skeleton
```hlsl
import lib.region;   // RWRegion, BufferRegion
import lib.rng;      // pcg2d + helpers (§7)
import lib.coord;    // gen_coord helper (§9.2)
```

### 9.2 The coordinate helper (`shaders/lib/coord.slang`, new)
```hlsl
struct GenCoord {
    int2  lod_coord;   // absolute LOD-space integer coord
    float2 full_px;    // full-res pixel center
    float2 uv;         // normalized [0,1)
    int   lod;
};
// Build from idx + the 5 canonical params the blanket lower always passes.
GenCoord gen_coord(uint2 idx, int ox, int oy, int lod, int fw, int fh) {
    GenCoord c;
    c.lod_coord = int2(ox, oy) + int2(idx);
    int scale   = 1 << lod;
    c.full_px   = (float2(c.lod_coord) + 0.5) * float(scale);
    c.uv        = c.full_px / float2(fw, fh);
    c.lod       = lod;
    return c;
}
```

### 9.3 Kernel signature convention (MANDATORY)
Every generator kernel:
```hlsl
public void <name>_kernel(
    uint2 idx, RWRegion output,            // emitter passes idx then out_var
    int gen_ox, int gen_oy, int gen_lod,   // the 5 canonical params, in order
    int gen_fw, int gen_fh,
    /* ...this generator's own params, in cx.param push order... */)
{
    GenCoord c = gen_coord(idx, gen_ox, gen_oy, gen_lod, gen_fw, gen_fh);
    float4 value = /* pure function of c (+ rng using c.lod_coord) */;
    output.write(idx, value);              // write at LOCAL idx
}
```

> **RULE K1.** Arg order is fixed: `idx, output, gen_ox, gen_oy, gen_lod,
> gen_fw, gen_fh, <own params…>`. It must match the `cx.param` push order in
> §3.1 + §4.2 exactly (the emitter passes params positionally). Get this wrong
> and you read `angle` as `gen_fw` — silent garbage, no compile error.

### 9.4 Worked kernel — `gradient_kernel`
```hlsl
public void gradient_kernel(
    uint2 idx, RWRegion output,
    int gen_ox, int gen_oy, int gen_lod, int gen_fw, int gen_fh,
    float4 c0, float4 c1, float angle)
{
    GenCoord c = gen_coord(idx, gen_ox, gen_oy, gen_lod, gen_fw, gen_fh);
    float2 dir = float2(cos(angle), sin(angle));
    float  t   = clamp(dot(c.uv, dir), 0.0, 1.0);
    output.write(idx, lerp(c0, c1, t));
}
```
Matching `Gradient::gpu_params`: `cx.param4("c0", self.c0); cx.param4("c1", self.c1); cx.param("angle", self.angle);` — see §9.5 for `param4`.

### 9.5 Vector params
`GpuBuilder::param` takes a `SlangScalar`. For `[f32;4]`/`[f32;2]` add thin
helpers `param4`/`param2` on `GpuBuilder` (push as Slang `float4`/`float2`, 16/8
bytes, std430-aligned) OR push components as four `param` floats and declare the
kernel arg as `float4` (std430 packs 4 consecutive floats as a float4 only if
aligned — **prefer explicit `param4`** to avoid alignment surprises).

> **RULE K2.** Add `GpuBuilder::param4(&mut self, name, [f32;4])` and `param2`.
> They mirror `param` but with the correct Slang type + 16/8-byte payload. This
> is the only `GpuBuilder` addition generators need.

### 9.6 `gaussnoise_kernel` (RNG path)
```hlsl
public void gaussnoise_kernel(
    uint2 idx, RWRegion output,
    int gen_ox, int gen_oy, int gen_lod, int gen_fw, int gen_fh,
    float mean, float sigma, uint seed)
{
    GenCoord c = gen_coord(idx, gen_ox, gen_oy, gen_lod, gen_fw, gen_fh);
    float3 g = float3(
        gauss(c.lod_coord, c.lod, seed + 0u),
        gauss(c.lod_coord, c.lod, seed + 1u),
        gauss(c.lod_coord, c.lod, seed + 2u));
    output.write(idx, float4(mean + sigma * g, 1.0));
}
```

---

## 10. Ergonomic constructors (`src/data/generator.rs`)

Generic over backend via the `GenSource<G>: Source<B>` bound (both blanket impls
satisfy it). One free function per generator; optionally also inherent methods.

```rust
use crate::node::Data;
use crate::data::image::Image2D;
use crate::pixel::PixelLayout;

/// Build a generator pipeline tip on backend `B`.
fn gen<B: Backend, G: Generator>(g: G, ctx: Arc<B::Ctx>) -> Image2D<B>
where GenSource<G>: Source<B, Kind = ImageKind> {
    Data::from_source(Arc::new(GenSource(g)), ctx)
}

impl<B: Backend> Image2D<B> {
    pub fn gradient(ctx: Arc<B::Ctx>, w: i32, h: i32, layout: PixelLayout,
                    c0: [f32;4], c1: [f32;4], angle: f32) -> Self
    where GenSource<Gradient>: Source<B, Kind = ImageKind> {
        gen(Gradient { w, h, layout, c0, c1, angle }, ctx)
    }

    pub fn gauss_noise(ctx: Arc<B::Ctx>, w: i32, h: i32, layout: PixelLayout,
                       mean: f32, sigma: f32, seed: u32) -> Self
    where GenSource<GaussNoise>: Source<B, Kind = ImageKind> {
        gen(GaussNoise { w, h, layout, mean, sigma, seed }, ctx)
    }
    // ... constant, xyz, zone, sines, perlin, worley ...
}
```

Usage:
```rust
// GPU: a noise layer added to a photo, blurred — all one fused pass family.
let noise = Image2D::<GpuBackend>::gauss_noise(ctx.clone(), w, h, layout, 0.0, 0.05, 42);
let grainy = photo.add(&noise).blur(2.0);

// CPU parity (same bytes, tiling-correct):
let noise_cpu = Image2D::<VipsBackend>::gauss_noise((), w, h, layout, 0.0, 0.05, 42);
```

---

## 11. File map (everything to add)

```
src/
  data/generator.rs        Generator trait, GenSource<G>, blanket Source impls,
                           concrete generators (Constant/Gradient/Xyz/GaussNoise/…),
                           ergonomic constructors.            (mostly agnostic + per-backend lower)
  generator_rng.rs         Rust PCG2D mirror + uniforms/gauss (matches lib/rng.slang).
  backend/vips/mod.rs      + image_from_memory(bytes,w,h,layout) guarded helper (RULE GEN4).
  backend/gpu/mod.rs       + GpuBuilder::param4 / param2 (RULE K2).
  data/mod.rs / lib.rs     register `pub mod generator;` (+ generator_rng).
shaders/
  lib/coord.slang          gen_coord helper (§9.2).
  lib/rng.slang            pcg2d + uniforms + gauss (§7).
  ops/generators.slang     one *_kernel per generator (§9).
tests/
  gpu/generators.rs        tiling-correctness, LOD, fusion, GPU≈CPU, determinism (§13).
```

No edits to `emit.rs`, `compile.rs`, `node.rs`, `work_unit.rs`, or any
`Operation`. (RULE GEN6.)

---

## 12. Recipe — add a new generator (copy this)

1. **Slang kernel** in `shaders/ops/generators.slang`: `public void
   foo_kernel(uint2 idx, RWRegion output, int gen_ox, …, <own params>)` using
   `gen_coord` (§9.3). Add `import lib.rng;` if it samples noise.
2. **Struct** in `src/data/generator.rs`: `pub struct Foo { pub w: i32, pub h:
   i32, pub layout: PixelLayout, /* own params */ }`.
3. **`impl Generator for Foo`**:
   - `spec` → `Arc::new(ImageKind::new(self.layout, self.w, self.h))`.
   - `gpu_kernel` → `("ops.generators", "foo_kernel")`.
   - `gpu_params` → `cx.param/param4(...)` for own params, **in kernel arg
     order**.
   - `render_cpu` → the SAME function in Rust over the region (§6.2), encoded to
     `self.layout`.
   - `dyn_hash` → hash dims, layout (Debug proxy), every own param's bits.
4. **Constructor** on `Image2D<B>` (§10).
5. **Test** in `tests/gpu/generators.rs` (§13).

That's it — both backends, fusion, LOD, and tiling come from the blanket impls.

### 12.4 Reserved
- Frequency-domain mask builders (`mask_ideal`, `mask_butterworth`, …): a
  generator whose basis is frequency `(u, v)` rather than `uv` — same trait, a
  different coord helper. Add when FFT ops land.
- Non-image generators (`PointListKind`): generalize `Generator::spec` to an
  associated `Kind` + `WorkUnit`; the blanket impls already isolate the
  image-specific `RegionParams`/codec so this is additive.

---

## 13. Tests (`tests/gpu/generators.rs`) — what to assert

Use the `tests/common` harness (`gpu_ctx`, `poc_materialize`). Serialize is no
longer needed (vips auto-inits, see the vips-init fix).

1. **Tiling-correctness (the headline).** Pull a generator over the full region;
   separately pull two halves `A` and `B` and stitch them. Assert the stitched
   bytes equal the full pull **exactly** (integer RNG ⇒ bit-exact for noise;
   gradients exact too). This is the property libvips can't provide.
2. **LOD nativeness.** Pull at `Lod(0)` full, and at `Lod(1)`; assert the
   `Lod(1)` result equals sampling the `Lod(0)` function at the §3.2 `Lod(1)`
   grid (compute expected in the test). Assert the `Lod(1)` pull touched only a
   quarter-area buffer (no full-canvas materialization — check dispatch dims via
   a small instrumentation hook or just the output size).
3. **Fusion.** `gradient(...).invert()` and `gauss_noise(...).blur(1.0)` pull
   correctly and (optional) assert the pipeline cache compiled **one** shader
   for the fused family (generator step + op step in one pass).
4. **GPU ≈ CPU parity.** Same generator + params on `GpuBackend` and
   `VipsBackend`; RMS within rounding (`< 1.0` on u8). For `GaussNoise`, assert
   bit-exact (both use the integer PCG mirror, §7) modulo storage rounding.
5. **Determinism / seed.** Same seed ⇒ identical; different seed ⇒ different.

---

## 14. Hard rules recap

- **GEN0/GEN1** generator `lower` (GPU) = `cx.kernel` (zero reads) + always
  `cx.output`; never `cx.input`, never upload.
- **GEN2/RNG1** all spatial values from `lod_coord/full_px/uv`; RNG keyed on
  `lod_coord` (+lod+seed). Never use raw `idx` as a world coord; never a
  sequential counter.
- **GEN3** GPU kernel math == `render_cpu` math == one documented formula.
- **GEN4** vips image creation only via `image_from_memory` (guarded init).
- **GEN6** zero edits to the emitter/compiler/materializer — a zero-read kernel
  step already lowers correctly.
- **K1/K2** kernel arg order = canonical 5 params then own params, matching
  `cx.param` push order; add `param4`/`param2` for vector params.
- Engine invariants in `CLAUDE.md` are untouched: a generator is just a
  `Source` that lowers to a kernel.
```
