# GPU Operations — Implementation Guide

This document is the worklist for bringing the GPU backend (`src/backend/gpu`)
up to parity with the Vips (CPU) backend. It assumes the Tier1 correctness
fixes (output_spec/demand bugs, `set_array_double` fixes, kernel-name typos)
and the Tier2 `emit.rs` import wiring are already done (see git history:
"refactor(engine): split data modules..." and surrounding commits).

Every operation lives in `src/operation/<file>.rs` as `pub struct Foo<B: Backend>`,
with:
- `impl<B: Backend> Operation<B> for Foo<B> where Foo<B>: Lower<B>` — generic,
  defines `inputs()`, `demand()`, `output_spec()`, `dyn_hash()`.
- `impl Lower<VipsBackend> for Foo<VipsBackend>` — CPU reference (usually done).
- `impl Lower<GpuBackend> for Foo<GpuBackend>` — **this is what's missing** for
  most ops below.

"GPU MISSING" = no `Lower<GpuBackend>` impl exists at all. "Dangling kernel" =
a `.slang` kernel exists in `shaders/ops/*.slang` but nothing in Rust calls
`cx.kernel("...")` with that name (and/or it isn't imported in `emit.rs`).

---

## 1. The Lowering Recipe

### 1.1 Minimal single-kernel op

This is the pattern used by `Exposure`, `Gamma`, `Invert`, `Composite2`, etc.
(see `src/operation/misc.rs`, `composite.rs`):

```rust
impl Lower<GpuBackend> for Foo<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new()
            .param("my_scalar", "float", self.my_field as f32)
            .param("my_flag", "uint", self.my_flag as u32)
        );
        cx.kernel("foo_kernel");
        cx.output(self.output_spec().output());
    }
}
```

- `param_block()` registers scalar fields onto the shared `ChainParams` SSBO
  and queues them as the *next* `kernel()` call's trailing args, in
  declaration order. Field names get namespaced (`s{step}_{name}`) so two
  steps never collide — you don't need to worry about uniqueness.
- `kernel("foo_kernel")` must match a `public void foo_kernel<R: IRegion, ...>(uint2 idx, ..., RWRegion output, <your params in order>)`
  entry point in some `shaders/ops/*.slang` file.
- `cx.output(self.output_spec().output())` registers the output wrapper —
  `output_spec()` must be **correct** (see §1.4); the emitted shader's
  `region_out` and target buffer size are derived from it.
- **Do not** reference `wgpu` or do any device work here — this only builds
  a `GpuBuilder` (a description of one fused shader). Real dispatch happens
  later in `materialize.rs`.

### 1.2 Multi-input ops (binary/ternary kernels)

No special wiring needed beyond `inputs()`. `GpuBuilder::enter()` resolves
each `inputs()` entry (in order) into `cur_inputs: Vec<StepInput>`, and the
first `kernel()` call of the node consumes them positionally — so a kernel
declared as

```slang
public void compose_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 bg, R2 fg, RWRegion output, uint mode, int ox, int oy)
```

just works if `inputs()` returns `vec![&self.base, &self.overlay]` in that
order. Confirmed precedent: `Add::inputs() -> [left, right]` →
`add_kernel<R1,R2>`, `Convolution::inputs() -> [input, mask]` →
`convolution_kernel<R1,R2>`, `Composite2::inputs() -> [base, overlay]` →
`compose_kernel<R1,R2>`. **Ops with a LUT/matrix/mask second input
(`Maplut`, `Recomb`, `Case`, `Ifthenelse`) follow this exact pattern** — the
second (or third) image is just another entry in `inputs()`.

### 1.3 Zero-cost "free view" aliasing (`cx.alias()`)

`ExtractBand` (single-band case) calls `cx.alias(channel)` instead of
`cx.kernel(...)`. This adds **no kernel step** — it marks the current node as
a pure `SwizzleView` of its input, which a downstream consumer (or the final
codec encode) reads through directly. See `src/backend/gpu/mod.rs`
(`GpuBuilder::alias`, `StepInput::{SwizzleSource,SwizzleStep}`) and
`shaders/lib/region.slang` (`SwizzleView<R: IRegion>`). §2 proposes a
coordinate-remap analog (`RemapView`) for geometry ops.

### 1.4 `demand()` / `output_spec()` MUST be correct before you write the kernel

The Tier1 audit found multiple GPU ops with wrong `output_spec()` (wrong
width/height) or wrong `demand()` (wrong source rect requested). **These bugs
are silent** — the kernel runs, produces *some* output, but it's the wrong
shape or sampled from the wrong region. Before adding `Lower<GpuBackend>`:

- `output_spec()` must return the **exact** output `ImageKind` (width,
  height, `PixelFormat` via `spec.with_band_count(n)`) — this sizes the
  target buffer and `region_out`.
- `demand()` must return the **exact** source rect(s) needed to produce
  `out` (in input-image coordinates, accounting for offsets, scale factors,
  and filter halos). If `demand()` under-requests, the kernel reads
  `read_clamped` garbage at the edges; if it over-requests, you waste
  bandwidth but stay correct.
- For geometry ops, `demand()` and `output_spec()` for `Crop`, `Embed`,
  `Flip`, `Rot90`, `Gravity`, `Resize`, `Reduce*`, `Shrink*`, `Subsample`,
  `Zoom`, `ExtractArea`, `Replicate` are **already fixed** (geometry.rs Tier1
  pass) — read the corrected formulas there before writing GPU kernels for
  these, since the kernel's coordinate math must match `demand()` exactly.

### 1.5 `emit.rs` import checklist

`emit_slang()` (`src/backend/gpu/emit.rs`) prepends `import ops.X;` for a
fixed list of modules (POC simplification — a production emitter would carry
the module path on `KernelCall`, but for now every module a kernel might come
from must be in this list). **If you add a kernel in a new `.slang` file (or
a file not yet imported), add `s.push_str("import ops.<file>;\n");`** to the
import block near the top of `emit_slang`, or you'll get a Slang "undefined
symbol" compile error at first use.

Currently imported: `lib.region`, `lib.io`, `lib.codecs`, `lib.pixel`,
`ops.invert`, `ops.gaussian_blur`, `ops.histogram`, `ops.arithmetic`,
`ops.bands`, `ops.composite`, `ops.convolution`, `ops.exposure`, `ops.gamma`,
`ops.passthrough`.

---

## 2. New primitive: `RemapView<R: IRegion>` (geometry "free views")

### Motivation

`SwizzleView` (§1.3) generalizes "read one channel of `inner`, broadcast it."
Many geometry ops are the **coordinate-space dual**: "read `inner` at a
*different* `idx` than the one being written, with no per-pixel math
otherwise." If `RWRegion.read`/`.write` and `CodecRegion.read` all take
`uint2 idx` / `int2 idx` and the *only* thing a geometry op changes is which
`idx` `inner` is sampled at, then **Flip, Rot90 (D0/D180), Replicate,
Subsample, Zoom (nearest), and `ExtractArea`/offset-`Crop` are all candidates
for a zero-kernel-step "free view"** — exactly like `ExtractBand` today.

### Proposed interface (`shaders/lib/region.slang`)

```slang
// Coordinate-remap view: wraps ANY IRegion, transforms the read index before
// delegating. `out_height`/`out_width` are the OUTPUT region's dims (needed
// because some transforms are dimension-dependent, e.g. flip).
enum RemapKind : uint {
    Identity   = 0u,
    FlipH      = 1u,   // x' = out_w - 1 - x
    FlipV      = 2u,   // y' = out_h - 1 - y
    Rot180     = 3u,   // x' = out_w-1-x, y' = out_h-1-y
    Scale      = 4u,   // x' = x * sx,  y' = y * sy   (Subsample / Zoom-nearest)
    Tile       = 5u,   // x' = x % in_w, y' = y % in_h (Replicate)
};

struct RemapView<R: IRegion> : IRegion {
    R inner;
    uint kind;
    uint out_w, out_h;   // output region dims (for Flip/Rot180)
    float sx, sy;        // scale factors (for Scale)
    uint in_w, in_h;     // inner dims (for Tile)

    int2 remap(int2 idx) {
        switch (kind) {
            case 1u /*FlipH*/: return int2(int(out_w) - 1 - idx.x, idx.y);
            case 2u /*FlipV*/: return int2(idx.x, int(out_h) - 1 - idx.y);
            case 3u /*Rot180*/: return int2(int(out_w)-1-idx.x, int(out_h)-1-idx.y);
            case 4u /*Scale*/: return int2(int(float(idx.x)*sx), int(float(idx.y)*sy));
            case 5u /*Tile*/: return int2(idx.x % int(in_w), idx.y % int(in_h));
            default: return idx;
        }
    }

    float4 read(uint2 idx) {
        int2 r = remap(int2(idx));
        return inner.read_clamped(r);
    }
    float4 read_clamped(int2 idx) {
        return inner.read_clamped(remap(idx));
    }
}
```

Note `read()` delegates to `inner.read_clamped` (not `inner.read`) because
the remapped coordinate is in the *inner* region's coordinate space, whose
bounds differ from the output's — clamping is the correct "free view"
behavior for in-bounds remaps (Flip/Rot180/Tile always stay in-bounds by
construction; Scale can round to `in_w`/`in_h` at the last pixel, which
`read_clamped` handles).

### `GpuBuilder` API addition

Mirror `alias()`:

```rust
/// Make the current node a zero-cost coordinate-remap view of its single
/// input. Like `alias()`, but transforms `idx` instead of swizzling channels.
pub fn remap(&mut self, kind: RemapKind, params: RemapParams) -> &mut Self {
    let Some(&input) = self.cur_inputs.first() else { /* error */ };
    if let Some(k) = self.cur_node {
        self.remap_views.insert(k, (input, kind, params));
    }
    self
}
```

`emit_slang` needs a new `StepInput`-adjacent case (`RemapSource`/`RemapStep`,
parallel to `SwizzleSource`/`SwizzleStep`) that emits `RemapView<...> r = {
in_i, kind, out_w, out_h, sx, sy, in_w, in_h };` and reads through it. This is
the same plumbing as the swizzle case in `emit.rs` — copy the
`SwizzleSource`/`SwizzleStep` arms and adjust the constructed type and
constant fields. `RemapParams` is just the 6 scalar fields above, written
into the per-node alias table the same way `alias_swizzles` is today.

### Which ops use this (Group C, §5)

`Flip` (FlipH/FlipV), `Rot90` D0/D180 (D90/D270 need a *transpose*, i.e.
`(x,y) -> (y,x)` plus a flip — `RemapKind` can grow a `Transpose` /
`TransposeFlip` variant the same way), `ExtractArea`/offset-only `Crop`
(`Identity` + an `(x,y)` translate — actually simplest as a 7th `RemapKind`,
`Translate`, with `tx,ty` in `RemapParams`; `Crop`'s current
`passthrough_kernel` could become `remap(Translate)` too, though leaving it
as-is is also fine since it already works), `Subsample` (`Scale`, `sx =
horizontal`, `sy = vertical`), `Zoom` nearest-neighbor (`Scale`, `sx = 1/horizontal`),
`Replicate` (`Tile`).

**Do not** use `RemapView` for ops that need *interpolation* (bilinear
`Resize`/`Reduce`, `Rotate`, `Rot45` D45/...) — those need a real kernel that
reads multiple neighboring `inner` pixels and blends (Group E, §5).

---

## 3. Tier 2 — Dangling Kernels

These `.slang` kernels exist and compile but have **zero callers**. For each:
either wire up the (existing or new) op that should call it, or — if it's
genuinely obsolete — delete it.

| Kernel | File | Status / Action |
|---|---|---|
| `shrink_kernel<R>(idx, input, output, h_factor, v_factor)` | `shrink.slang` | Box-filter average. Wire to `ShrinkHorizontal`/`ShrinkVertical`/`Shrink` (Group A, §5.1). Also a reasonable (non-Lanczos) fallback `kernel: Block` mode for `Reduce`/`ReduceHorizontal`/`ReduceVertical`. Add `import ops.shrink;` to `emit.rs`. |
| `opacity_kernel<R>(idx, input, output, amount)` | `opacity.slang` | Wire to `Opacity<GpuBackend>` (Group A, §5.1). `Opacity`'s `output_spec`/vips lower already fixed in Tier1 — only `Lower<GpuBackend>` is missing. Add `import ops.opacity;`. |
| `saturation_kernel<R>(idx, input, output, amount)` | `saturation.slang` | **No `Saturation<B>` op struct exists yet.** Create one (Group A, §5.1) — straightforward, same shape as `Brightness`/`Exposure`. Add `import ops.saturation;`. |
| `brightness_kernel<R>(idx, input, output, factor)` | `brightness.slang` | **Dead.** `Brightness<GpuBackend>` already lowers via `exposure_kernel` (gain=`value`, preserve=0) — works, if semantically odd (multiplicative gain named "brightness"). Recommend **delete `brightness.slang`** (or leave as documented-but-unused; do not import it). |
| `color_convert_kernel<R>(idx, input, output, ColorConvertParams p)` | `color.slang` | Blocked: `ColorConvertParams` is a struct param, and the POC's `ParamBlock`/`ChainParams` only supports flat scalars (`bytemuck::Pod` + `"scalar"`/named-type strings, all emitted as top-level `ChainParams` fields — see `ParamBlock::param`). Flatten `ColorConvertParams`'s fields into individual scalar `param()` calls (matrix as 9× `float`, etc.) if/when a color-management op needs this; until then, leave unimported. Not in scope for this pass. |
| `histogram_kernel<G>(tid, input, HistogramOut output, channel)` | `histogram.slang` | Non-image output (`HistogramOut`, not `RWRegion`). See Group G (§5.7) — architecture gap, needs `GpuBuilder`/`OutputWrap` support for non-tile-shaped accumulator outputs before this can be wired. |
| `vectorscope_kernel<G>(tid, input, HistogramOut output, grid_size)` | `vectorscope.slang` | Same as above — Group G. |

---

## 4. Architectural flag: `Bandfold`/`Bandunfold` kernel/Rust mismatch

`Bandfold<B>::output_spec`/`demand` were corrected in Tier1 to the *width*-axis
semantics matching libvips (`bandfold`: width /= factor, bands *= factor —
folds `factor` adjacent **pixels** into one pixel with `factor`× the bands).
The existing `bandfold_kernel`/`bandunfold_kernel`
(`shaders/ops/bands.slang:141,155`) instead shuffle bands **within a single
`float4`** along what was the *height* axis — i.e. they implement a different
operation than the (now-correct) Rust `output_spec`/`demand`.

These kernels cannot be patched in place: `float4` has only 4 lanes, so
`factor > 1` on an already-multi-band input produces `bands * factor > 4`
output bands, which doesn't fit one working temp. Folding `factor` adjacent
*pixels'* worth of `float4` into one output pixel requires reading `factor`
separate `inner.read(idx_x*factor + k, idx_y)` calls and packing their
channels — fundamentally a multi-source-pixel kernel, not a per-pixel
remap/swizzle.

**Recommendation:** leave `Bandfold`/`Bandunfold` GPU-unimplemented
(`Lower<GpuBackend>` absent) for this pass. If/when needed, write new
`bandfold_kernel`/`bandunfold_kernel` that:
- `Bandfold`: for `factor ∈ {2,3,4}` and 1-band input (the common case, e.g.
  packing 4 grayscale columns into RGBA), read `inner.read(idx.x*factor+k,
  idx.y).r` for `k in 0..factor` and write into output channel `k`. For
  multi-band inputs with `factor*bands <= 4`, generalize the channel mapping.
  `factor*bands > 4` is out of scope (no `float4`-based representation).
- `Bandunfold`: the inverse — `inner.read(idx.x/factor, idx.y)[idx.x %
  factor]` (for the 1-band-output case).

---

## 5. Tier 3 — Missing GPU Operations, by Group

### 5.1 Group A — Direct kernel reuse (easiest)

#### `Opacity<GpuBackend>` (`src/operation/opacity.rs`)

```rust
impl Lower<GpuBackend> for Opacity<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::scalar("amount", "float", self.amount as f32));
        cx.kernel("opacity_kernel");
        cx.output(self.output_spec().output());
    }
}
```
- `output_spec()` already fixed (Tier1): adds an alpha band if input has none.
- Add `import ops.opacity;` to `emit.rs`.
- `opacity_kernel` does `c.a *= amount` — matches the Tier1-fixed vips lower
  (which does the analogous `linear` on the alpha band).

#### `Saturation<GpuBackend>` (new op — `src/operation/misc.rs` or new `color_adjust.rs`)

No Rust struct exists. Create:

```rust
pub struct Saturation<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub amount: f32,   // 0 = grayscale, 1 = unchanged, >1 = boosted
}

impl<B: Backend> Operation<B> for Saturation<B> where Saturation<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) { state.write_u32(self.amount.to_bits()); }
}
```

- **CPU (`Lower<VipsBackend>`)**: vips has no direct "saturation" op on RGB;
  the standard approach is `colourspace` to LCh/HSV, scale the
  chroma/saturation channel by `amount`, convert back — OR (cheaper, matches
  the GPU kernel's luma-lerp) use `recomb`/`linear` to compute
  `out = luma + (in - luma) * amount` per-channel via vips arithmetic
  (`luma = bandmean`-like weighted sum, broadcast, `lerp` via
  `add`/`multiply`/`subtract`). Implement via a small chain of existing vips
  ops (no new FFI needed) so CPU/GPU match the same formula as
  `saturation.slang`'s `_luma()` (BT.709 weights 0.2126/0.7152/0.0722).
- **GPU**:
```rust
impl Lower<GpuBackend> for Saturation<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::scalar("amount", "float", self.amount));
        cx.kernel("saturation_kernel");
        cx.output(self.output_spec().output());
    }
}
```
- Add `import ops.saturation;` to `emit.rs`.
- Add `Image2D<B>::saturation(amount: f32) -> Self` convenience method
  (follow the `bandmean`/`extract_band` pattern at the bottom of `bands.rs`).

#### `ShrinkHorizontal`/`ShrinkVertical`/`Shrink` via `shrink_kernel` (`src/operation/geometry.rs`)

`shrink_kernel<R>(idx, input, output, h_factor, v_factor)` is a box-filter
average over `h_factor × v_factor` input cells per output pixel — this is
*exactly* what `ShrinkHorizontal`/`ShrinkVertical`/`Shrink` need (their
`output_spec`/`demand` were fixed in Tier1 to the ceil-divide / `x*factor`
relationship that matches this kernel's addressing).

```rust
impl Lower<GpuBackend> for ShrinkHorizontal<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new()
            .param("h_factor", "uint", self.shrink as u32)
            .param("v_factor", "uint", 1u32)
        );
        cx.kernel("shrink_kernel");
        cx.output(self.output_spec().output());
    }
}
// ShrinkVertical: h_factor=1, v_factor=self.shrink
// Shrink: h_factor=self.horizontal.ceil() as u32, v_factor=self.vertical.ceil() as u32
//   (NOTE: Shrink's factors are f64 — shrink_kernel takes integer factors,
//   so this is an approximation for non-integer shrink. Vips' own `shrink`
//   handles fractional factors with partial-pixel weighting; shrink_kernel
//   does not. Acceptable for integer factors; flag fractional Shrink as
//   "approximate on GPU" in docs/comments if used.)
```
Add `import ops.shrink;` to `emit.rs`.

#### `Reduce`/`ReduceHorizontal`/`ReduceVertical` — partial reuse

`shrink_kernel` only supports *integer* factors and box averaging (no
Lanczos/cubic). `Reduce*`'s `horizontal`/`vertical` are `f64` and carry an
optional `Kernel` (resampling filter). Two options:
1. **Quick win**: when `self.kernel` is `None` or `Lanczos3`-ish and the
   factor happens to be integral, lower to `shrink_kernel` (same as `Shrink`
   above) — covers the common "reduce by exactly N" case.
2. **Correct general case**: needs a new bilinear/area-resample kernel (Group
   E, §5.5). Recommended: implement (1) now as a fast path, defer (2).

---

### 5.2 Group B — New simple per-pixel kernels (new `.slang` file, e.g. `shaders/ops/unary.slang`)

These all follow `round_kernel`'s shape exactly: read `c`, transform all (or
selected) channels, write back. No multi-input, no halo.

| Op | Struct | New kernel | Notes |
|---|---|---|---|
| `Abs<GpuBackend>` (`edge.rs`) | `{ input }` | `abs_kernel<R>(idx, input, output)` → `output.write(idx, abs(c))` | Trivial. |
| `Sign<GpuBackend>` (`edge.rs`) | `{ input }` | `sign_kernel<R>(idx, input, output)` → `output.write(idx, sign(c))` | Vips `sign` maps to `{-1,0,1}`; `sign()` in Slang matches. |
| `Msb<GpuBackend>` (`misc.rs`) | `{ input, band: Option<i32> }` | `msb_kernel<R>(idx, input, output, int band)` | Vips `msb` extracts the most-significant byte of each (integer) sample. The working space is float `[0,1]` — this only makes sense applied to the **decoded integer sample value**, i.e. operate on `c * 255.0` (for U8 source) → `floor(c*255)` is already the MSB for 8-bit. For 16-bit sources this needs the *codec's* raw integer, which the working-space sandwich has already normalized away. **Flag**: `Msb` may need to run *before* the ACEScg sandwich (i.e. a `WorkingSource`-bypassing kernel) to be meaningful for >8-bit formats — for 8-bit formats, `floor(c.rgb * 255.0)` suffices (it's already the single byte). Implement the 8-bit case now; document the 16-bit limitation. `band` selects single-channel output (`if band >= 0`, broadcast that channel; else operate per-channel). |
| `Invertlut<GpuBackend>` (`misc.rs`) | `{ input, size: Option<i32> }` | N/A — **not a per-pixel op**, it's a 1D-curve *inversion* (takes a LUT image, produces the inverse LUT image). This is fundamentally a different shape (small 1×N image → 1×N image, not a per-pixel map over the *photo*). Recommend: **Group G (CPU-only)** — leave `Lower<GpuBackend>` absent; LUT images are tiny (≤65536×1), CPU cost is negligible. |
| `Cast<GpuBackend>` (`misc.rs`) | `{ input, format: PixelFormat, shift: Option<bool> }` | No new kernel — **this is a codec/format change, not a pixel-math op.** | The GPU pipeline already re-encodes to the target `PixelFormat` via `output_spec().output()`'s `OutputWrap`/codec (see `GpuView::output()` — the encode codec is selected by the output `Kind`). `Cast<GpuBackend>::lower()` likely just needs: `cx.kernel("passthrough_kernel"); cx.output(self.output_spec().output())` where `output_spec()` returns `spec.format = self.format` (same width/height/bands... or different band count if cast changes channel layout — check `PixelFormat::channel_count` compatibility). `shift` (vips `cast` with bit-shift for differing bit depths) — if the codec's `encode`/`decode` already normalize to `[0,1]` float regardless of source bit depth, `shift` may be a no-op on GPU (the working-space sandwich already handles depth conversion). Verify against `lib/codecs.slang`'s U8/U16/F16/F32 codec `decode`/`encode` before assuming `passthrough_kernel` suffices — if band *count* changes (e.g. RGB→RGBA cast), need a `bandjoinN_kernel`-style step instead. |

---

### 5.3 Group C — Geometry via `RemapView` (§2)

Once `RemapView`/`GpuBuilder::remap()`/`emit.rs` support lands (§2):

| Op | `RemapKind` | Params | Notes |
|---|---|---|---|
| `Flip<GpuBackend>` (Horizontal) | `FlipH` | `out_w = output_spec().width` | `output_spec()` unchanged (same dims); `demand()` already mirrors the region (Tier1-fixed) — the *source rect fetched* is already the mirrored rect, so `RemapView` here must mirror **within that fetched rect**, i.e. `out_w`/`out_h` = the **fetched region's** `w`/`h` (same as output `w`/`h` since flip preserves dims). Use `params[0].region_out.width/height`. |
| `Flip<GpuBackend>` (Vertical) | `FlipV` | same | as above |
| `Rot90<GpuBackend>` (D180) | `Rot180` | `out_w/out_h = region_out.width/height` | `output_spec()` unchanged for D180. |
| `Rot90<GpuBackend>` (D0) | `Identity` (or just `passthrough_kernel`) | — | trivial |
| `Rot90<GpuBackend>` (D90/D270) | needs new `Transpose`/`TransposeFlip` `RemapKind` | swap `idx.x`/`idx.y` plus a flip on one axis depending on direction | `output_spec()` already swaps width/height (Tier1-fixed). Verify against vips' `rot` D90/D270 convention (clockwise vs counter-clockwise) for which axis flips. |
| `ExtractArea<GpuBackend>` | new `Translate` `RemapKind` (`tx=left, ty=top`) — or just reuse `Crop`'s `passthrough_kernel` approach (`demand()` already offsets by `left`/`top`, so the *fetched* region is already correctly positioned; `passthrough_kernel` with `Identity` mapping suffices, same as `Crop`) | — | **Simplest: copy `Crop<GpuBackend>`'s exact `lower()`** (`cx.kernel("passthrough_kernel"); cx.output(...)`) — `ExtractArea` and `Crop` have identical `demand()`/`output_spec()` shapes (both `output_spec` sets width/height from fields, both `demand` offsets by `left`/`top`). No `RemapView` needed at all for this one. |
| `Subsample<GpuBackend>` | `Scale` | `sx = horizontal, sy = vertical` (as floats) | `output_spec()`/`demand()` Tier1-fixed (ceil-div output size, `idx*factor` source mapping). `point: Option<bool>` — vips `subsample` with `point=true` is nearest-neighbor (matches `RemapView` exactly); `point=false`/default may imply averaging — check vips docs; if averaging is the default, `Subsample` may belong in Group E instead (needs `shrink_kernel`-style averaging over the `factor×factor` cell, not a single sample). **Verify vips semantics before choosing Remap vs Shrink-kernel.** |
| `Zoom<GpuBackend>` | `Scale` | `sx = 1.0/horizontal, sy = 1.0/vertical` | Nearest-neighbor zoom (pixel replication). `output_spec()`/`demand()` Tier1-fixed (`width *= horizontal`). This is the *enlarging* counterpart of `Subsample` — same `RemapView::Scale` with reciprocal factors. |
| `Replicate<GpuBackend>` | `Tile` | `in_w/in_h = input.spec.width/height` | `output_spec()` = `width *= across, height *= down` (Tier1-fixed). `demand()` (Tier1-fixed) wraps mod input dims with a full-input fallback when the region spans a tile boundary — when the fallback fires (full input fetched), `RemapView::Tile`'s `idx % in_w` still produces the correct per-output-pixel source coordinate against the full-input buffer, so this works in both `demand()` cases. |

---

### 5.4 Group D — Embed / Gravity (extend modes)

`Embed`/`Gravity` (`output_spec`/`demand` Tier1-fixed: output is the new
canvas size, `demand()` offsets the input region by the placement offset) are
**not** pure remaps when `out` (the WorkUnit's `Region`, in *output*
coordinates) extends beyond where the input was placed — those pixels must
come from `extend: Option<Extend>` (`Black`/`White`/`Copy`/`Repeat`/`Mirror`/
`Background`), not from the input buffer at all.

Recommended approach: a dedicated `embed_kernel<R>(idx, input, output, ox,
oy, in_w, in_h, extend_mode, bg_r, bg_g, bg_b)`:

```slang
public void embed_kernel<R: IRegion>(uint2 idx, R input, RWRegion output,
    int ox, int oy, uint in_w, uint in_h, uint extend_mode,
    float bg_r, float bg_g, float bg_b)
{
    int2 src = int2(idx) - int2(ox, oy);
    bool inside = src.x >= 0 && src.y >= 0 && src.x < int(in_w) && src.y < int(in_h);
    float4 c;
    if (inside) {
        c = input.read(uint2(src));
    } else {
        switch (extend_mode) {
            case 0u /*Black*/:      c = float4(0,0,0,1); break;
            case 1u /*White*/:      c = float4(1,1,1,1); break;
            case 2u /*Background*/: c = float4(bg_r,bg_g,bg_b,1); break;
            case 3u /*Copy*/:       c = input.read_clamped(src); break; // edge-extend
            // Repeat/Mirror: need modular/reflected addressing into `input`
            default: c = float4(0,0,0,1); break;
        }
    }
    output.write(idx, c);
}
```

- `demand()` for `Embed`/`Gravity` currently requests `out` offset by
  `(x,y)`/`(ox,oy)` — i.e. it may request **negative or out-of-bounds**
  source coordinates when the output canvas is larger than the input. Check
  how `materialize_batch`/region fetch handles a `demand()` rect that's
  partially outside `[0, in_w) × [0, in_h)` — it may already clamp, in which
  case `input` as bound in the shader covers only the in-bounds portion and
  `ox`/`oy` passed to `embed_kernel` must account for **where that fetched
  rect sits** relative to `idx`, not the raw `self.x`/`self.y`. This needs
  careful coordinate-frame bookkeeping between `demand()`'s returned rect and
  `params[0].region_in_0`/`region_out` — read `RegionParams::push_into` and
  how `region_in_0` vs `region_out` offsets compose in `emit_slang` before
  implementing.
- `Repeat`/`Mirror` extend modes need modular/reflected `input.read` —
  doable but adds kernel complexity; consider deferring those two modes
  (fall back to `Black`) in a first pass and documenting the gap.
- Add `import ops.embed;` (new file `shaders/ops/embed.slang`) to `emit.rs`.

---

### 5.5 Group E — Resampling kernels (Resize, Reduce, Thumbnail, Rotate, Rot45)

These need **interpolation** (read multiple neighboring source pixels, blend
by fractional position) — not expressible as `RemapView` or `shrink_kernel`.

#### `Resize<GpuBackend>` / general-factor `Reduce*<GpuBackend>`

New `resize_kernel<R>(idx, input, output, float inv_hscale, float inv_vscale)`:
bilinear sample at `(idx.x + 0.5) * inv_hscale - 0.5, (idx.y + 0.5) *
inv_vscale - 0.5` (standard pixel-center convention), `floor`/`frac` to get
the 4 neighbor texel weights, `input.read_clamped` each of the 4, weighted
sum. `inv_hscale = 1.0/scale` for `Resize` (enlarging when `scale>1`),
`inv_hscale = horizontal` for `Reduce` (shrinking).

- `output_spec()`/`demand()` for `Resize`/`Reduce`/`ReduceHorizontal`/
  `ReduceVertical` are Tier1-fixed (floor/ceil inverse-scale bounds with the
  necessary ±1 halo built into the ceil). Bilinear needs a 1-texel halo on
  the high side, which the existing `ceil(...)`-based `demand()` already
  provides — verify against the formulas in `geometry.rs` before assuming no
  extra halo is needed.
- `self.kernel: Option<Kernel>` (Nearest/Linear/Cubic/Lanczos2/Lanczos3) —
  start with `Linear` (bilinear, as above) for all `Kernel` values as an
  approximation; `Lanczos3` would need a wider (6-tap) kernel and is a later
  refinement. Document the approximation.

#### `Thumbnail<GpuBackend>`

`output_spec()` already computes the aspect-fit target size (Tier1-fixed).
**Recommend NOT a single kernel** — `Thumbnail` is a vips convenience op that
internally does shrink + resize + (optional) crop/rotate/color-management.
Lower it as a **composition** in `Image2D<GpuBackend>`, the same way
`blur()` composes `BlurH`+`BlurV` (see `filters.rs`):
```rust
impl crate::data::image::Image2D<GpuBackend> {
    pub fn thumbnail(&self, width: i32, height: Option<i32>, crop: Option<Interesting>) -> Self {
        let target = /* compute scale from output_spec's formula */;
        let resized = self.push(Resize { input: self.as_input(), scale: target, .. });
        match crop { Some(_) => resized.push(ExtractArea { .. }), None => resized }
    }
}
```
Leave `Thumbnail<GpuBackend>`'s `Lower<GpuBackend>` **absent** (same status as
`Blur<GpuBackend>`, §5.8) — callers use `.thumbnail(...)` instead of
`.push(Thumbnail{..})`.

#### `Rotate<GpuBackend>` (arbitrary angle), `Rot45<GpuBackend>` (D45/135/...)

Both need an **affine-transform + bilinear-sample** kernel:
`rotate_kernel<R>(idx, input, output, float2x2 inv_rotation, float2 center,
float2 out_center, bg...)` — for each output pixel, compute the source
position via inverse rotation about the center, bilinear-sample (with
out-of-bounds → `background`).

- `Rotate::demand()` is Tier1-fixed to "full input" (correct but
  pessimistic — fine for correctness, revisit for perf later).
- `Rot45` D0/D90/D180/D270 are just `Rot90`-equivalent (no 45° involved
  despite the name) — route those through the `Rot90`/`RemapView` path
  (Group C) by angle-mapping `Angle45::{D0,D90,D180,D270} -> Angle::{D0,D90,D180,D270}`.
- `Rot45` D45/135/225/315 (the actual 45°-rotation cases) use the
  `rotate_kernel` above with a fixed 45°/135°/... rotation matrix;
  `output_spec()` already computes the enlarged diagonal-square canvas
  (Tier1-fixed).
- This is a meaningful new kernel (~40-60 lines of Slang) — budget
  accordingly; not a "quick win".

---

### 5.6 Group F — Convolution-reuse for edge detection

`Sobel<GpuBackend>`, `Prewitt<GpuBackend>`, `Scharr<GpuBackend>`
(`src/operation/edge.rs`, all currently `{ input }` only) are textbook 3×3
convolutions. **Do not write new kernels** — `Convolution<GpuBackend>`
already exists and works (`convolution_kernel<R1,R2>`, takes a `mask: Input<ImageKind,B>`).

Plan:
1. Add a small helper that generates a constant 1-band 3×3 (or 3×3 ×2 for
   Sobel's Gx/Gy) image from a `&[f32]` array — check `src/data/image.rs` /
   image-generation API (`Checkerboard` in `custom_ops.rs` is a precedent for
   procedural image generation) for how to construct a constant `Image<GpuBackend>`
   from host floats. If no such helper exists, add
   `Image2D::<GpuBackend>::from_constant_f32(width, height, data: &[f32]) ->
   Self` that uploads a small `GpuBuffer` and treats it as a `Source` leaf
   (1-band `Gray32f`-equivalent `ImageKind`).
2. `Sobel`/`Prewitt`/`Scharr` each need **two** convolutions (Gx and Gy
   masks) followed by `sqrt(gx² + gy²)` (magnitude) — i.e. a 3-step
   composition:
   ```rust
   impl crate::data::image::Image2D<GpuBackend> {
       pub fn sobel(&self) -> Self {
           let gx = self.push(Convolution { input: self.as_input(), mask: sobel_gx_mask(), .. });
           let gy = self.push(Convolution { input: self.as_input(), mask: sobel_gy_mask(), .. });
           // magnitude: sqrt(gx*gx + gy*gy) — needs a 2-input "hypot" kernel,
           // OR: gx*gx via `multiply_kernel(gx,gx)`, gy*gy similarly, add, then
           // math_kernel sqrt — 3 extra arithmetic passes, all using EXISTING
           // arithmetic.slang kernels (multiply_kernel, add_kernel, math_kernel
           // with math=??? — math_kernel has no sqrt; sqrt is case-less in the
           // current switch). Add `sqrt` as a new math_kernel case (cheap,
           // arithmetic.slang edit) OR add a dedicated `hypot_kernel<R1,R2>`.
       }
   }
   ```
   Given the number of passes, a dedicated `edge_magnitude_kernel<R1,R2>(idx,
   gx, gy, output)` doing `sqrt(gx*gx+gy*gy)` per-channel in one pass is
   cleaner than chaining 4 generic arithmetic kernels — add this to a new
   `shaders/ops/edge.slang`.
3. `Sobel<GpuBackend>::lower()` then becomes: this is a **3-node
   composition** (Gx conv, Gy conv, magnitude), same shape as `blur()` — so
   like `Thumbnail`/`Blur`, implement as an `Image2D<GpuBackend>::sobel()`
   method, NOT a single `Lower<GpuBackend> for Sobel<GpuBackend>` impl (a
   single `lower()` call can only emit one fused multi-step shader over a
   *linear* chain of same-shaped steps with a single set of inputs per node —
   a 3-input-dependency DAG (Gx and Gy both read `self.input`, magnitude
   reads both) needs the graph-level node composition, not intra-node
   `cx.kernel()` chaining).
4. `Prewitt`/`Scharr` are identical structure with different 3×3 mask
   coefficients — share the `sobel()`-shaped helper via a private
   `fn edge3x3(&self, gx_mask: ..., gy_mask: ...) -> Self`.

This whole group is a good **Part 3 (datatype polymorphism / reuse) worked
example**: edge detection reuses `Convolution` + a tiny new magnitude kernel,
rather than 3 new bespoke ops.

---

### 5.7 Group G — Non-image output (architecture gap)

`Histogram` (custom_ops.rs, CPU-only `VipsCustomSink` today),
`HistogramSink`, `VectorscopeSink`, and the unused `histogram_kernel`/
`vectorscope_kernel` (which target a `HistogramOut` Slang type, not
`RWRegion`) all share one blocker:

**Open question — confirm before doing any work here**: does
`GpuBuilder`/`OutputWrap`/`GpuView::output()` support a non-tile-shaped
output at all? `cx.output(wrap: OutputWrap)` and `emit_slang`'s binding-0
`target_buffer: RWStructuredBuffer<{target_elem}>` assume the output is
addressed by `region_out` (a 2D tile) via `region_index(view, x, y)`. A
histogram output is a small fixed-size accumulator (`256 * bands` bins, or
`grid_size²` for vectorscope) written via **atomic add**, addressed by *value*
not by *position* — a completely different `OutputWrap`.

This is a **real architecture gap**, not a "just write the kernel" task.
Before assigning this:
1. Grep `src/backend/gpu/{view.rs,mod.rs,materialize.rs}` for any existing
   non-`RWRegion`/`RWCodecRegion` output path (`RWMaskRegion`,
   `RWComplexRegion` in `region.slang` suggest mask/FFT outputs *might* have
   precedent — check if those are wired through `OutputWrap` or are
   dead/aspirational too).
2. If no precedent exists, this needs a new `OutputWrap` variant (e.g.
   `OutBuffer::Atomic { size: u32 }`) + `emit_slang` support for a
   `RWStructuredBuffer<Atomic<uint>>` target bound at binding 0 with a fixed
   size independent of `region_out`, + a `materialize.rs` readback path that
   downloads this fixed-size buffer instead of a `region_out`-shaped tile.
3. Given the scope, recommend treating `Histogram`/`Vectorscope`/stats as
   **CPU-only for now** (they already work via `VipsCustomSink` on the CPU
   path) and scoping the `OutputWrap::Atomic` work as its own follow-up task
   — do not bundle it into per-op GPU work.

---

### 5.8 Group H — Out of scope / CPU-only by design

| Item | Why |
|---|---|
| `Copy<GpuBackend>`, `TileCache<GpuBackend>`, `Linecache<GpuBackend>` (`misc.rs`) | These are vips **demand-driven pipeline hints** (tile caching, line buffering, format/interpretation metadata copy) with no GPU analog — the GPU backend is already a lazy, tiled, cached DAG (`GpuCache`). `Lower<GpuBackend>` should likely be a **no-op passthrough** (`passthrough_kernel`, or even better, not a kernel at all but a graph-level "this node == its input" identity if the builder supports zero-step identity — check if `cx.kernel("passthrough_kernel")` with no params is already that, e.g. as `Crop` with zero offset). Low priority. |
| `Smartcrop<GpuBackend>` (`geometry.rs`) | Content-aware (saliency-based) crop — `demand()` Tier1-fixed to full-input fallback because the crop region is data-dependent. Requires a saliency-analysis pass (existing vips `smartcrop` uses entropy/attention heuristics) before a crop rect is even known — fundamentally CPU-side decision-making feeding a `Crop`. Out of scope. |
| `Grid<GpuBackend>` (`geometry.rs`) | Multi-tile rearrangement, `demand()` Tier1-fixed to full-input fallback (too complex for a single fused kernel). Possible future: `RemapView`-style per-output-tile remap, but the tile-index arithmetic is intricate — defer. |
| `Invertlut<GpuBackend>` (`misc.rs`) | See §5.2 — operates on tiny LUT images, CPU cost negligible. |
| `Maplut<GpuBackend>`, `Recomb<GpuBackend>`, `Ifthenelse<GpuBackend>`, `Case<GpuBackend>` (`misc.rs`) | **Re-scoped to Group D-equivalent, not actually out of scope** — see note below. |
| `icc.rs` (`IccImport`/`IccExport`/`IccTransform`) | ICC profile transforms require LittleCMS (CPU library), no GPU equivalent without reimplementing profile parsing + LUT/matrix application on GPU. Out of scope for this pass. |
| `fft.rs` (all) | FFT requires a GPU FFT implementation (e.g. compute-shader Stockham/Cooley-Tukey) — large standalone effort, unrelated to the per-op patterns here. Out of scope. |
| `mosaicing.rs` (all) | Seam-finding/blending mosaics are inherently iterative/data-dependent (similar to `Smartcrop`). Out of scope. |
| `NoiseReduction<GpuBackend>` (`misc.rs`, `{ input, strength: f32 }`) | Depends on what algorithm — if it's a simple bilateral/median-ish filter, could reuse `Median`'s eventual kernel (also missing) or `blur` as a crude approximation. Needs an algorithm decision before kernel work; flag for follow-up, not blocking. |
| `Sharpen`, `Canny`, `Median`, `HoughLine`, `HoughCircle` (`filters.rs`) | `Sharpen` = unsharp-mask, could reuse `Blur` + arithmetic (subtract blurred from original, add back scaled) — a Group F-style composition, but `flat`/`jagged`/`edge`/`smooth`/`maximum` params imply per-region adaptive behavior beyond simple unsharp; needs algorithm research. `Canny`/`HoughLine`/`HoughCircle` are multi-pass classical CV algorithms (gradient + non-max-suppression + hysteresis / Hough accumulator) — each is its own multi-kernel project, several of which hit the Group G non-image-output gap (Hough accumulator). `Median` needs a sorting-network kernel (`size×size` window). All: defer to dedicated follow-up tasks, not this pass. |

#### Correction: `Maplut`/`Recomb`/`Ifthenelse`/`Case` ARE in-scope (Group D-style multi-input)

These were initially miscategorized as "needs new infrastructure" — they
don't. They're **multi-input pointwise kernels**, same shape as
`Convolution`/`Compose` (§1.2: `inputs()` returns multiple images, kernel
takes `R1, R2[, R3...]`):

| Op | `inputs()` | New kernel | Sketch |
|---|---|---|---|
| `Maplut<GpuBackend>` | `[input, lut]` | `maplut_kernel<R1,R2: IRegion>(idx, input, lut, output, uint band)` | For each channel `c`, `idx_lut = uint(clamp(input.read(idx)[c], 0,1) * float(lut_width-1))`; `output[c] = lut.read(uint2(idx_lut, 0))[band>=0 ? band : c]`. `lut.spec.width` needed as a param (`lut_width`). |
| `Recomb<GpuBackend>` | `[input, matrix]` | `recomb_kernel<R1,R2: IRegion>(idx, input, matrix, output, uint n)` | `output[i] = sum_{j<n} input.read(idx)[j] * matrix.read(uint2(j,i))[0]` for `i in 0..n` (n = input band count, ≤4). `matrix` is an `n×n` 1-band image. |
| `Ifthenelse<GpuBackend>` | `[cond, if_true, if_false]` | `ifthenelse_kernel<R1,R2,R3: IRegion>(idx, cond, a, b, output, uint blend)` | `output = (cond.read(idx).r > 0.5) ? a.read(idx) : b.read(idx)`; if `blend`, lerp by `cond.r` instead of a hard switch. 3-input — confirm `bandjoin3_kernel`'s 3-`IRegion` precedent (`R1,R2,R3`) for the type-param pattern. |
| `Case<GpuBackend>` | `[input, ...cases]` (variable arity, like `Bandjoin`) | `case2_kernel`/`case3_kernel`/... (per arity, like `bandjoin{1..5}_kernel`) | `output = cases[uint(input.read(idx).r)].read(idx)` — index-select among N case images by the rounded value of `input`. Follow `Bandjoin`'s per-arity-kernel pattern (`bandjoin1_kernel`..`bandjoin5_kernel` in `bands.slang`) for `case1_kernel`..`case5_kernel` (or whatever max arity is realistic). |

All four: add `import ops.<new file>;` to `emit.rs`. All four have
`output_spec()`/`demand()` that should already be simple (`(*self.input.spec).clone()`
or similar) — verify when implementing, but none of these change image
dimensions.

---

### 5.9 `Blur<GpuBackend>` — not actually missing, just non-trivial

`Blur<B>` (generic struct, `filters.rs`) has no `Lower<GpuBackend>`, but
`Image2D<GpuBackend>::blur(sigma)` already composes `BlurH` + `BlurV` (two
separate graph nodes, each with correct per-axis halo via `demand()`) — this
is the **reference precedent** for "generic op needs multi-node composition,
provide an inherent method instead of `Lower<GpuBackend>`". Apply the same
pattern for `Thumbnail` (§5.5), `Sobel`/`Prewitt`/`Scharr` (§5.6). Leaving
`Lower<GpuBackend> for Blur<GpuBackend>` absent is **intentional, not a gap**
— don't "fix" it by trying to cram H+V into one `cx.kernel()` chain (the
intra-node temp buffers are sized to `region_out` with no extra halo, so a
chained `blur_v_kernel` reading a chained `blur_h_kernel`'s temp would read
garbage at the vertical edges — see §1.4).

---

## 6. Suggested implementation order

1. **Group A** (§5.1): `Opacity`, `Saturation` (new op), `ShrinkHorizontal`/
   `ShrinkVertical`/`Shrink` via `shrink_kernel`. Low risk, immediate
   parity wins, exercises the existing dangling kernels.
2. **Group B** (§5.2): `Abs`, `Sign` (trivial new kernels), `Cast` (verify
   codec-only approach), `Msb` (8-bit case).
3. **§2 `RemapView`** infrastructure, then **Group C** (§5.3): `Flip`,
   `Rot90` D0/D180, `ExtractArea` (via `Crop`'s existing pattern, no
   `RemapView` even needed), `Replicate`, `Subsample`/`Zoom` (after
   confirming vips semantics).
4. **§5.8 correction set**: `Maplut`, `Recomb`, `Ifthenelse`, `Case` —
   multi-input pointwise, no new infrastructure.
5. **Group E** (§5.5): bilinear `resize_kernel` → `Resize`/`Reduce*`, then
   `Thumbnail` composition.
6. **Group F** (§5.6): Sobel/Prewitt/Scharr via `Convolution` reuse + new
   magnitude kernel — good Part 3 reuse example.
7. **Rot90 D90/D270, Rot45 D45-family, Rotate, Embed/Gravity extend modes**
   (§5.3/§5.4) — more involved coordinate math.
8. **Group G** (§5.7): scope the `OutputWrap::Atomic` architecture work as
   its own task before touching `Histogram`/`Vectorscope` GPU paths.
9. Everything in §5.8's main table (`icc`, `fft`, `mosaicing`, `Smartcrop`,
   `Grid`, `Sharpen`/`Canny`/`Median`/`Hough*`, `NoiseReduction`): separate
   follow-up tasks, not this pass.

Throughout: add `import ops.<file>;` to `emit.rs` for every new `.slang`
file, and add a `gpu_probe.rs` test exercising each new op (follow
`newly_imported_kernel_modules_compile_and_run`'s pattern — push the op,
`pull(&RamImageTarget, rect)`, assert non-empty).
