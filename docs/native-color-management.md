# Native Color Management — Implementation Proposal

> **Status:** proposal. Not yet implemented.
> **Audience:** the engineer (or AI) who will implement this. Every section
> names exact files, exact type definitions, and exact wiring. Follow the
> ordered plan in §11; do not improvise data model shapes — they are given
> concretely here on purpose.
>
> **Prereqs:** read `docs/architecture.md` first (the DAG, `Operation`/`Lower`,
> the GPU fused-pass emitter, the codec sandwich) and `CLAUDE.md` §4 (the
> invariants). This proposal is designed to *honor* those invariants, in
> particular invariant #10 ("color/format conversion is an `Operation`, never
> an implicit fusion step").

---

## 1. Why — the current state is broken

Three independent problems, one root cause.

### 1.1 `PixelFormat` conflates three orthogonal axes

`src/pixel/format.rs` has **40 variants** named `{Model}{BitDepth}`:
`Rgb8`, `Rgba16`, `CmykAF32`, `LabF32`, `YCbCrF16`, `HsvU8`, `XyzF32`, …
Every variant bakes together three things that are mathematically
independent:

| Axis | Examples | What it controls |
|---|---|---|
| **Storage** (sample type) | `8` (u8), `16` (u16), `F16`, `F32` | how one channel is quantized in memory |
| **Color model** | `Rgb`, `Gray`, `Cmyk`, `YCbCr`, `Lab`, `Xyz`, `Hsv`, `Oklab`, … | what the channels *mean* |
| **Alpha** | `…A…` suffix, `Argb` | presence + ordering of an alpha channel |

Adding "Lab in u16" or "ACEScg in f16 with premultiplied alpha" means adding
yet another enum arm and threading it through `bytes_per_pixel`,
`channel_count`, `model_transform`, `has_alpha`, `to_f32`, `with_alpha`,
`component_max_f64`, `into_vips_band_format`, `from_vips_band_format`,
`codec()`, `layout()`, `with_band_count()`, … This is a combinatorial
enum explosion and it is the opposite of the additive philosophy in
`CLAUDE.md` §4. It also cannot represent libvips concepts like *N-band
multispectral* images (no color meaning at all).

### 1.2 The GPU fused pipeline does **no** color management

This is the big one. Trace what actually runs today for a GPU image op:

- `ImageKind::input()` (`src/data/image.rs:76`) builds
  `CodecRegion<{codec}, {layout}>` where `codec ∈ {U8Codec,U16Codec,F32Codec}`
  and `layout` is a `ChannelLayout` enum value. **`self.color_space` is never
  read.**
- `CodecRegion::read` (`shaders/lib/region.slang:31`) is just
  `C.decode(buf, idx, CH)` → raw bytes → `float4`. **No transfer-function
  decode, no primary matrix, no white-point adaptation.**
- The kernel (blur, invert, …) processes those raw, gamma-encoded,
  whatever-primaries values directly, then `RWCodecRegion` re-quantizes them.

So a "blur" silently runs in whatever encoding the bytes happened to be in,
and a P3 image and an sRGB image are treated identically. `ImageKind`
*carries* a `ColorSpace` but the GPU backend throws it away.

There **is** a complete working-space machinery written —
`shaders/lib/color/working.slang` (`to_working`/`from_working`,
`WorkingSource`/`WorkingSink`/`WorkingDecodeRegion`) and an XYZ-hub conversion
kernel `shaders/lib/color/convert.slang` (`color_convert` + `cc_kernel`) with
its param block `shaders/lib/color/params.slang` (`ColorConvertParams`). **None
of it is wired into the fused emitter.** `params.slang`'s header even points
at a dead path (`crates/pixors-shader/src/kernel/color.rs`). It is orphaned
code from an earlier architecture.

### 1.3 Vips color conversion is ICC-only and lossy

`src/color/convert.rs` (`ColorConversion::execute`) converts by calling the
vips `colourspace` operation with an interpretation integer from
`ColorSpace::into_vips_interpretation` (`src/color/space.rs:174`). That
function collapses **every** space to one of two values: `22` (sRGB) or `28`
(scRGB-linear). So Display-P3 → Adobe RGB through this path is silently
wrong. The only faithful path is `operation/icc.rs` (`IccImport`/`IccExport`/
`IccTransform`), which (a) only has a **vips** `Lower` (no GPU), and (b)
requires real ICC profile files.

### 1.4 What we have that is good (keep it)

The **Rust color-science layer is clean and complete** — reuse it verbatim:

- `src/color/space.rs` — `ColorSpace { primaries, white_point, transfer }`.
- `src/color/primaries.rs` — `RgbPrimaries` (incl. `Custom`/`CieXyz`),
  `WhitePoint`, chromaticities.
- `src/color/transfer.rs` — `TransferFn` with `decode`/`encode` (sRGB, Rec709,
  γ2.2/2.4/2.6, ProPhoto, PQ, HLG).
- `src/color/matrix.rs` — `Matrix3x3`, `rgb_to_xyz_matrix`, `bradford_cat`,
  `rgb_to_rgb_transform` (full src→XYZ→adapt→dst composite).
- `src/color/model.rs` — `ColorModelTransform` (CMYK/YCbCr/Lab ↔ RGB), SIMD.
- `src/pixel/*.rs` — the `Pixel` trait + concrete pixel types with SIMD
  pack/unpack (`Rgb`, `Rgba`, `Cmyk`, `Lab`, `Xyz`, `Oklab`, …).
- `src/color/detect.rs` — chromaticity/ICC → `ColorSpace` matching.

The shader-side math is also already correct in `color/convert.slang`
(XYZ-hub `color_convert`) and `color/transfer.slang` (`decode_tf`/`encode_tf`).
We will **resurrect and re-wire** these rather than rewrite them.

---

## 2. Design principles (do not violate)

1. **Mirror libvips' interpretation model.** A libvips image has an
   `interpretation`; operations run *in that interpretation*; you call
   `colourspace`/`icc_transform` to *change* it. We do the same: an image
   carries its true `(model, color_space, alpha)`; color-naive ops (blur,
   invert) run in whatever space the image is in; **color conversion is an
   explicit `Convert` operation**. Never auto-convert to sRGB. Never assume u8.

2. **Honor invariant #10.** Color/format conversion is an `Operation`, not an
   implicit decode baked into every codec read. The codec sandwich handles
   **storage only** (u8/u16/f16/f32 ↔ normalized `float4`). Model + transfer +
   primary-matrix math lives in the `Convert` op's kernel step. This also
   keeps blur's `(2r+1)²` neighbor reads cheap (no matrix per texel).

3. **Honor the two halves (`CLAUDE.md` §2).** The new `Storage` / `ColorModel`
   / `AlphaState` enums and `PixelLayout` are **AGNOSTIC** (no Slang, no vips).
   The storage codec strings, `ColorConvertParams` packing, and the vips
   interpretation mapping are **PER-BACKEND** (`GpuView` / `VipsBand` impls).

4. **Additivity.** Adding "Lab storage = f16" must be zero new enum arms:
   `Storage::F16 × ColorModel::Lab × AlphaState::None`. Adding a *new* model
   (e.g. `Ictcp`) is one arm in `ColorModel` + its `↔XYZ` math in one Slang
   function + one Rust `ColorModelTransform` arm — and nothing else.

5. **GPU and CPU produce the same numbers.** The shader `color_convert` and
   the Rust `ColorModelTransform`/`TransferFn`/`Matrix3x3` must implement the
   identical XYZ-hub pipeline. Cross-backend tests (`tests/cross_backend.rs`)
   enforce this with RMS thresholds — extend them.

---

## 3. The new data model (AGNOSTIC — `src/pixel/` + `src/color/`)

Replace the monolithic `PixelFormat` with three orthogonal enums plus one
descriptor struct. **All of these are `Copy`, `Eq`, `Hash`, `Serialize`.**

### 3.1 `Storage` — sample type only (`src/pixel/storage.rs`, NEW)

```rust
//! How a single channel sample is quantized in memory. No color meaning.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Storage {
    /// 8-bit unsigned, normalized [0,1] by /255.
    U8 = 0,
    /// 16-bit unsigned, normalized [0,1] by /65535.
    U16 = 1,
    /// 16-bit IEEE half float (stored as raw bits; see codec).
    F16 = 2,
    /// 32-bit IEEE float, already in working units.
    F32 = 3,
}

impl Storage {
    pub const fn bytes_per_sample(self) -> usize {
        match self { Storage::U8 => 1, Storage::U16 => 2, Storage::F16 => 2, Storage::F32 => 4 }
    }
    /// Normalization divisor to bring a raw sample into [0,1]; 1.0 for floats.
    pub const fn component_max(self) -> f32 {
        match self { Storage::U8 => 255.0, Storage::U16 => 65535.0, _ => 1.0 }
    }
}
```

`Storage` has **no** `gpu_codec`/`vips_band_format` inherent methods —
those are backend mappings and live in trait impls per §3.6 (the enum stays
AGNOSTIC; see `CLAUDE.md` §2). Do **not** add `gpu_*`/`vips_*` methods to any
of the three enums below.

### 3.2 `ColorModel` — what the channels mean (`src/color/model.rs`, EXTEND)

`ColorModelTransform` already exists for the *decode math*. Add a higher-level
`ColorModel` enum describing the **interpretation** (this is the libvips
`interpretation` analogue). Keep `ColorModelTransform` as the CMYK/YCbCr/Lab
decode helper that `ColorModel` delegates to.

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ColorModel {
    /// Additive RGB. Meaning fully defined by the attached `ColorSpace`.
    Rgb = 0,
    /// Single luminance channel. `ColorSpace` gives its transfer/primaries.
    Gray = 1,
    /// Subtractive CMYK (4 color channels). Naive (no ICC) unless converted.
    Cmyk = 2,
    /// Y'CbCr (full-range, centered 0.5). 3 channels.
    YCbCr = 3,
    /// CIE L*a*b* (D50 connection). 3 channels.
    Lab = 4,
    /// CIE 1931 XYZ tristimulus. 3 channels. The conversion hub.
    Xyz = 5,
    /// CIE xyY. 3 channels.
    Yxy = 6,
    /// Cylindrical Lab (LCh). 3 channels.
    Lch = 7,
    /// HSV (cyclic hue). 3 channels.
    Hsv = 8,
    /// Oklab perceptual. 3 channels.
    Oklab = 9,
    /// Cylindrical Oklab (OkLCh). 3 channels.
    Oklch = 10,
    /// Linear scene-referred RGB ("scRGB"). 3 color channels. (Alpha via AlphaState.)
    ScRgb = 11,
    /// N opaque bands with NO color meaning (multispectral, masks, data planes).
    /// Carries its own channel count. No conversion possible except identity.
    Multiband(u8) = 12,
}

impl ColorModel {
    /// Number of *color* channels (excludes alpha). Multiband returns its N.
    pub const fn color_channels(self) -> usize {
        match self {
            ColorModel::Gray => 1,
            ColorModel::Rgb | ColorModel::YCbCr | ColorModel::Lab | ColorModel::Xyz
            | ColorModel::Yxy | ColorModel::Lch | ColorModel::Hsv | ColorModel::Oklab
            | ColorModel::Oklch => 3,
            ColorModel::Cmyk => 4,
            ColorModel::ScRgb => 3,
            ColorModel::Multiband(n) => n as usize,
        }
    }
    /// The legacy decode transform (RGB-family + Gray => None).
    pub fn transform(self) -> ColorModelTransform { /* map to existing enum */ }
    /// Whether this model is convertible via the XYZ hub (false for Multiband).
    pub const fn is_colorimetric(self) -> bool { !matches!(self, ColorModel::Multiband(_)) }
    // NOTE: no `gpu_model()` here — that backend mapping is a trait impl (§3.6).
}
```

> **Note on `Multiband`.** This is the libvips "bands but no color" case and
> the home for `Mask2DKind`-style data. A `Convert` between a colorimetric
> model and `Multiband` is **rejected at the type/spec level** (return an
> `Error::TypeMismatch`) — you can only `ExtractBand`/`Bandjoin`/reinterpret.

### 3.3 `AlphaState` — alpha presence + premultiplication (`src/pixel/mod.rs`)

Today `AlphaPolicy` (`Straight`/`PremultiplyOnPack`/`OpaqueDrop`) exists but
"has alpha at all" is encoded in `PixelFormat`. Promote alpha to a first-class
axis:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AlphaState {
    /// No alpha channel.
    None = 0,
    /// Straight (non-premultiplied) alpha channel present.
    Straight = 1,
    /// Premultiplied alpha channel present.
    Premultiplied = 2,
}

impl AlphaState {
    pub const fn extra_channels(self) -> usize { matches!(self, AlphaState::None) as usize ^ 1 }
    //                                            ^ None => 0, otherwise 1
    /// Bridge to the existing shader `AlphaPolicy` (Straight=0,Premul=1,OpaqueDrop=2).
    pub const fn to_shader(self) -> u32 {
        match self { AlphaState::Straight => 0, AlphaState::Premultiplied => 1, AlphaState::None => 0 }
    }
}
```

> Keep `AlphaPolicy` as the *operation-time* knob for a `Convert`'s destination
> packing (it already maps to the shader). `AlphaState` describes what an image
> *has*; `AlphaPolicy` describes what a conversion *should produce*.

### 3.4 `PixelLayout` — the replacement descriptor (`src/pixel/meta.rs`)

This replaces `PixelMeta` and is the new single source of truth carried by
`ImageKind`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PixelLayout {
    pub storage: Storage,
    pub model: ColorModel,
    pub alpha: AlphaState,
    /// Primaries + white point + transfer. Meaningful for RGB-family/Gray;
    /// for Lab/Xyz it pins the connection white point; ignored for Multiband.
    pub color_space: ColorSpace,
}

impl PixelLayout {
    pub const fn channel_count(self) -> usize {
        self.model.color_channels() + self.alpha.extra_channels()
    }
    pub const fn bytes_per_pixel(self) -> usize {
        self.storage.bytes_per_sample() * self.channel_count()
    }
    pub fn has_alpha(self) -> bool { !matches!(self.alpha, AlphaState::None) }
}
```

### 3.5 Migration map (old `PixelFormat` → new triplet)

The implementer must port every call site. The mapping is mechanical:

| Old `PixelFormat` | `storage` | `model` | `alpha` |
|---|---|---|---|
| `Rgb8` / `Rgb16` / `RgbF16` / `RgbF32` | U8/U16/F16/F32 | `Rgb` | `None` |
| `Rgba8` / `Rgba16` / `RgbaF16` / `RgbaF32` | … | `Rgb` | `Straight` |
| `Gray8` / `Gray16` / `GrayF16` / `GrayF32` | … | `Gray` | `None` |
| `GrayA*` | … | `Gray` | `Straight` |
| `Cmyk8/16/F16/F32` | … | `Cmyk` | `None` |
| `CmykA*` | … | `Cmyk` | `Straight` |
| `YCbCr8/F16/F32` | … | `YCbCr` | `None` |
| `Lab8/16/F32` | U8/U16/F32 | `Lab` | `None` |
| `XyzF32` | F32 | `Xyz` | `None` |
| `YxyF32` | F32 | `Yxy` | `None` |
| `LChF32` | F32 | `Lch` | `None` |
| `HsvU8` / `HsvF32` | U8/F32 | `Hsv` | `None` |
| `OklabF32` / `OkLChF32` | F32 | `Oklab` / `Oklch` | `None` |
| `ScRgbF32` | F32 | `ScRgb` | `Straight` |
| `Argb32` | U8 | `Rgb` | `Straight` (note byte order — see §5.4) |

**Strategy:** keep `PixelFormat` as a *thin deprecated shim* for one
transition step — add `PixelFormat::into_layout(self, cs: ColorSpace) ->
PixelLayout` and `PixelLayout::legacy_format(self) -> Option<PixelFormat>` so
existing tests compile, then delete `PixelFormat` once all call sites move
(§11 step 8). Do **not** keep both long-term.

### 3.6 Backend mappings are **traits**, not inherent methods

The three enums above are AGNOSTIC and must stay free of any `gpu_*`/`vips_*`
method (a `match self { … "U8Codec" … }` on `Storage` hard-codes a Slang type
name into the agnostic layer — forbidden by `CLAUDE.md` §2). Every
enum→backend mapping is a **trait defined by the backend and implemented for
the agnostic enum**. The impl may live in the same file as the enum (Rust's
coherence allows it since the enum is local), but it must go **through a
backend-owned trait** so the enum has no standalone knowledge of Slang/vips.

GPU side (`src/backend/gpu/`, PER-BACKEND):

```rust
/// Maps an agnostic enum to the Slang token(s) the emitter splices. One trait
/// per axis; the GPU backend owns them, the color/pixel enums implement them.
pub trait GpuStorageCodec { fn gpu_codec(&self) -> &'static str; }   // -> "U8Codec" | ...
pub trait GpuModelId      { fn gpu_model(&self) -> u32; }            // -> ColorModel.slang value
pub trait GpuTransferId   { fn gpu_transfer(&self) -> u32; }         // -> TransferFn.slang value
pub trait GpuAlphaId      { fn gpu_alpha(&self) -> u32; }            // -> AlphaPolicy.slang value

impl GpuStorageCodec for Storage   { fn gpu_codec(&self) -> &'static str { match self { … } } }
impl GpuModelId      for ColorModel { fn gpu_model(&self) -> u32 { match self { … } } }
impl GpuTransferId   for TransferFn { fn gpu_transfer(&self) -> u32 { match self { … } } }
impl GpuAlphaId      for AlphaState { fn gpu_alpha(&self) -> u32 { match self { … } } }
```

Vips side (`src/backend/vips/`, PER-BACKEND) — the existing
`IntoVipsBandFormat` / `IntoVipsInterpretation` traits already follow this
pattern; extend them rather than adding inherent methods:

```rust
impl IntoVipsBandFormat   for Storage    { fn into_vips_band_format(self) -> i32 { … } }
impl IntoVipsInterpretation for (ColorModel, ColorSpace) { … }  // model-aware, §8
```

**Why a trait and not an inherent method:** it keeps the agnostic enum usable
by a backend that does not exist yet (invariant #5 additivity) — a new backend
adds its *own* trait + impls without touching `Storage`/`ColorModel`, exactly
as `GpuView`/`VipsBand` gate a Kind's backend support. Every `gpu_model()` /
`gpu_codec()` call elsewhere in this doc is a **trait method call** on a value
whose trait is in scope, never an inherent method.

---

## 4. `ImageKind` revamp (AGNOSTIC — `src/data/image.rs`)

`ImageKind` currently is `{ format: PixelFormat, color_space, width, height }`.
Replace `format` + `color_space` with the single `layout`:

```rust
#[derive(Clone)]
pub struct ImageKind {
    pub layout: PixelLayout,   // storage + model + alpha + color_space
    pub width: i32,
    pub height: i32,
}

impl ImageKind {
    pub fn new(layout: PixelLayout, width: i32, height: i32) -> Self { ... }
    pub fn dims(&self) -> (i32, i32) { (self.width, self.height) }
    // Convenience used by ops:
    pub fn with_layout(&self, layout: PixelLayout) -> Self { Self { layout, ..self.clone() } }
}

impl AnyKind for ImageKind {
    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        let bpp = self.layout.bytes_per_pixel() as u64;
        match wu { WorkUnit::Region(r) => (r.w.max(0) as u64)*(r.h.max(0) as u64)*bpp, _ => 0 }
    }
    fn dyn_hash(&self, s: &mut dyn Hasher) {
        // PixelLayout is Hash now — no more Debug-string proxy hack.
        self.layout.hash(&mut HasherShim(s)); // or write each field
        s.write_i32(self.width); s.write_i32(self.height);
    }
}
impl Kind for ImageKind { type WorkUnit = Region; }
```

The ergonomic accessors (`width`/`height`/`color_space`) stay; add
`storage()`, `model()`, `alpha()`, `layout()`.

---

## 5. Shader-side plumbing: storage codecs + the view-wrap framework (PER-BACKEND)

§5.1–§5.4 cover the storage codecs (bytes ↔ `float4`). §5.5–§5.12 add the
generic **view-wrap framework** — the ergonomic, composable, lazily-applied
replacement for the prehistoric `working.slang` sandwich, used by `Convert`
(§6) and by any op that wants its kernel to see/emit data through an
interpretation it controls. The framework is **datatype-agnostic**; color is
its first client.

### 5.0 Codecs are storage-only (where invariant #10 is enforced)

The codec sandwich must become
**purely a quantizer**: bytes ↔ normalized `float4` (+ a 5th `extra` for
5-channel CmykA). It must know only `(storage, channel_count)` — **never the
model, never the transfer, never the matrix.** All color math moves to the
`Convert` op (§6).

### 5.1 Slang: generalize codecs to N channels (`shaders/lib/codecs.slang`)

Today each codec branches on `ChannelLayout` (Rgba/Rgb/Gray/GrayA/CmykA),
which entangles channel **count** with model. Replace the per-layout branches
with a generic channel **count** loop. New interface:

```hlsl
interface IStorageCodec {
    // Decode up to 4 channels into rgba (missing channels => 0, alpha=>1 if absent).
    static float4 decode(StructuredBuffer<uint> buf, uint idx, uint nchan);
    // 5th channel (CmykA) or 1.0.
    static float  decode_extra(StructuredBuffer<uint> buf, uint idx, uint nchan);
    static void   encode(RWStructuredBuffer<Atomic<uint>> buf, uint idx, float4 c, uint nchan);
    static void   encode_extra(RWStructuredBuffer<Atomic<uint>> buf, uint idx, float v, uint nchan);
}
```

The existing byte helpers (`byte_buf`, `_u16_norm`, `_f16_at`, `_write_byte`,
`_write_u16`, `_write_f16`) already handle arbitrary byte offsets — keep them.
Rewrite each codec's `decode`/`encode` as a count-driven loop:

```hlsl
struct U8Codec : IStorageCodec {
    static float4 decode(StructuredBuffer<uint> buf, uint idx, uint nchan) {
        uint base = idx * nchan;                  // u8: 1 byte/sample
        float4 c = float4(0,0,0,1);
        for (uint k = 0; k < min(nchan, 4u); k++)
            c[k] = float(byte_buf(base + k, buf)) / 255.0;
        return c;
    }
    static float decode_extra(StructuredBuffer<uint> buf, uint idx, uint nchan) {
        return (nchan >= 5u) ? float(byte_buf(idx*nchan + 4u, buf))/255.0 : 1.0;
    }
    static void encode(RWStructuredBuffer<Atomic<uint>> buf, uint idx, float4 c, uint nchan) {
        uint base = idx * nchan;
        for (uint k = 0; k < min(nchan, 4u); k++)
            _write_byte(buf, base + k, uint(saturate(c[k]) * 255.0 + 0.5));
    }
    static void encode_extra(...) { if (nchan >= 5u) _write_byte(...4th...); }
}
```

**Deletions:** the `ChannelLayout` enum's *model* role disappears.
`shaders/lib/pixel.slang`'s `ChannelLayout { Rgba,Rgb,Gray,GrayA,CmykA }` is
replaced by passing the raw channel count `nchan`. Remove the luma
(`0.2126r+0.7152g+0.0722b`) Gray-encode logic from codecs entirely — Gray is
now produced by a `Convert(Rgb→Gray)` model op, not a codec side effect (this
*fixes* the bug where a true single-channel image got luma-mixed). The
`Atomic<uint>` packing for sub-word formats stays exactly as-is.

### 5.2 Slang: `CodecRegion` takes a count, not a layout (`shaders/lib/region.slang`)

```hlsl
struct CodecRegion<C: IStorageCodec, let N: uint> : IRegion {
    StructuredBuffer<uint> buf;
    BufferRegion view;
    float4 read(uint2 idx) {
        if (idx.x >= view.width || idx.y >= view.height) return float4(0);
        return C.decode(buf, region_index(view, idx.x, idx.y), N);
    }
    float4 read_clamped(int2 idx) { return C.decode(buf, region_index_clamped(view, idx.x, idx.y), N); }
}
struct RWCodecRegion<C: IStorageCodec, let N: uint> : IRegion {
    RWStructuredBuffer<Atomic<uint>> buf;
    BufferRegion view;
    void write(uint2 idx, float4 v) {
        if (idx.x < view.width && idx.y < view.height)
            C.encode(buf, region_index(view, idx.x, idx.y), v, N);
    }
}
```

`SwizzleView`, `RemapView`, `MaskRegion`, `Region`/`RWRegion` are unchanged
(they already operate on `float4`/`float` with no model).

### 5.3 Rust: `ImageKind`'s `GpuView` becomes storage-only (`src/data/image.rs`)

```rust
impl GpuView for ImageKind {
    fn input(&self) -> View {
        View::new(
            "uint",
            format!("CodecRegion<{}, {}>", self.layout.storage.gpu_codec(), self.layout.channel_count()),
            "{ {buf}, {params}[0].region_in_{slot} }",
        )
    }
    fn output(&self, wu: &WorkUnit) -> OutputWrap {
        let r = Region::typed(wu).expect("Region-shaped WorkUnit");
        OutputWrap {
            arg: View::new("uint", "RWRegion", "{ {buf}, {region} }"),
            dest: OutBuffer::Scratch,
            encode: Some(View::new(
                "Atomic<uint>",
                format!("RWCodecRegion<{}, {}>", self.layout.storage.gpu_codec(), self.layout.channel_count()),
                "{ {buf}, {region} }",
            )),
            params: RegionParams::tight(r.w, r.h).into_block("region_out"),
        }
    }
}
```

`codec()` / `layout()` / `with_band_count()` are **deleted** (their jobs are
now `storage.gpu_codec()` and `channel_count()`).

### 5.4 `Argb32` byte order

`Argb32` is the one format whose channel *order* differs (A,R,G,B vs R,G,B,A).
Model it as `Rgb`+`Straight`+`Storage::U8` plus a dedicated storage codec
`ArgbCodec` (or handle the swizzle in the `Convert`/source). Simplest: keep an
`ArgbCodec` storage variant whose `decode`/`encode` reorders bytes; it is still
storage-only (a permutation, no color math). Add `Storage`… no — do **not**
add a Storage arm for it (Argb is u8). Instead let the *source* that produces
Argb emit a one-shot `Convert`/swizzle to canonical RGBA on import. Document
this as the single special case.

### 5.5 The view-wrap framework — motivation

The codec sandwich (§5.1–§5.3) handles **storage**. But ops frequently need
their kernel to see the data in a *different interpretation* than how it is
stored or than the previous step left it: "run the blur in linear light",
"give this kernel Lab so it can shift `a*`", "this op only knows sRGB". The old
answer was `shaders/lib/color/working.slang` — `WorkingSource`/`WorkingSink`
that **hard-coded one** working space (sRGB-gamma straight) and forcibly
bracketed every read/write through it. The shader author had zero control:
direction, target space, and which edges got wrapped were all fixed and global.

The replacement is a **generic, composable, per-edge wrapper stack**, declared
by the op *in its `lower`*, applied lazily, and collapsed to plain arithmetic
by spirv-opt. An op says "wrap my input's `IRegion` so reads arrive in space X"
and/or "wrap my output so writes leave in space Y". Wrappers nest, so several
compose. Nothing is implicit or global. This is the framework the GPU `View`
machinery exists to provide (`docs/architecture.md` §5.2.3/§5.3) — we are
extending it to the **write** side and to **op-driven** (not only alias-node)
use, and color conversion is just one instance.

### 5.6 Two Slang interfaces (read dual + write dual)

`IRegion` already exists (read side: `read`/`read_clamped` → `float4`). Add the
symmetric **write** interface so wrappers compose in both directions:

```hlsl
// shaders/lib/region.slang
interface IWritableRegion { void write(uint2 idx, float4 value); }
```

Make `RWRegion`, `RWCodecRegion`, `RWMaskRegion` declare `: IWritableRegion`
(they already have a matching `write` — just add the conformance). Now any
buffer-backed sink is a generic `W: IWritableRegion`.

### 5.7 The generic wrapper shape (any datatype, any transform)

A read wrapper is `IRegion → IRegion`; a write wrapper is
`IWritableRegion → IWritableRegion`. Both are generic over the wrapped type, so
they **stack by nesting**:

```hlsl
struct XView<R: IRegion> : IRegion {
    R inner;            /* + params */
    float4 read(uint2 idx)        { return f(inner.read(idx)); }
    float4 read_clamped(int2 idx) { return f(inner.read_clamped(idx)); }
}
struct YSink<W: IWritableRegion> : IWritableRegion {
    W inner;            /* + params */
    void write(uint2 idx, float4 v) { inner.write(idx, g(v)); }
}
```

`f`/`g` are the interpretation transform on read / inverse on write.
`SwizzleView<R>` and `RemapView<R>` (`region.slang`) are existing read-side
instances — proof the pattern is already idiomatic. The write side is new.
**This is the only framework primitive; everything else (color, future LUTs,
data reinterpretation) is an instance of it.**

### 5.8 Rust wrap-spec types (`src/backend/gpu/view.rs`)

`ViewAdapter` already models a read wrapper for zero-cost *alias nodes*
(`cx.adapt`). Generalize to an explicit read/write pair usable **inside any
op's `lower`**:

```rust
/// Read-side IRegion→IRegion wrapper. `wrapper` carries an `{inner}` slot for
/// the wrapped Slang type; `ctor` carries `{value}` (wrapped var) + `{params}`.
pub struct ReadWrap  { pub wrapper: String, pub ctor: String, pub params: ParamBlock, pub module: Option<String> }
/// Write-side IWritableRegion→IWritableRegion wrapper. Same templating, applied
/// at the final write/encode.
pub struct WriteWrap { pub wrapper: String, pub ctor: String, pub params: ParamBlock, pub module: Option<String> }
```

(`ReadWrap` ≡ today's `ViewAdapter`, reused/renamed; `WriteWrap` is the new
symmetric half.)

### 5.9 Builder methods (`src/backend/gpu/mod.rs::GpuBuilder`)

Two methods an op calls **inside `lower`**, after `cx.kernel(...)`:

```rust
impl GpuBuilder {
    /// Wrap THIS step's input(s) in a read-wrapper. Stacks (last call = outermost).
    pub fn read_wrap(&mut self, w: ReadWrap) { /* push onto current step's input-wrap stack */ }
    /// Wrap THIS node's output in a write-wrapper. Stacks. Applied around the
    /// final encode/temp write.
    pub fn write_wrap(&mut self, w: WriteWrap) { /* push onto current node's output-wrap stack */ }
}
```

Each wrapper's `params` get a per-wrapper namespace prefix (`w{n}_`), exactly
like the existing alias-adapter prefix `a{n}` — this avoids the std430 field
collision hazard documented in `docs/architecture.md` §5.2.4.

### 5.10 Emitter changes (`src/backend/gpu/emit.rs`)

The emitter already nests `{inner}`/`{value}` for alias-node read adapters
(`read_expr` in `docs/architecture.md` §5.4). Extend that nesting to:

1. **op-driven read wraps** — when a `Step` has an input-wrap stack, nest each
   `ReadWrap.wrapper` around the resolved input expression (innermost = the
   real source/temp), so `blur_kernel`'s `R input` arrives as
   `ColorReadView<CodecRegion<…>>`.
2. **the write side (new)** — when a node has an output-wrap stack, the final
   write target (the codec-sandwich `encode` view, or the step's `RWRegion`
   temp) is nested inside each `WriteWrap.wrapper`, so the kernel's `output`
   parameter is `ColorWriteSink<RWCodecRegion<…>>` and `output.write(...)`
   transforms before encoding.

Both are pure string nesting + param-block merges — no new emitter concepts,
no per-datatype branches (it stays generic per `CLAUDE.md` invariant #4/§2).

### 5.11 Why this is strictly better than `working.slang`

| | `working.slang` (old) | view-wrap framework (new) |
|---|---|---|
| Working space | one, hard-coded (sRGB-gamma straight) | any, chosen per edge by the op |
| Direction | forced both ways, every read/write | opt-in per input / per output |
| Composition | none | wrappers nest arbitrarily |
| Params | baked `ColorSpace` struct only | any `ParamBlock`, namespaced |
| Generality | color-only | any datatype (color is one client) |
| Control surface | none (global) | the op author, in `lower` |
| Cost after opt | inline math | inline math (identical) |

`working.slang` is therefore **deleted** (§9); its math (`color_convert`, the
Lab/XYZ helpers) survives in `lib/color/convert.slang` and is reused by the
color wrapper instances (§5.12).

### 5.12 Color wrappers — the first instance (`shaders/lib/color/interp.slang`, NEW)

```hlsl
import lib.color.convert;   // color_convert(raw, extra, ColorConvertParams)
import lib.region;

/// Read: reinterpret a region from (model,space,alpha)_src to a target via the
/// precomputed params. Generic over ANY IRegion.
struct ColorReadView<R: IRegion> : IRegion {
    R inner; ColorConvertParams p;
    float4 read(uint2 idx)        { float4 c = inner.read(idx);        return color_convert(c, c.a, p); }
    float4 read_clamped(int2 idx) { float4 c = inner.read_clamped(idx); return color_convert(c, c.a, p); }
}
/// Write: convert working float4 to the destination interpretation, then delegate
/// to the wrapped sink (codec/temp). Generic over ANY IWritableRegion.
struct ColorWriteSink<W: IWritableRegion> : IWritableRegion {
    W inner; ColorConvertParams p;
    void write(uint2 idx, float4 v) { inner.write(idx, color_convert(v, v.a, p)); }
}
```

Rust constructors (PER-BACKEND, next to `ConvertParams`):

```rust
pub fn color_read_wrap(p: ConvertParams) -> ReadWrap {
    ReadWrap {
        wrapper: "ColorReadView<{inner}>".into(),
        ctor:    "{ {value}, {params} }".into(),
        params:  ParamBlock::from_pod("cc", &p),
        module:  Some("lib.color.interp".into()),
    }
}
pub fn color_write_wrap(p: ConvertParams) -> WriteWrap {
    WriteWrap {
        wrapper: "ColorWriteSink<{inner}>".into(),
        ctor:    "{ {value}, {params} }".into(),
        params:  ParamBlock::from_pod("cc", &p),
        module:  Some("lib.color.interp".into()),
    }
}
```

Consequence: **`Convert` needs no bespoke `convert_kernel`** (§6 is simplified
accordingly). And "blur in linear" is expressed purely by wrapping a stock
`blur_kernel`:

```rust
// Inside some Blur/LinearBlur Lower<GpuBackend>::lower, after cx.kernel(...):
let src = self.input.spec.layout;
let lin = { let mut l = src; l.color_space = src.color_space.as_linear(); l };
cx.read_wrap(color_read_wrap(ConvertParams::build(src, lin)?));   // reads arrive linear
cx.write_wrap(color_write_wrap(ConvertParams::build(lin, src)?)); // writes leave gamma
cx.output(self.output_spec().output(cx.wu()));
```

The `blur_kernel` is colorspace-naive and untouched; the two wraps inline to
transfer math; the conversion is fully lazy and fully under the op's control.
The CmykA 5th-channel caveat (§6.1.2, CmykA note) applies identically to `ColorReadView`
(it reads `c.a` as the K-alpha proxy; use the specialized source-bound wrapper
when `nchan == 5`).

---

## 6. The color operation family (AGNOSTIC structure + PER-BACKEND lower)

All of these are ordinary `Operation`s (`docs/architecture.md` §8). New file
`src/operation/color.rs`. The flagship is `Convert`.

### 6.1 `Convert` — the universal color/format conversion op

```rust
// src/operation/color.rs
pub struct Convert<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub target: PixelLayout,           // full destination descriptor
    pub intent: RenderingIntent,       // for future gamut mapping; default Relative
}

impl<B: Backend> Operation<B> for Convert<B> where Convert<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]   // pointwise: same region
    }
    fn output_spec(&self) -> ImageKind { self.input.spec.with_layout(self.target) }
    fn dyn_hash(&self, s: &mut dyn Hasher) { self.target.hash(&mut shim(s)); self.intent.hash(...); }
}
```

The **conversion math is identical** on both backends — it is the XYZ-hub
pipeline already written in `shaders/lib/color/convert.slang::color_convert`
and mirrored in `src/color/model.rs` + `transfer.rs` + `matrix.rs`. The
*matrices are precomputed in Rust* and shipped as parameters (this is why no
backend needs color science at runtime).

#### 6.1.1 Building the params — split AGNOSTIC math from PER-BACKEND packing

The conversion needs two things: the **matrices** (pure color science,
AGNOSTIC) and a **shader-shaped param struct** holding shader-enum
discriminants (PER-BACKEND, because `gpu_model()`/`gpu_transfer()`/`gpu_alpha()`
are the §3.6 GPU traits). Keep them in separate halves.

AGNOSTIC helper (`src/color/convert.rs` or a new `src/color/pipeline.rs`) —
just the two `Matrix3x3`, no backend types:

```rust
/// src primaries+wp -> XYZ(D50) hub, and XYZ(D50) -> dst primaries+wp.
/// Pure color science; reuses matrix.rs. Errors on Multiband endpoints.
pub fn convert_matrices(src: PixelLayout, dst: PixelLayout)
    -> Result<(Matrix3x3 /*A: src->XYZ@D50*/, Matrix3x3 /*B: XYZ@D50->dst*/), Error>
{
    if !src.model.is_colorimetric() || !dst.model.is_colorimetric() {
        return Err(Error::TypeMismatch("cannot color-convert a Multiband image".into()));
    }
    let a   = rgb_to_xyz_matrix(src.color_space.primaries(), src.color_space.white_point())?;
    let a50 = bradford_cat(src.color_space.white_point(), WhitePoint::D50).mul(&a);
    let xyz_to_dst = rgb_to_xyz_matrix(dst.color_space.primaries(), dst.color_space.white_point())?.inverse()?;
    let b   = xyz_to_dst.mul(&bradford_cat(WhitePoint::D50, dst.color_space.white_point()));
    Ok((a50, b))
}
```

PER-BACKEND param struct (`src/backend/gpu/color_params.rs`, NEW) — byte-identical
to `ColorConvertParams` in `shaders/lib/color/params.slang`; built where the
§3.6 GPU traits are in scope:

```rust
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ConvertParams {
    pub transfer_src: u32, pub transfer_dst: u32,
    pub a: [f32; 12],   // 3 rows padded to vec4
    pub b: [f32; 12],
    pub alpha_src: u32, pub alpha_dst: u32,
    pub model_src: u32, pub model_dst: u32,
    pub nchan_src: u32, pub nchan_dst: u32,
}

impl ConvertParams {
    pub fn build(src: PixelLayout, dst: PixelLayout) -> Result<Self, Error> {
        let (a, b) = crate::color::convert_matrices(src, dst)?;   // agnostic math
        Ok(Self {
            transfer_src: src.color_space.transfer().gpu_transfer(), // §3.6 GPU trait
            transfer_dst: dst.color_space.transfer().gpu_transfer(),
            a: pad_rows(a), b: pad_rows(b),
            alpha_src: src.alpha.gpu_alpha(), alpha_dst: dst.alpha.gpu_alpha(),
            model_src: src.model.gpu_model(), model_dst: dst.model.gpu_model(),
            nchan_src: src.channel_count() as u32, nchan_dst: dst.channel_count() as u32,
        })
    }
    pub fn identity() -> Self { /* RGB->RGB, identity matrices, sRGB->sRGB */ }
}
```

> The XYZ hub is at **D50** to match the existing Lab math in
> `color/convert.slang` (`XN_D50`, etc.). Keep D50; do not change the hub.

#### 6.1.2 GPU lowering — via the wrap framework, no bespoke kernel

`Convert` does **not** introduce a `convert_kernel`. It uses the §5.5 view-wrap
framework: read the source through a `ColorReadView` (which runs the existing
`color_convert` on read) and write through the **target layout's own codec
sandwich**. The only kernel involved is a stock, fully generic copy:

```hlsl
// shaders/lib/io.slang (generic, datatype-agnostic — NOT color-specific)
public void copy_kernel<R: IRegion>(uint2 idx, R input, RWRegion output) {
    output.write(idx, input.read(idx));
}
```

`Convert`'s `Lower<GpuBackend>`:

```rust
impl Lower<GpuBackend> for Convert<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let params = ConvertParams::build(self.input.spec.layout, self.target)
            .unwrap_or_else(|e| { cx.fail(e); ConvertParams::identity() });
        cx.kernel("lib.io", "copy_kernel");                 // generic move
        cx.read_wrap(color_read_wrap(params));               // src -> target interp on read
        cx.output(self.output_spec().output(cx.wu()));       // target codec sandwich encodes
    }
}
```

`copy_kernel` + a read-wrap is the **universal "materialize a wrapped read"**
pattern — reusable for any terminal view-only transform (a standalone
`flip`/`swizzle`/reinterpret that must produce a real buffer), not just color.
Because the wrap inlines, spirv-opt emits exactly the same code a bespoke
`convert_kernel` would have. The old orphaned `convert_kernel`/`cc_kernel` in
`color/convert.slang` is **not** resurrected as an entry point — only its
`color_convert` *function* is reused (by `ColorReadView`/`ColorWriteSink`).

> **Fusion bonus:** because the conversion is a read-wrap on a generic kernel,
> a `Convert` fused *after* another op reads that op's temp through the wrap for
> free; a `Convert` fused *before* a color-naive op can instead be expressed by
> giving that op a `read_wrap` directly (no separate `Convert` node), which is
> the "blur in linear" pattern (§5.12).

#### 6.1.3 CmykA 5th channel

For 5-channel (CmykA) endpoints the generic `float4`-based `ColorReadView`
cannot carry the K-alpha 5th sample. Use the specialized source-bound wrapper
described next:

```hlsl
// reads BOTH the float4 and the extra channel straight from a storage codec
public void convert_cmyka_kernel<S: IStorageCodec>(uint2 idx,
    StructuredBuffer<uint> src, BufferRegion region, RWRegion output, ColorConvertParams p) {
    uint i = region_index(region, idx.x, idx.y);
    float4 raw = S.decode(src, i, p.nchan_src);
    float extra = S.decode_extra(src, i, p.nchan_src);
    output.write(idx, color_convert(raw, extra, p));
}
```

This is the one place `Convert` binds the raw source slot directly (a legit
"new datatype Slang wrapper" per `CLAUDE.md` §8). All non-CmykA conversions use
the generic `copy_kernel` + `ColorReadView` path above.

> **`ParamBlock::from_pod`** (used by `color_read_wrap`/`color_write_wrap` in
> §5.12) is a small new helper on `view.rs`'s `ParamBlock` that appends a
> `#[repr(C)] Pod` struct as one contiguous std430 field group, under a name
> prefix (`"cc"`) that namespaces it against the §5.2.4 collision hazard. The
> wrap's `params` carry the whole `ConvertParams` — `Convert` itself adds no
> loose `cx.param` calls.

#### 6.1.4 CPU / vips lowering

Two sub-cases, decided by whether vips can represent both endpoints faithfully:

```rust
impl Lower<VipsBackend> for Convert<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let h = cx.input(self.input.src());
        let src = self.input.spec.layout;
        let dst = self.target;
        let out = if vips_can_convert(src, dst) {
            // Native fast path: unpremultiply -> colourspace(interp) -> premultiply/flatten -> cast.
            // Reuse the existing src/color/convert.rs::ColorConversion chain,
            // but driven by PixelLayout instead of PixelMeta.
            vips_native_convert(&h, src, dst)?
        } else {
            // Faithful fallback: our own CPU XYZ-hub via a custom region processor
            // (src/backend/vips/working.rs RegionProcessor) using the SAME
            // ColorModelTransform + TransferFn + Matrix3x3 as the shader. This is
            // what guarantees GPU==CPU for arbitrary spaces vips can't name.
            cpu_convert_region(&h, src, dst)?
        };
        cx.emit(out);
    }
}
```

`vips_can_convert` returns true only for spaces with a faithful vips
interpretation (sRGB, scRGB-linear, Lab, XYZ, CMYK, B_W, …). Everything else
(P3, Adobe RGB, ACEScg, arbitrary primaries) takes the **custom-region CPU
path**, which is a `RegionProcessor` that runs `ColorModelTransform::decode_*`
→ `TransferFn::decode` → `Matrix3x3 A` → `Matrix3x3 B` → `TransferFn::encode`
→ model encode, all already in `src/color/`. The `dispatch_format!` macro in
`working.rs` selects the concrete `Pixel` type per `Storage`.

> **This dual path is the heart of the proposal**: native vips where it is
> correct and fast, our own math where vips would lie. Both share the Rust
> color-science layer, so they cannot diverge from the GPU.

### 6.2 Derived ergonomic ops (thin wrappers over `Convert`)

These are convenience methods on `Image2D<B>` that construct a `Convert` with a
mutated `target` layout — **no new kernels**:

```rust
impl<B: Backend> Image2D<B> where Convert<B>: Lower<B> {
    /// Change the color space (primaries/wp/transfer), keep model+storage+alpha.
    pub fn to_color_space(&self, cs: ColorSpace) -> Self {
        let mut t = self.spec.layout; t.color_space = cs;
        self.push(Convert { input: self.as_input(), target: t, intent: RenderingIntent::Relative })
    }
    /// Change storage (e.g. cast to f32) keeping color identical.
    pub fn to_storage(&self, s: Storage) -> Self { let mut t=self.spec.layout; t.storage=s; self.push(Convert{...}) }
    /// Change model (RGB->Lab, RGB->Gray, ...).
    pub fn to_model(&self, m: ColorModel) -> Self { let mut t=self.spec.layout; t.model=m; self.push(Convert{...}) }
    /// RGB <-> linear of same space (just a transfer change).
    pub fn linearize(&self) -> Self { self.to_color_space(self.color_space().as_linear()) }
}
```

`Gamma` (`src/operation/icc.rs`) stays but gains a **GPU+CPU** `Convert`-style
identity: a pure transfer-curve change is `to_color_space(cs.with_transfer(..))`.
Keep the standalone `gamma_kernel` (`shaders/ops/icc.slang`) for the raw
`exponent` knob (libvips `gamma`), but route the *colorimetric* gamma through
`Convert`.

### 6.3 Band operations (already mostly present)

`ExtractBand`/`Bandjoin`/`Bandbool`/`Bandfold`/`Bandmean` (`src/operation/
bands.rs`) already work on raw channels and are model-agnostic — **they need no
color management**, which is correct: bands are storage-level. After the split,
their `output_spec` should produce a `Multiband(n)` model (not a guessed RGB
format) unless the band count maps to a known model. Update
`with_band_count` (deleted in §5.3) → a free helper
`layout_with_bands(base: PixelLayout, n: usize) -> PixelLayout` that sets
`model = Multiband(n)` for n∉{1,3,4-with-alpha} and `Gray`/`Rgb`/`Rgba`
otherwise. This is the libvips behavior (extracting 1 band gives a `B_W`/mono;
joining 3 gives `sRGB`-tagged only if you say so).

---

## 7. Source/IO: set the real layout, don't assume (`src/data/image.rs`, `src/color/detect.rs`)

Today `FileImageSource` hardcodes `ColorSpace::SRGB` (`image.rs:224,298,529`).
Fix the importers to populate `PixelLayout` faithfully:

- **vips `FileImageSource`**: read `storage` from `vips_image_get_format`,
  `model`+`alpha` from `vips_image_get_bands` + `vips_image_get_interpretation`
  (map the vips interpretation enum → `ColorModel` + a default `ColorSpace`),
  and the real `color_space` from the embedded ICC/chromaticities via the
  **already-written** `src/color/detect.rs::match_chromaticities` /
  ICC-profile parse. Add `FromVipsInterpretation -> ColorModel` (currently it
  only yields `ColorSpace`, and only sRGB/linear at that — replace with a full
  table: 22→Rgb/sRGB, 28→ScRgb/linear, `VIPS_INTERPRETATION_LAB`→Lab,
  `_XYZ`→Xyz, `_CMYK`→Cmyk, `_B_W`→Gray, `_GREY16`→Gray/U16, …).
- **`VipsImageSource`** (vips→GPU bridge): forward the detected layout into the
  uploaded `ImageKind` so the GPU side knows the true space.
- **ICC profiles**: when a matrix/TRC profile is detected, parse it to
  `RgbPrimaries::Custom { red, green, blue }` + `WhitePoint` + `TransferFn`
  (or `Custom` TRC — see §10). When it is a LUT profile, keep the bytes and
  force the **vips `icc_transform`** path for any `Convert` touching it (GPU
  cannot run arbitrary LUT profiles — fall back, or pre-bake to a 3D LUT, out
  of scope here).

---

## 8. Vips interpretation mapping — replace the lossy one (`src/color/space.rs`)

`IntoVipsInterpretation` (collapses to 22/28) and `FromVipsInterpretation`
(only sRGB/linear) are replaced by a proper, **model-aware** mapping:

```rust
// PER-BACKEND (lives behind the vips feature). Maps our (model, color_space)
// to the closest faithful VipsInterpretation, or returns None when no faithful
// vips interpretation exists (caller then takes the CPU custom-region path, §6.1.4).
pub fn to_vips_interpretation(model: ColorModel, cs: ColorSpace) -> Option<i32> {
    match model {
        ColorModel::Gray  => Some(VIPS_INTERPRETATION_B_W),
        ColorModel::Lab   => Some(VIPS_INTERPRETATION_LAB),
        ColorModel::Xyz   => Some(VIPS_INTERPRETATION_XYZ),
        ColorModel::Cmyk  => Some(VIPS_INTERPRETATION_CMYK),
        ColorModel::ScRgb => Some(VIPS_INTERPRETATION_scRGB),
        ColorModel::Rgb if cs == ColorSpace::SRGB => Some(VIPS_INTERPRETATION_sRGB),
        ColorModel::Rgb if cs == ColorSpace::LINEAR_SRGB => Some(VIPS_INTERPRETATION_scRGB),
        _ => None, // P3/AdobeRGB/ACEScg/custom => no faithful vips interp
    }
}
```

`to_pixors_id`/`from_pixors_id` (the compact metadata int) stay for round-trip
tagging, extended with the model so the GPU bridge keeps full fidelity.

---

## 9. What to delete / repurpose (orphan cleanup)

- **Keep the `color_convert` *function*** in `shaders/lib/color/convert.slang`
  (the XYZ-hub math + Lab helpers) — it is reused by the `ColorReadView` /
  `ColorWriteSink` wrappers (§5.12). Make its `ColorConvertParams` byte-identical
  to Rust `ConvertParams`. **Do not** resurrect its `cc_kernel` entry point —
  the wrap framework + `copy_kernel` replace it (§6.1.2).
- **Delete** `shaders/lib/color/working.slang` (`to_working`/`from_working`/
  `WorkingSource`/`WorkingSink`/`WorkingDecodeRegion`) — its hard-coded
  single-working-space sandwich is wholly superseded by the per-edge, composable
  view-wrap framework (§5.5–§5.12). Keeping it would mean two conflicting color
  paths. (If an opt-in "auto-linearize before filter" mode is ever wanted, it is
  an inserted `Convert` node / a `read_wrap` on the filter — never a codec hack.)
- **Remove** the model role of `ChannelLayout` (`shaders/lib/pixel.slang`) — keep
  only the `AlphaPolicy` enum there; channel count is passed as `uint`.
- **Delete** `PixelFormat` after migration (§11 step 8). Keep `PixelMeta` only
  if something outside images needs it; otherwise replace with `PixelLayout`.

---

## 10. Future-proofing hooks (design for, don't fully build)

- **`TransferFn::Custom`** (parametric ICC TRC: a gamma + a/b/c/d/e/f piecewise,
  or a sampled 1D LUT). Add a `Custom { ... }` arm later; the shader
  `decode_tf`/`encode_tf` gain a `Custom` branch reading curve params from the
  param block. Out of scope now but `ConvertParams` should reserve space.
- **`RgbPrimaries::Custom`** already exists — `ConvertParams::build` already
  derives matrices from chromaticities, so arbitrary primaries already work
  end-to-end on GPU the moment importers populate `Custom`.
- **3D LUT / gamut mapping** (`RenderingIntent::Perceptual/Saturation`): a
  separate `Lut3D` datatype + op later; `intent` is plumbed now so the API is
  stable.
- **HDR / unbounded** (`ScRgb`, PQ, HLG): already representable; ensure codecs
  for `F16`/`F32` do **not** `saturate` on encode for `ScRgb`/linear targets
  (add a `clamp_on_encode: bool` derived from `model.is_hdr()`).

---

## 11. Ordered implementation plan (for the implementer)

Each step compiles and tests green before the next. Run after every step:
`cargo build --lib && cargo test --test smoke --test vips_smoke`
(per `CLAUDE.md` §9). Add color cases to `tests/cross_backend.rs` as you go.

1. **Add the enums (no behavior change).** Create `src/pixel/storage.rs`
   (`Storage`), extend `src/color/model.rs` (`ColorModel`), add `AlphaState` to
   `src/pixel/mod.rs`, add `PixelLayout` to `src/pixel/meta.rs`. Derive `Hash`.
   Add `PixelFormat::into_layout` / `PixelLayout::legacy_format` shims.
   *Verify:* unit tests for `channel_count`/`bytes_per_pixel`/migration map.

2. **Switch `ImageKind` to `layout`** (`src/data/image.rs`), keeping
   `color_space()`/`format()` accessors working via the shim. Update every
   constructor call site (sources, tests). *Verify:* full build + smoke.

3. **Storage-only codecs (Slang).** Rewrite `shaders/lib/codecs.slang` to the
   `IStorageCodec` count-driven form; update `CodecRegion`/`RWCodecRegion` in
   `region.slang` to `let N: uint`; add `F16Codec` to the codec name set.
   Update `ImageKind::GpuView` (§5.3); delete `codec()`/`layout()`. *Verify:*
   `tests/cross_backend.rs` storage round-trips (u8/u16/f16/f32) still pass —
   these now prove the codec is pure quantization.

4. **Color math + params.** AGNOSTIC `convert_matrices` in `src/color/` (unit-test
   against `rgb_to_rgb_transform`). PER-BACKEND `ConvertParams::build` +
   `pad_rows`/`identity` in `src/backend/gpu/color_params.rs` using the §3.6 GPU
   traits. *Verify:* `build(sRGB,sRGB)` ≈ identity; `build(sRGB,linear-sRGB)` is
   transfer-only.

5. **The view-wrap framework (Slang + Rust).** Add `IWritableRegion` to
   `region.slang` (conform `RWRegion`/`RWCodecRegion`/`RWMaskRegion`); add the
   generic `copy_kernel` to `lib/io.slang`; add `ColorReadView`/`ColorWriteSink`
   to new `lib/color/interp.slang`; add `ReadWrap`/`WriteWrap` +
   `ParamBlock::from_pod` to `view.rs`; add `cx.read_wrap`/`cx.write_wrap` to
   `GpuBuilder` and the matching read/write nesting to `emit.rs`. Make
   `ColorConvertParams` (params.slang) byte-match `ConvertParams`. *Verify:* a
   `copy_kernel` + identity-wrap round-trips an image unchanged; a `read_wrap`
   with a non-identity matrix changes pixels as expected.

6. **`Convert` op + GPU `Lower`** (`src/operation/color.rs`) via `copy_kernel` +
   `color_read_wrap` (§6.1.2); the CmykA path via `convert_cmyka_kernel`.
   *Verify:* cross-backend `convert_srgb_to_p3_matches_reference` (GPU vs a Rust
   CPU reference from `src/color/`).

7. **`Convert` vips/CPU `Lower`** — `vips_native_convert` (port
   `ColorConversion::execute` to `PixelLayout`) + `cpu_convert_region`
   (`RegionProcessor` using `src/color/`). `to_vips_interpretation` (§8).
   *Verify:* GPU vs CPU `Convert` RMS under threshold for sRGB↔P3↔AdobeRGB↔Lab.

8. **Migrate + delete `PixelFormat`.** Replace remaining `PixelFormat` uses with
   `PixelLayout`; delete the shim, `PixelFormat`, the old
   `IntoVipsInterpretation`/`FromVipsInterpretation` lossy impls, and (decision
   §9) `working.slang`. *Verify:* full `cargo test`.

9. **Fix importers** (§7): faithful layout detection in `FileImageSource` /
   `VipsImageSource`; full interpretation→model table. *Verify:* open a P3 PNG
   and a Lab TIFF, assert the detected `PixelLayout`.

10. **Ergonomic methods + band model fix** (§6.2, §6.3): `to_color_space`,
    `to_storage`, `to_model`, `linearize`; `layout_with_bands`. *Verify:* the
    public-API examples in `tests/` and update `docs/architecture.md`'s op list.

---

## 12. Invariant checklist (must hold at the end)

- **#2 type-blind materializer** — the new color params are injected inside
  `Convert::lower`; the materializer never sees `ColorModel`/`ConvertParams`. ✅
- **#3 one shape enum** — no new `WorkUnit` shape; `Convert` is `Region`. ✅
- **#4 additive datatype** — `Multiband`/new models are arms in `ColorModel` +
  one Slang `↔XYZ` fn; no central match grows. ✅
- **#6 capability gating** — a `Multiband` image simply has no faithful
  `to_vips_interpretation`; conversion errors at `ConvertParams::build`, not via
  a runtime "unsupported backend". (This one error is a *value* error, not a
  backend-capability error — acceptable; it is a genuine type mismatch.) ✅
- **#10 color = Operation** — codecs are storage-only; all color math lives in
  the `Convert` op's read-wrap (or an op's own `read_wrap`/`write_wrap`), never
  baked into a codec read. This proposal exists to *make this true*. ✅
- **Two halves (§2)** — `Storage`/`ColorModel`/`AlphaState`/`PixelLayout` are
  AGNOSTIC (zero Slang/vips, **no** `gpu_*`/`vips_*` inherent methods); the
  mappings `gpu_codec()`/`gpu_model()`/`gpu_transfer()`/`gpu_alpha()` /
  `to_vips_interpretation` are PER-BACKEND **trait impls** (§3.6). ✅
- **Generic, not datakind-specific** — the read/write wrap framework
  (`IRegion`/`IWritableRegion`, `ReadWrap`/`WriteWrap`, `cx.read_wrap`/
  `cx.write_wrap`, `copy_kernel`) is datatype-agnostic plumbing in the GPU
  backend; color is one client. Any datatype/op reuses it for shader-side
  interpretation (§5.5–§5.12). ✅

---

## 13. Concrete end-to-end example (what success looks like)

```rust
// Open a Display-P3, 8-bit image (importer detects P3 primaries from ICC).
let img = Image2D::<GpuBackend>::open("photo_p3.png")?;       // layout: U8/Rgb/Straight/DISPLAY_P3
assert_eq!(img.color_space(), ColorSpace::DISPLAY_P3);

// Blur runs in P3-gamma space (libvips semantics: op acts in current interp).
let soft = img.blur(3.0);

// Explicitly convert to linear ACEScg f32 for compositing — ONE Convert node,
// runs as a fused GPU step (P3 gamma -> linear -> XYZ(D50) -> ACEScg -> linear).
let acescg = soft.to_color_space(ColorSpace::ACES_CG).to_storage(Storage::F32);
assert!(acescg.color_space().is_linear());

// Export back to sRGB 8-bit for display.
let out = acescg.to_color_space(ColorSpace::SRGB).to_storage(Storage::U8);
let bytes = out.pull(&RamImageTarget, Region::full(out.dims(), Lod(0)))?;
```

The same chain on `Image2D::<VipsBackend>` produces matching pixels: P3 and
ACEScg have no faithful vips interpretation, so the two `Convert`s take the CPU
custom-region path (`src/color/` math), while the sRGB export uses native vips
`colourspace`. GPU and CPU agree because both call the identical color-science
functions.
