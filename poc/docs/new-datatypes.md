# New Datatypes — What Deserves Its Own Kind

Status: **proposal — ready for implementation**
Companion docs: `core-simplification.md` (core hygiene), `kind-polymorphism.md`
(casting taxonomy: ViewAdapter / Reinterpret / cross-Kind op). This doc answers a
different question: **which payloads currently mis-modeled as `ImageKind` should become
first-class Kinds**, driven by the use cases already in the codebase.

## 1. The symptom and the root cause

Today several vips operations return buffers that we wrap as `Image2D`, and the wrapper
lies. The reported failure mode: *a consumer expects RGB and receives Gray* — e.g.
`HistogramFind` on an RGB image returns a `256×1` buffer whose `output_spec` is
guessed via `with_band_count(...)`, `ForwardFft` returns complex-valued bands that
`PixelFormat` cannot even represent (the impl carries a TODO admitting "only the format
is wrong"), and every convolution mask is a `GrayF32` image **tagged
`ColorSpace::SRGB`** (`Image2D::from_constant_f32`).

The root cause is structural, not a missing `if`: `ImageKind` asserts **colorimetric
semantics** — a `PixelFormat` from the RGB/Gray family, a `ColorSpace`, and (on the
GPU) the codec sandwich that decodes/encodes through it. Histogram bins, convolution
weights, LUT entries, recombination matrices, and FFT spectra are *numeric grids*, not
colorimetric pixels. Forcing them into `ImageKind` forces `output_spec` to guess a
band count and a color space that have no meaning, and every consumer inherits the
guess.

> **Decision rule.** *If the payload must not pass through the color pipeline, it is
> not an Image.* Corollary: a new Kind's job is to have **fewer** degrees of freedom
> than `ImageKind` — no `PixelFormat`, no `ColorSpace` — so there is nothing left to
> lie about.

One latent correctness bug makes this urgent rather than cosmetic: the POC's
`F32Codec.decode` is currently a bit-cast, so mask weights survive the image decode *by
accident*. The moment the working-space conversion lands inside the codec sandwich
(`lib/srgb.slang` already exists for it, and the pixors engine does exactly this),
every mask/LUT/matrix wrapped as an sRGB image gets its weights nonlinearly remapped —
negative convolution weights through a transfer function are garbage. New Kinds with a
**raw, codec-free GPU view** make that breakage impossible by construction.

## 2. Inventory — where `ImageKind` is currently a lie

| Op / site | Field or output | What it really is | Today's damage |
|---|---|---|---|
| `Convolution`, `Compass`, `Morph`, `Conva`, `Convf`, `Convi`, `Convsep`, `Convasep` | `mask: Input<ImageKind, B>` | small grid of raw f32 weights | mask tagged `GrayF32 + SRGB`; goes through image decode; `with_band_count` guesses |
| `Sobel`/`Prewitt`/`Scharr` (GPU composition) | `Image2D::from_constant_f32(3,3,…)` | 3×3 weight matrix | same as above, with hardcoded `ColorSpace::SRGB` |
| `Recomb` | `matrix: Input<ImageKind, B>` | bands×bands recombination matrix | same |
| `Maplut` | `lut: Input<ImageKind, B>` | 1×N lookup table | LUT modeled as picture; band/format guesses |
| `Invertlut` | input *and* output | LUT → LUT | `output_spec` = input image spec (wrong dims, wrong everything) |
| `HistogramFind` | output | per-band bin counts (256×N) | the literal reported bug: "expected RGB, got Gray"; spec hand-built as `256×1` image |
| `HistFindNdim`, `HistFindIndexed` | output | n-dim bin grid | spec TODO admits it only models the 2-band case |
| `HistogramCumulative` / `HistogramNormalize` / `HistMatch` | input+output | histogram → histogram / LUT | chain of pseudo-images; each link re-guesses |
| `ForwardFft` / `InverseFft` / `Spectrum` | output / input | complex-valued frequency plane | `PixelFormat` cannot represent complex; spec TODO: "only the format is wrong" |
| `Match` (mosaicing) | output spec | aligned mosaic with data-dependent extent | placeholder spec (flagged TODO; not fixable by a Kind — listed for completeness) |

Counting use sites: the **mask** misuse appears in 11 ops + 3 edge-detector
compositions; the **histogram** misuse in 6 ops; **LUT** in 3; **FFT** in 3.

## 3. Proposed Kinds, ranked by current use cases

### P0 — `Mask2DKind` (convolution weights / matrices)

The highest-leverage Kind: most use sites, and the one sitting on the codec landmine.

```rust
// data/mask2d.rs
/// A small raw-f32 weight grid: convolution masks, morphology elements,
/// band-recombination matrices. NOT colorimetric: no PixelFormat, no
/// ColorSpace, no codec sandwich — weights are bound and read as plain f32.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask2DKind {
    pub width: i32,
    pub height: i32,
}

impl Kind for Mask2DKind { type WorkUnit = Region; }
// byte_size = w * h * 4

impl GpuView for Mask2DKind {
    /// Raw read, no codec: `MaskRegion` reads f32 and broadcasts (v,v,v,1)
    /// like the existing Gray wrappers — kernels keep their `IRegion` inputs.
    fn input(&self) -> View {
        View::new("float", "MaskRegion", "{ {buf}, {params}[0].region_in_{slot} }")
    }
    fn output(&self) -> OutputWrap { /* raw f32 write, encode: None */ }
}

impl VipsBand for Mask2DKind {
    fn band_format(&self) -> i32 { /* VIPS_FORMAT_DOUBLE — vips matrix images */ }
}

pub type Mask2D<B> = Data<Mask2DKind, B>;
```

Constructors are where the ergonomics live (and where `from_constant_f32` dies):

```rust
impl<B: Backend> Mask2D<B> {
    pub fn from_values(ctx, width, height, values: &[f32]) -> Self;   // replaces from_constant_f32
    pub fn gaussian(ctx, sigma: f64, min_amplitude: f64) -> Self;     // vips_gaussmat
    pub fn log(ctx, sigma: f64, min_amplitude: f64) -> Self;          // vips_logmat (laplacian of gaussian)
    pub fn identity(ctx, n) -> Self;
}
```

Signature changes (mechanical): `Convolution.mask`, `Compass.mask`, `Morph.mask`,
`Conva/Convf/Convi/Convsep/Convasep.mask` become `Input<Mask2DKind, B>`;
`Recomb.matrix` becomes `Input<Mask2DKind, B>` (vips calls all of these "matrix
images" — one Kind covers them; do **not** mint a separate `MatrixKind` for the same
payload). `sobel()/prewitt()/scharr()` use `Mask2D::from_values`. The
`convolution_kernel<R1, R2>` shader is already generic over `IRegion` — it needs **no
change**; only the Slang `MaskRegion` wrapper struct is new (`lib/region.slang`).

Decisions:

- **No `Mask1DKind` for now.** Separable masks (`Convsep`/`Convasep`) are
  `Mask2D` with `height == 1` (helpers `Mask2D::row(...)`/`::col(...)`). A 1-D Kind
  would be the first `Range`-shaped Region-payload hybrid for zero benefit — the
  demand walk for a ≤25-element mask doesn't need 1-D pruning. Revisit only if a
  genuinely 1-D, large, prunable payload appears. (`LutKind` below takes the
  Range-shaped pioneer role instead, where 1-D is semantically true.)
- **f32 canonical on both backends.** Vips matrix images are double; the vips lower
  casts at the boundary. One numeric story, no `PixelFormat` resurrection.
- **Polymorphism hook:** `impl ReinterpretAs<ImageKind> for Mask2DKind`
  (→ `GrayF32`, linear) gives free debug visualization of any mask through the whole
  image pipeline — the byte layouts are identical. This is the `kind-polymorphism.md`
  cast, exercised by a real Kind on day one.

### P1 — `HistogramKind` unification (vips side)

This fixes the *reported* bug directly. The Kind already exists
(`data/histogram.rs`, GPU-only, `Atomic`-shaped). Extend and adopt it as the output of
the vips histogram family:

```rust
pub struct HistogramKind {
    pub bins: u32,
    pub bands: u32,   // NEW: per-band histograms (hist_find of RGB → bands = 3)
}

impl VipsBand for HistogramKind { /* NEW: payload = vips bins×1, uint, `bands` bands */ }
```

Re-typed operations:

| Op | New signature | Notes |
|---|---|---|
| `HistogramFind` | `Image → HistogramKind` | the guessing `output_spec` (256×1, `with_band_count`) is deleted; `bins`/`bands` are now *true* statements |
| `HistogramCumulative` | `Histogram → Histogram` | |
| `HistogramNormalize` | `Histogram → Histogram` | |
| `HistMatch` | `(Histogram, Histogram) → LutKind` | its output *is* the mapping table vips feeds to `maplut` |
| `HistogramPlot` | `Histogram → ImageKind` | the plot is a genuine picture — correct as a cross-Kind op |
| `HistFindIndexed` | `(Image, Image) → Histogram` | |

Stays `Image → Image` deliberately: `HistogramEqualize`, `HistLocal`, `Stdif` (they
*use* histograms internally but consume and produce pictures). `HistFindNdim` keeps
`ImageKind` for now — its output is a genuine n-dim grid that neither `HistogramKind`
nor a flat image models honestly; it keeps its TODO until someone needs it (then:
`HistogramNdKind { bins, dims }`).

Consolidation: `custom_ops.rs`'s CPU `Histogram`/`HistogramSink`/`VectorscopeSink`
become `Lower<VipsBackend>`/Target impls for the same `HistogramKind` instead of a
parallel universe.

Shape note: `HistogramKind` stays `Atomic` — a histogram is indivisible; vips ops
that consume one demand it whole (`WorkUnit::Atomic`), which the demand walk already
supports. This makes the histogram family the first **cross-shape** op chain
(Region→Atomic→Atomic→Region), a good stress test for the core's shape-blindness.

### P1 — `LutKind` (lookup tables)

```rust
// data/lut.rs
/// A 1-D lookup table: `entries` samples × `bands` channels, raw f32.
#[derive(Clone, Debug, PartialEq)]
pub struct LutKind {
    pub entries: u32,
    pub bands: u32,
}

impl Kind for LutKind { type WorkUnit = crate::work_unit::Range; }  // first Range-shaped Kind
```

Re-typed operations: `Maplut` becomes `(Image, Lut) → Image`; `Invertlut` becomes
`Lut → Lut` (its current `output_spec` — "clone the input image spec" — is the purest
example of the lie this doc removes); `HistMatch` produces it (above). Future sources
map 1:1 to vips: `Lut::identity(bits)`, `Lut::build(points)` (`vips_buildlut`),
`Lut::tone(...)` (`vips_tonelut`).

Why it earns Kind status now rather than later: the **histogram-equalization /
tone-curve pipeline** (`hist_find → hist_cum → hist_norm → maplut`) is a current use
case spanning three of its links, and on the GPU a LUT is exactly a small storage
buffer the `maplut` kernel indexes — typed correctly from day one, it ports to
`GpuView` (raw f32, no codec) with no redesign. Being genuinely 1-D, it exercises the
`Range` arm of `WorkUnit` that has had zero users since the shape vocabulary was
introduced — proving the core is shape-generic in practice, not just in comments.

Polymorphism hook: `ReinterpretAs<ImageKind>` is **not** implemented (Range vs Region
shape — the `kind-polymorphism.md` bound correctly rejects it at compile time). Debug
viz goes through `HistogramPlot`-style cross-Kind ops or a host Target
(`Vec<f32>` readback).

### P2 — `Fft2DKind` (complex frequency plane)

```rust
// data/fft2d.rs
/// A complex-valued frequency plane: width × height × bands, 2×f32 per sample.
#[derive(Clone, Debug, PartialEq)]
pub struct Fft2DKind {
    pub width: i32,
    pub height: i32,
    pub bands: u32,
}

impl Kind for Fft2DKind { type WorkUnit = Region; }
// byte_size = w * h * bands * 8
```

Re-typed: `ForwardFft: Image → Fft2D`, `InverseFft: Fft2D → Image`,
`Spectrum: Fft2D → Image` (magnitude display — a real picture). This deletes the
structural TODO in `fft.rs` ("PixelFormat has no complex representation … only the
format is wrong") *without* polluting `PixelFormat` with complex variants that no
image codec could encode — complexity lives where it's true.

Vips-only initially (`VipsBand` → `VIPS_FORMAT_DPCOMPLEX`, vips casts); `GpuView`
(raw `float2` region view) lands if/when a GPU FFT exists. Unlocks the classic
frequency-domain use case with honest types:
`img.fwfft().multiply(mask_spectrum).invfft()` — where the mask multiply is a future
`(Fft2D, Mask2D) → Fft2D` cross-Kind op.

Lower priority than P0/P1 because FFT ops have no GPU path and fewer call sites — but
the Kind definition is ~80 lines and removes a known-wrong spec.

### P3 — sketched, not scheduled (no current ops demand them)

These have **no wrapped operations yet** in the POC; defining them now would be
speculation. Listed so the next person doesn't re-derive the taxonomy:

- **`ScalarKind`** (`Atomic`, one f64) — for `vips_avg`, `vips_deviate`, `vips_max`
  /`min` values, when those reductions get wrapped. GPU path = the `HistogramOut`
  atomic-accumulate pattern with one bin.
- **`StatsKind`** (`Atomic`, fixed table: 10 columns × bands+1 rows of f64) — for
  `vips_stats`/`vips_measure`. A table, never a picture.
- **`PointKind` / `PointListKind`** — `vips_max(&x, &y)` positions, `find_trim`
  rects, mosaicing tie-points. Becomes worth it the day `Match`'s placeholder spec
  (§2 last row) is actually fixed, since the honest model there is
  *(transform estimation → typed transform → apply)* rather than one opaque op.
- **`VideoFrameKind`** — fully designed in `kind-polymorphism.md` §4.5; waits on a
  demuxer.

## 4. What deliberately stays `ImageKind`

To prevent over-rotation: Gray output is **not** the bug — *wrong* Gray is. These are
genuine pictures and keep `ImageKind`, even when single-band:

- Edge magnitudes (`sobel()` result), `Bandmean`, `ExtractBand` results — single-band
  *colorimetric* data; `with_band_count` is truthful here.
- `HistogramPlot` / `Spectrum` outputs — visualizations are pictures.
- `HistogramEqualize`, `HistLocal`, `Stdif`, all of `filters/arithmetic/geometry/
  composite` — picture in, picture out.
- Checkerboard/test-pattern generators.

## 5. Implementation order and cost

| Phase | Work | Touches | Est. |
|---|---|---|---|
| 1 | `Mask2DKind` + `MaskRegion` Slang wrapper + constructors; re-type 11 conv ops + 3 edge compositions; delete `from_constant_f32` | `data/mask2d.rs` (new), `lib/region.slang`, `operation/{convolution,edge,misc(Recomb)}.rs` | the only phase with broad (mechanical) churn |
| 2 | `HistogramKind { bins, bands }` + `VipsBand`; re-type the hist family; consolidate `custom_ops.rs` sinks | `data/histogram.rs`, `operation/stats.rs`, `operation/custom_ops.rs` | medium |
| 3 | `LutKind`; re-type `Maplut`/`Invertlut`/`HistMatch` | `data/lut.rs` (new), `operation/misc.rs`, `operation/stats.rs` | small |
| 4 | `Fft2DKind`; re-type the fft trio | `data/fft2d.rs` (new), `operation/fft.rs` | small |

Each phase compiles + `cargo test --lib` + `cargo test --test gpu_probe` green before
the next. Note phase 1 changes *user-visible* signatures (`Convolution { mask }`), so
land it before more call sites accumulate.

## 6. Verification

- **The codec landmine test (phase 1):** GPU-convolve with a mask containing negative
  weights (Sobel gx) and assert the result matches the vips reference within the
  existing RMS tolerance — *then* keep this test when working-space conversion lands
  in the codec; it is the regression guard this whole doc exists for.
- **The reported bug (phase 2):** `hist_find` on an RGB image pulls a
  `Histogram` whose `bands == 3` and `sum(bins per band) == w*h` — no `ImageKind`
  involved, nothing to mis-expect as RGB.
- **Shape-blindness (phase 2/3):** the `hist_find → hist_cum → hist_norm → maplut`
  chain type-checks as Region→Atomic→Atomic→(Atomic,Region)→Region and matches the
  vips-only reference.
- **Polymorphism cross-check (phase 1):** `mask.reinterpret::<ImageKind>()` renders a
  gaussian mask through the image pipeline (manual/visual; guards the
  `ReinterpretAs` impl).
- `grep -n "from_constant_f32" src/` → no hits after phase 1;
  `grep -n "Input<ImageKind" src/operation/{convolution,fft}.rs` → only the actual
  image inputs remain.
