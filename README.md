# pixors-engine

Backend-agnostic image processing engine powering the Pixors editor. Provides a unified Rust API over multiple processing backends (libvips CPU, wgpu GPU), color science primitives, and a rich operation library spanning arithmetic, geometry, filtering, compositing, statistics, and more.

## Quick start

```rust
use pixors_engine::*;

fn main() -> Result<(), Error> {
    init();

    // VipsBackend is the CPU reference backend
    let img = Image2D::<VipsBackend>::open("input.jpg")?;

    // Chain operations
    let blurred = img.execute(&GaussianBlurOperation { sigma: 3.0, minimum_amplitude: None, precision: None })?;
    let resized = blurred.execute(&ResizeOperation { scale: 0.5, vscale: None, kernel: Some(Kernel::Lanczos3) })?;

    resized.save("output.png")?;
    Ok(())
}
```

## Design philosophy

### Backend-generic models, backend-specific backends

The core abstraction is `Image2D<B: Backend>`, which holds an opaque `B::Handle`. The image struct itself knows nothing about the backend — it delegates to capability traits. Backend-specific methods live on `impl Image2D<VipsBackend>` (e.g. `.execute()`, `.sink()`, `.custom()`) or `impl Image2D<GpuBackend>` (e.g. `.convert()`, `.execute()`, `.fork()`, `.graph()`).

New backends only need to implement `Backend` + the capability traits they support.

### Capability traits

| Trait | Gives you |
|---|---|
| `Backend` | The fundamental marker — `type Handle: Send + Sync`, `type Buffer: Send + Sync` |
| `OpenFile` | `Image2D::<B>::open(path)` — decode from filesystem |
| `OpenBuffer` | `Image2D::<B>::from_buffer(bytes)` — decode from memory |
| `SourceInput` | `Image2D::<B>::new_from_source(src)` — open from stream |
| `TargetOutput<Image2D<B>>` | `img.write_to_target(tgt)` — write to stream |
| `ImageTargetCapability` | `pull_image`/`pull_image_batch` — materialize tiles to host bytes |
| `ColorConversionCapability` | `img.pixel_meta()` / `img.convert(target)` — format/color-space conversion |
| `Operation<Image2D<B>>` | `img.execute(op)` — run a specific operation, typed `Output` |

### Operations are compile-time gated

`img.execute(op)` only compiles if the backend implements `Operation<Image2D<B>> for Op`. This means you cannot accidentally call a GPU-only operation on the Vips backend, or vice versa.

### Vips is the ground truth

Libvips serves as the reference implementation. Every GPU operation is validated against vips — if they produce different results on the same input, the GPU path is considered broken. Cross-backend validation tests live in `tests/cross_backend.rs`.

## Backends

### VipsBackend (CPU) — default

Wraps libvips via FFI through the GObject API. All operations go through `vips_operation_new` → set properties → `vips_cache_operation_buildp` → extract output. Full access to ~200+ libvips operations.

```rust
let img = Image2D::<VipsBackend>::open("photo.jpg")?;
let rgb = img.convert(PixelMeta::new(              // color space conversion
    PixelFormat::RgbaF32, ColorSpace::LINEAR_SRGB, AlphaPolicy::Straight
))?;
let bytes = rgb.write_to_buffer(".png")?;          // encode to memory
```

### GpuBackend (GPU) — lazy, tile-based

Uses wgpu to run compute shaders (compiled from Slang via `pixors-shader`). Operations build a lazy computation graph — no GPU work happens until you materialize a region.

```rust
let ctx = Arc::new(GpuContext::new());

// Wrap a vips image as a graph source — no upload happens yet
let source = GpuSource::new_vips(img, ctx.clone());
let gpu_img = Image2D::<GpuBackend>::new_from_source(&source)?;

// Lazy operation — appends a node to the graph, no GPU dispatch
let blurred = gpu_img.execute(&GaussianBlurOperation {
    sigma: 3.0, minimum_amplitude: None, precision: None,
})?;

// Materialize a tile (this is when GPU work happens)
let region = GpuRegion::new(blurred.graph().clone(), ctx.cache.clone(),
    blurred.root_id(), ctx.clone(), Lod::FULL);
region.prepare(Rect::new(0, 0, 256, 256));
let result = region.materialize()?;          // Arc<MaterializedValue>
let pixels = result.read_bytes(&ctx)?;       // Vec<u8>, image working-space bytes
```

For the typed end-to-end path (decode to a target `PixelMeta`), use
`ImageTargetCapability::pull_image` (`Image2D::<GpuBackend>` does not expose `pull_image`
directly today — drive it via the document/render layer, or call
`<GpuBackend as ImageTargetCapability>::pull_image(&gpu_img.handle, rect, lod)`).

GPU operations follow the same convention as vips: input → ACEScg working space → operation → output space. This ensures mathematically identical results regardless of the image's native color space.

## Operations

Pixors supports three operation implementation styles:

### 1. Native vips operations (`VipsOperation`)

Bridge to libvips native operations through the GObject API. Used for arithmetic, filters, geometry, stats — anything libvips provides natively.

```rust
img.execute(&GaussianBlurOperation { sigma: 3.0, ... })?;
img.execute(&ResizeOperation { scale: 0.5, ... })?;
```

### 2. Custom Rust operations (`VipsCustomOperation` / `VipsCustomSink`)

Run pure Rust code inside the libvips demand-driven pipeline, region by region. No full-image download — vips calls your Rust callback per output tile, reusing all of libvips' threading and I/O optimizations.

**`VipsCustomOperation`** produces an output image:
```rust
struct AddConst { k: u8 }
impl VipsCustomOperation for AddConst {
    fn generate(&self, out: &mut CustomRegion, input: &CustomRegion) -> Result<(), Error> {
        let (_, top, _, h) = out.rect();
        for y in top..top + h {
            let src = input.row(y);
            let dst = out.row_mut(y);
            for (d, s) in dst.iter_mut().zip(src) { *d = s.saturating_add(self.k); }
        }
        Ok(())
    }
}
let out = img.custom(AddConst { k: 10 })?;
// or through the unified API:
let out = img.execute(&Custom(AddConst { k: 10 }))?;
```

**`VipsCustomSink`** reduces an image to an arbitrary Rust value (not an Image). Runs across vips' threadpool with per-thread accumulators:
```rust
impl VipsCustomSink for MySink {
    type Output = MyStats;
    type Acc = MyAcc; // Default + Send
    fn fold(&self, acc: &mut MyAcc, region: &CustomRegion) { ... }
    fn merge(&self, total: &mut MyAcc, part: MyAcc) { ... }
    fn finish(&self, acc: MyAcc) -> MyStats { ... }
}
let stats = img.sink(MySink)?;
let stats = img.execute(&Reduce(MySink))?;
```

`CustomRegion` provides raw row access: `rect()` → `(left, top, width, height)`, `pixel_bytes()`, `row(y)` (read-only), `row_mut(y)` (write, output only).

Wrapper types `Custom<O>` and `Reduce<S>` in `operation/custom_ops.rs` bridge these into the `Operation<VipsBackend>` trait so they work with `Image::execute()`. Example mocks included: `Invert` (per-band `255-x`) and `HistogramSink` (per-band 256-bin counter).

### 3. GPU operations (`GpuOperation`)

Lazy compute graph via wgpu shaders (see GPU pipeline section below).

### Operation categories

All operation structs live in `src/operation/`, organized by category:

| Module | Operations |
|---|---|
| `arithmetic` | Add, Subtract, Multiply, Divide, Linear, Math (sin/cos/log/exp), Boolean, Relational, Round |
| `bands` | Bandbool, Bandfold, Bandmean |
| `composite` | Composite2 (vips), GpuComposite (GPU), Join, ExtractBand, Insert |
| `convolution` | Convolution, Compass, Morph, Conva/Convf/Convi, Convsep, Fastcor/Spcor/Phasecor |
| `edge` | Sobel, Prewitt, Scharr, Invert, Sign, Abs |
| `fft` | Forward FFT, Inverse FFT, Spectrum |
| `filters` | GaussianBlur (vips + GPU), Sharpen, Canny, Median, Hough Line/Circle |
| `geometry` | Crop, Resize, Rotate, Affine, Similarity, Mapim, Embed, Flip, Shrink, Reduce, Zoom, Replicate |
| `icc` | Gamma, Colourspace, IccImport/Export/Transform, Saturation |
| `misc` | Cast, Copy, TileCache, Clamp, Maplut, Recomb, Wrap, Autorotate |
| `mosaicing` | Mosaic, Mosaic1, Match, Merge, GlobalBalance |
| `stats` | Average, Deviate, Min/Max, Histogram (find/equalize/normalize/plot/match), Stats, Project, Profile |

### Image generation

Generator structs in `src/generator.rs` implement `GenerateOperation` and create new images:

```rust
let black = Image::generate(&Black { width: 512, height: 512 })?;
let noise = Image::generate(&GaussNoise { width: 256, height: 256 })?;
let sdf = Image::generate(&Sdf { width: 128, height: 128, shape: SdfShape::Circle })?;
```

### Drawing

Draw operations in `src/draw.rs` draw into existing images:

```rust
img.draw(&DrawCircle { ink: vec![255.0, 0.0, 0.0], center_x: 50, center_y: 50, radius: 30 })?;
img.draw(&DrawRect { ink: vec![0.0, 255.0, 0.0], left: 10, top: 10, width: 100, height: 100 })?;
img.draw(&DrawImage { sub: &overlay, x: 25, y: 25 })?;
```

## Color science

The `color/` module provides primaries, transfer functions, chromatic adaptation, and conversion matrices — all backend-agnostic f32 math, no vips dependency.

```rust
use pixors_engine::color::*;

// Build a full color transform chain
let transform = matrix::rgb_to_rgb_transform(
    RgbPrimaries::Bt709, WhitePoint::D65,    // from sRGB
    RgbPrimaries::Bt2020, WhitePoint::D65,   // to Rec.2020
)?;

// Decode/encode transfer functions
let linear = TransferFn::SrgbGamma.decode(0.5);  // sRGB EOTF
let encoded = TransferFn::Pq.encode(0.3);         // PQ OETF
```

Predefined `ColorSpace` constants: `SRGB`, `LINEAR_SRGB`, `REC2020`, `DISPLAY_P3`, `DCI_P3`, `ACES2065_1`, `ACES_CG`, `PROPHOTO`, `REC2100_PQ`, `REC2100_HLG`, and more.

## Pixel types

The `Pixel` trait provides bidirectional conversion between concrete pixel types and the `[f32; 4]` RGBA intermediate:

```rust
pub trait Pixel: Copy + Pod {
    fn unpack(self) -> [f32; 4];
    fn pack_x4(r: f32x4, g: f32x4, b: f32x4, a: f32x4, mode: AlphaPolicy, out: &mut [Self]);
    fn pack_one(rgba: [f32; 4], mode: AlphaPolicy) -> Self;
}
```

Supported types: `Rgb<T>`, `Rgba<T>`, `Cmyk<T>`, `CmykA<T>`, `Gray<T>`, `GrayAlpha<T>`, `Lab<T>`, `Hsv<T>`, `YCbCr<T>`, `Xyz<f32>`, `Yxy<f32>`, `LCh<f32>`, `Oklab<f32>`, `OkLCh<f32>`, `ScRgb<f32>` — all with `T = u8 | u16 | f16 | f32` as applicable.

## Metadata

The `Metadata` enum provides typed, format-agnostic access to image metadata with ~120+ variants covering EXIF, XMP, IPTC, ICC profiles, and HDR metadata:

```rust
let meta: Vec<Metadata> = img.extract_metadata();
for m in meta {
    println!("{}: {}", m.label(), m.value_str());
}
```

## GPU pipeline internals

### Lazy computation graph

`Image2D<GpuBackend>::execute()` does not compute — it appends a node to a lazy `Graph`:

```
Source(VipsImage) → Op(Blur) → Op(Resize) → ...
```

Each node carries `datatype: Arc<dyn DataType>` (`ImageType`, `HistogramType`, …) — the
open vocabulary describing what it produces. See "The datatype model" in `AGENTS.md` /
`CLAUDE.md`.

### Materialization

When `GpuRegion::materialize()` is called for a tile:

1. **Walk demands** — starting from the requested `WorkUnit` (e.g. `Region{rect, lod}`), walk
   the graph via each op's `input_demands()`, determining which source pixels / ranges each
   input needs (accounts for filter halos)
2. **Fetch source** — read source pixels from vips (upload to GPU buffer) or copy from an
   existing GPU buffer
3. **Fuse + dispatch** — nodes whose `datatype.needs_fused_temp()` is true are fused into one
   Slang shader / one `queue.submit()`; non-image nodes (histograms, …) write directly to
   their target
4. **Cache** — store the resulting `MaterializedValue` in `TieredCache` (VRAM→RAM→Disk),
   keyed by `(Graph::content_hash(root), x, y, w, h)`

For batch tile access, `materialize_batch()` merges overlapping source rects to minimize vips fetches.

### Buffer path (upload → kernel → download)

**Upload:** `ImageBuffer::upload()` / `ImageBuffer::alloc()` create a wgpu buffer via
`GpuBuffer` (STORAGE | COPY_SRC | COPY_DST).

**Kernel dispatch:** `dispatch_kernel()` binds input/output buffers to group 0, params (std430)
to group 1, and dispatches workgroups (8×8). Pipelines are lazily compiled and cached by Slang
text hash in `GpuContext.pipeline_cache`.

**Download:** `GpuBuffer::read_to_cpu()` stages through a MAP_READ buffer, copies the GPU
result, maps asynchronously, and reads back. `MaterializedValue::read_bytes(ctx)` wraps this
for `Storage::Vram`, or returns the bytes directly for `Storage::Host`.

### Coordinate frames

A `MaterializedValue` carries `extent: WorkUnit` — for `Region { rect, lod }` this is the
image-space rect the data represents, which may be a sub-region of a larger VRAM buffer (from
a merged fetch or cache hit). Shaders address sub-rects within larger buffers via
`BufferRegion { stride, x, y, width, height }` params — no wasted copies to tightly-pack
sub-rects before dispatch.

## Adding a new backend

1. Define a marker struct and implement `Backend`:
```rust
struct MyBackend;
impl Backend for MyBackend {
    type Handle = MyHandle;
    type Buffer = MyBuffer;
}
```

2. Implement capability traits as needed (`OpenFile`, `OpenBuffer`, `SourceInput`, `TargetOutput<Image2D<MyBackend>>`, `ImageTargetCapability`, `ColorConversionCapability`).

3. For operations, implement `Operation<Image2D<MyBackend>>` for each op struct. The GPU graph model (`GpuOperation`/`TypedOperation`, lazy `Graph`, `DataType`) is specific to `GpuBackend` — a new GPU-like backend would need its own equivalent execution model, not a shared `GpuOperation` bridge.

4. For backend-specific enum mappings, create an `IntoMyEnum` trait (mirroring `IntoVipsEnum`).

See `tests/cross_backend.rs` for examples of interoperating between backends.

## Cross-backend validation

Every GPU operation has a corresponding test in `tests/cross_backend.rs` that runs both vips and GPU paths on the same input and asserts their outputs match within tolerance. The working convention:

- Linear float operations compare via RMS in the 0..1 range
- Quantized U8 operations compare via RMS in the 0..255 range
- Edge pixels are excluded from comparison where vips and GPU differ in edge handling (vips: extend, GPU: clamp)
- Composite operations run all 13 Porter-Duff blend modes
- Color space round-trips validate the XYZ-hub A/B matrix inversion

## Running tests

```
cargo test                        # all tests
cargo test -- blur                # specific test filter
cargo test --test cross_backend   # cross-backend validation only
```
