# Chromors

A next-generation, backend-agnostic image and vector graphics processing engine for Rust. Chromors provides a unified, type-safe API over multiple heterogeneous backends, bridging the gap between pixel-perfect CPU processing, JIT-compiled GPU acceleration, high-fidelity RAW decoding, and vector graphics rendering.

> **Disclaimer:** This project is actively under development and is **not production ready**. Interfaces, internals, and operations are subject to breaking changes without notice. Use at your own risk.

## Key Features

* **Unified Polymorphic Graph:** A single evaluation graph that orchestrates nodes across different computation backends.
* **Backend-Agnostic Core:** The core abstraction `Data<K, B>` isolates operations, allowing you to seamlessly swap or interact between different processing engines.
* **Type-Safe `Kinds`:** Operations are strongly typed by data kinds (`ImageKind`, `Mask2DKind`, `HistogramKind`, `LutKind`, `Fft2DKind`, `VectorGraphicsKind`). The compiler guarantees you don't accidentally treat a histogram or an FFT spectrum as a colorimetric picture.
* **Zero-Copy Interoperability:** Effortlessly bridge buffers between CPU, GPU, and memory via the `Source`/`Target` materialization abstraction.
* **Tiered Caching:** JIT operations are intelligently cached across VRAM, RAM, and Disk to ensure blazing fast real-time responsiveness.
* **Color Science Engine:** Industry-standard color space primitives, chromatic adaptation, and transfer functions (`SRGB`, `ACES_CG`, `DISPLAY_P3`, `REC2020`, etc.).

## The Backends

Chromors delegates the heavy lifting to specialized backends, ensuring the right tool is used for the right job:

### 1. `GpuBackend` (Slang / WGPU)
The workhorse for real-time manipulation. It builds a lazy computation graph and compiles operations Just-In-Time (JIT) using Slang into highly optimized compute shaders. Perfect for interactive filters, live color grading, and heavy mathematical operations. Operations are fused whenever possible to minimize VRAM roundtrips.

### 2. `VipsBackend` (libvips)
The CPU reference implementation. It wraps the legendary `libvips` library through the GObject API. Provides access to hundreds of rock-solid operations. Every GPU operation in Chromors is strictly validated against its Vips counterpart to guarantee sub-pixel mathematical equivalence.

### 3. `RawBackend` (LibRaw)
First-class RAW photo development. It handles `.CR3`, `.ARW`, `.NEF`, and other RAW formats out of the box, preserving the uncompressed sensor data, Bayer patterns, and extreme dynamic range before demosaicing into the working color space.

### 4. `VelloBackend` (Vector Graphics)
A modern 2D vector graphics rasterization backend using Vello. It natively supports the `VectorGraphics` data kind, allowing vector scenes and SVG data to be seamlessly rasterized into `Image2D` buffers for downstream pixel processing by the GPU or CPU.

## Quick Start

Chromors makes it simple to load an image, process it on the GPU, and extract the results.

```rust
use chromors::*;

fn main() -> Result<(), Error> {
    init();
    let ctx = Arc::new(GpuContext::new());

    // 1. Load an image from disk using the CPU reference backend
    let source_img = Image2D::<VipsBackend>::open("photo.jpg")?;

    // 2. Wrap it for the GPU pipeline (lazy upload)
    let gpu_source = Arc::new(VipsImageSource::new(source_img));
    let gpu_img = Image2D::<GpuBackend>::from_source(gpu_source, ctx.clone());

    // 3. Chain lazy operations (No GPU work happens yet!)
    let blurred = gpu_img.blur(4.5);

    // 4. Materialize the result to host memory!
    let host_pixels = blurred.pull(
        &RamImageTarget, 
        Region { x: 0, y: 0, w: 1920, h: 1080, lod: Lod(0) }
    )?;

    Ok(())
}
```

```rust
use chromors::*;

// Build a vector scene
let scene = VectorGraphics::<VelloBackend>::new(my_vello_scene, width, height);

// Rasterize directly into the GPU pipeline
let gpu_img: Image2D<GpuBackend> = scene.rasterize_gpu(&gpu_context)?;

// Or rasterize to CPU RAM for VIPS
let vips_img: Image2D<VipsBackend> = scene.rasterize_vips()?;
```

## Operations Library

Chromors includes a comprehensive suite of operations. The GPU backend provides real-time counterparts for most common tasks, while VIPS covers everything natively.

| Category | Operations | `VipsBackend` | `GpuBackend` |
|---|---|:---:|:---:|
| **Arithmetic** | `Add`, `Subtract`, `Multiply`, `Divide`, `Math`, `Min`, `Max`, `Round`, `Complex` | Yes | Yes |
| **Geometry** | `Crop`, `Resize`, `Rotate`, `Flip`, `Shrink`, `Embed`, `Subsample`, `Replicate` | Yes | Yes |
| **Filtering** | `Blur`, `Sharpen`, `Canny`, `Median`, `HoughLine`, `HoughCircle` | Yes | Yes (Blur/Edge) |
| **Convolution** | `Convolution`, `Convsep`, `Conva`, `Convf`, `Convi`, `Morph` | Yes | Yes |
| **Edge/Color** | `Sobel`, `Prewitt`, `Scharr`, `Abs`, `Sign`, `Invert` | Yes | Yes |
| **Misc & Color** | `Exposure`, `Brightness`, `Saturation`, `Maplut`, `Recomb`, `Cast` | Yes | Yes |
| **Composite** | `Composite2` (Porter-Duff), `Join`, `Insert` | Yes | Yes |
| **Bands** | `Bandmean`, `Bandfold`, `Bandbool`, `ExtractBand`, `Bandjoin` | Yes | Yes |
| **Stats & Hist**| `HistogramFind`, `HistogramEqualize`, `HistMatch`, `Deviate` | Yes | WIP |
| **Frequency** | `ForwardFft`, `InverseFft`, `Spectrum` | Yes | No |
| **Mosaicing** | `Mosaic`, `Match`, `Merge`, `GlobalBalance` | Yes | No |
| **ICC Profiles**| `IccImport`, `IccExport`, `IccTransform` | Yes | No (Uses `Gamma`) |

## Vector Graphics & Interoperability

## Architecture Highlights

### Abstraction Model
Backends are bound by a strict contract of traits. You interact with data through unified mechanisms rather than backend-specific calls:

* `Source<B>`: The only door into the model. Translates backend-specific inputs into the computation graph.
* `Target<K, B>`: The only door out of the model. Materializes graph results into physical RAM or backend-specific structures.
* `Operation<B>`: Represents polymorphic data transformations.
* `Data<K, B>`: The core node structure that holds a lazy evaluation tree. Provides ergonomic inherent methods (like `.blur()`, `.sobel()`) that push operations into the graph.

### Vips is the Ground Truth
Chromors tests heavily rely on cross-backend validation. Operations running on the GPU MUST produce identical mathematical results as `libvips` within a strict tolerance limit. 

### Data Kinds
Not everything is an image. Chromors treats different memory topologies as explicit structs:
* `ImageKind`: Colorimetric picture data (e.g., RGBA pixels).
* `Mask2DKind`: Floating-point weight grids (e.g., convolution kernels).
* `HistogramKind`: Atomic bin counts.
* `LutKind`: 1D lookup tables.
* `Fft2DKind`: Complex-valued frequency planes.
* `VectorGraphicsKind`: Resolution-independent paths and curves.

## Building & Testing

**Requirements:**
- Rust 1.94+
- `libvips` >= 8.6 (dev headers)
- `pkg-config`
- GPU with Vulkan/Metal/DX12 support (wgpu)

```bash
# Build the engine
cargo build

# Run standard tests
cargo test

# Run strict cross-backend mathematical validation
cargo test --test cross_backend
```
