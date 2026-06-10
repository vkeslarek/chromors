# CLAUDE.md — pixors-engine deep reference

## Overview

`pixors-engine` is the image processing core of Pixors. It provides a backend-agnostic Rust API for image I/O, color science, pixel manipulation, and an operation library with ~200+ image processing primitives. Two backends exist: `VipsBackend` (libvips CPU, the reference) and `GpuBackend` (wgpu compute shaders, lazy/tiled).

The crate sits at the bottom of the Pixors dependency stack — it depends on `pixors-shader` (for GPU kernels) but nothing else in the workspace. It is consumed by `pixors-ops`, `pixors-document`, and `pixors-desktop`.

---

## 1. The Backend Abstraction

### Central philosophy: generic models, specific backends

`Image2D<B: Backend>` is a phantom-typed handle. The struct itself holds only `B::Handle` + `PhantomData<B>`. No field of `Image2D` knows what backend it is. All backend-specific behavior lives on `impl Image2D<SpecificBackend>` blocks.

This is NOT a trait object pattern — it's compile-time monomorphization. The type parameter `B` selects the backend at compile time, and capability traits gate which methods are available:

```rust
// Only compiles if B: OpenFile
impl<B: Backend + OpenFile> Image2D<B> {
    pub fn open(path: &str) -> Result<Self, Error> { ... }
}

// Only compiles if B: SourceInput
impl<B: Backend> Image2D<B> {
    pub fn new_from_source(source: &B::Source) -> Result<Self, Error>
    where B: SourceInput { ... }
}

// Only compiles if B has Operation<Image2D<B>> for Op
impl<B: Backend> Image2D<B> {
    pub fn execute<Op: Operation<Image2D<B>>>(&self, params: &Op) -> Result<Op::Output, Error> { ... }
}
```

### The trait hierarchy

```
Backend                       ← marker: type Handle: Send + Sync, type Buffer: Send + Sync
├── OpenFile                  ← capability: open from filesystem path
├── OpenBuffer                ← capability: decode from byte buffer
├── SourceInput                ← capability: open from stream (type Source: Send + Sync)
├── TargetOutput<Input>        ← capability: write `Input` to stream (type Target: Send + Sync)
├── ImageTargetCapability      ← capability: pull_image / pull_image_batch — materialize tiles
├── HistogramTargetCapability   ← capability: create_histogram / pull_histogram (type HistogramHandle)
├── ColorConversionCapability   ← capability: pixel_meta / convert (enables Image2D::pixel_meta/convert)
└── Operation<Input> for Op    ← capability: execute a specific operation, typed Output
```

There are no `SourceOps`/`TargetOps`/`RegionOps` marker traits — `SourceInput::Source` and
`TargetOutput::Target` are plain `Send + Sync` associated types, and tile-level pixel access
goes through `ImageTargetCapability::pull_image`/`pull_image_batch` (vips path) or
`GpuRegion::prepare`/`materialize` (GPU path), not a generic `Region` handle.

### The IntoVipsEnum pattern

Rust enums like `BlendMode`, `Kernel`, `OperationMath` have fixed discriminant values that happen to match libvips integer constants. Rather than embedding `i32` repr directly in the enum (which couples the generic type to a specific backend), the pattern is:

```rust
pub trait IntoVipsEnum {
    fn into_vips(self) -> i32;
}
```

The enum's `#[repr(i32)]` provides automatic `as i32` casting, and `IntoVipsEnum` is a zero-cost conversion trait. A new backend would define its own `IntoMyBackendEnum` trait instead. The generic structs never see vips-specific integers unless you call `.into_vips()` on them.

---

## 2. VipsBackend (CPU reference)

### VipsHandle

```rust
pub struct VipsHandle {
    ptr: *mut ffi::VipsImage,
}
```

GObject ref-counted: `Clone` calls `g_object_ref`, `Drop` calls `g_object_unref`. The handle wraps a raw pointer — all safety relies on libvips' own ref-counting being correct.

### VipsGObject operation execution

All vips operations go through the GObject property system, NOT through `vips_call` or the C convenience functions. The flow:

```
VipsGObject::new(b"gaussblur\0")
  → vips_operation_new("gaussblur")
  → wraps as *mut VipsOperation

set_image("in", ptr) / set_double("sigma", 3.0) / etc.
  → g_object_set_property + GValue (boxed type for VipsImage, fundamental for scalars)

build()
  → vips_cache_operation_buildp(&mut op)  ← THIS is where computation happens

run() / run_body() / run_generator()
  → g_object_get_property(op, "out", &out_value)
  → extract VipsImage from GValue
  → vips_object_unref_outputs(op)  ← CRITICAL: must be called AFTER output extraction
```

**Key invariant:** `g_object_get_property` for the output image must happen BEFORE `vips_object_unref_outputs`. Calling unref first drops the output and leaves a dangling pointer.

### GType constants

Bindgen does not generate G_TYPE_* macros for fundamental types (G_TYPE_DOUBLE etc.), so they are hardcoded in `gobject.rs`. The formula is `type_number << 2`:
- G_TYPE_DOUBLE  = 15 << 2 = 60
- G_TYPE_INT     = 6  << 2 = 24
- G_TYPE_STRING  = 16 << 2 = 64
- G_TYPE_BOOL    = 5  << 2 = 20
- G_TYPE_OBJECT  = 20 << 2 = 80 (used for GValue init of Interpolate)

### Runner trait

```rust
pub trait Runner: Sized {
    fn run(op: VipsGObject) -> Result<Self, Error>;
}
```

Different operations return different types. The blanket `impl<T: VipsOperation> Operation<VipsBackend> for T` dispatches to `T::Output::run(op)`, where `Output: Runner`. Implementations:
- `Image2D<VipsBackend>` — the common case (extracts an output image)
- `f64` — for stats operations (Average::Output = f64)
- `Bounds`, `ImagePair`, `Filled`, etc. — domain-specific output types

### VipsCustomOperation and VipsCustomSink

For operations libvips doesn't provide natively, custom operations embed into the vips demand-driven pipeline:

```rust
pub trait VipsCustomOperation {
    fn generate(&self, out: &mut CustomRegion, input: &CustomRegion) -> Result<(), Error>;
}

pub trait VipsCustomSink {
    type Output;
    type Acc: Default + Send + 'static;
    fn fold(&mut self, acc: &mut Self::Acc, region: &CustomRegion) -> Result<(), Error>;
    fn merge(&self, total: &mut Self::Acc, part: Self::Acc);
    fn finish(&self, acc: Self::Acc) -> Result<Self::Output, Error>;
}
```

`VipsCustomSink` runs via `vips_sink` with threadpool — `fold` can be called from multiple threads, `merge` combines partial results. Used by `HistogramSink` for per-band histogram accumulation.

Wrapper types `Custom<O>` and `Reduce<S>` exist to avoid coherence conflicts when implementing `Operation<VipsBackend>` for the same operation from two different angles.

### Regions (vips)

`Region` wraps `*mut VipsRegion` (libvips tile cache). The flow:
1. `region.prepare(left, top, width, height)` → `vips_region_prepare`
2. `region.fetch(left, top, width, height)` → `vips_region_fetch` → copies bytes to Vec → `g_free`
3. `width()`, `height()` → full image dimensions

`Region` is vips-specific (`backend/vips/region.rs`) — there is no generic `RegionOps` trait; `ImageTargetCapability::pull_image`/`pull_image_batch` build on `Region` for tile-based access to arbitrarily large images without full decode.

---

## 3. GpuBackend (GPU — lazy, tiled)

### Design principle: lazy graph, eager materialization

The GPU backend NEVER computes anything on `execute()`. Each operation adds a `GraphNode` to a shared flat DAG. Computation is deferred until `GpuRegion::materialize()` is called for a specific tile, triggering JIT Slang shader emission, compilation, and wgpu dispatch.

This enables:
- Viewport-aware batching (only render visible tiles)
- Source fetch coalescing (merge overlapping source regions into one Vips fetch)
- Output caching (reuse materialized results across tiles)
- Full graph fusion (all nodes in a pass compile to one shader, one dispatch)

### The flat DAG — `Graph`

```rust
// backend/gpu/graph.rs
pub struct Graph {
    pub nodes: Vec<GraphNode>,
    pub sources: Vec<SourceNode>,
    next_id: u32,
}
```

The graph is a flat `Vec`, not a recursive `Arc<enum>`. `topo_order()` (Kahn's algorithm) gives a deterministic execution order. `merge_from(other)` imports another graph with id remapping (used by composite to inject overlay sources). `subgraph_with_overrides(root, overrides)` builds a cut subgraph for staged compilation.

### `GraphNode` — the computation unit

```rust
pub struct GraphNode {
    pub id: NodeId,
    pub inputs: Vec<NodeId>,        // 0 = primary, 1+ = extra (e.g. composite overlay)
    pub eval: NodeEval,             // how to evaluate this node
    pub op: Arc<dyn GpuOperation>,  // for input_demands during materialize walk
    pub params: Vec<Param>,         // scalar GPU params (I32, U32, F32)
    pub datatype: Arc<dyn DataType>, // what kind of value this node produces
}
```

`datatype` replaces the old closed `ValueKind` enum + `dst_meta` override field — see
[`DataType`](#datatype--the-graph-edge-vocabulary) below. Output color space/format
overrides (color conversion) are now expressed by emitting a node whose `datatype`
is an `ImageType { color_space, format }` carrying the *target* metadata, with a
matching `output_decoder()` codec (see `GpuColorConvertOperation` in `op.rs`) —
there is no separate `dst_meta` side channel.

### `NodeEval` — evaluation strategy

```rust
// backend/gpu/value.rs
pub enum NodeEval {
    Kernel(KernelSpec), // fused Slang kernel dispatch — the common case
    // Future: View(ChannelRewrite), Reduction(KernelSpec), Host(Arc<dyn HostOp>)
}

pub struct KernelSpec {
    pub module: &'static str,   // Slang module, e.g. "ops.filters"
    pub function: &'static str, // Slang function name, e.g. "gaussian_blur_kernel"
}
```

`NodeEval::Kernel` maps 1:1 to one Slang function call in the fused shader. Future variants will allow no-dispatch view nodes (for band-channel fusion) and CPU-side host ops (for feature extraction, alignment).

### `DataType` — the graph edge vocabulary

`ValueKind` (closed enum) and `GpuData` (per-kind trait) are gone. Every
[`GraphNode`](#graphnode--the-computation-unit) carries `Arc<dyn DataType>` —
adding a new datatype means writing a new struct + impl, not editing a central
enum:

```rust
// backend/gpu/datatype/mod.rs
pub trait DataType: Send + Sync + Debug + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
    fn needs_fused_temp(&self) -> bool { false }       // only ImageType -> true
    fn write_mode(&self) -> WriteMode { WriteMode::Positional }
    fn byte_size(&self, w: u32, h: u32, image_format: PixelFormat) -> u64;
    fn work_unit_kind(&self) -> WorkUnitKind;          // Region | Range | Atomic
}

/// Static, request-side: decodes a MaterializedValue into a typed payload.
pub trait TypedData: DataType + Sized {
    type Value: Clone + Send + Sync;
    type WorkUnit: AnyWorkUnit;
    fn finish(&self, value: &MaterializedValue, lod: Lod, wu: &Self::WorkUnit, ctx: &GpuContext)
        -> Result<Self::Value, Error>;
}
```

Concrete datatypes live one-per-file in `backend/gpu/datatype/`: `ImageType { color_space, format }`
(`image.rs`), `HistogramType { bins }` (`histogram.rs`), `Mask1dType`/`Mask2dType` (`mask.rs`),
`Fft1dType`/`Fft2dType` (`fft.rs`), `ScalarType`/`PointListType`/`FeaturesType` (`reduction.rs`).
Only `ImageType::needs_fused_temp()` is `true` — it's the only kind that gets a float4
`RWRegion` temp buffer in `alloc_temps`; everything else writes straight to its target.

Two capability sub-traits, implemented per-datatype where applicable:

```rust
/// Datatypes that can be a graph leaf (today: only ImageType).
pub trait Sourceable: DataType {
    fn fetch_region(&self, src: &GpuSource, rect: Rect, lod: Lod, ctx: &Arc<GpuContext>)
        -> Result<Storage, Error>;
}

/// Datatypes that can terminate a pull — blanket-impl'd for every TypedData.
pub trait Targetable: TypedData + Clone {
    fn pull(&self, node: &GraphNodeHandle, lod: Lod, wu: &Self::WorkUnit) -> Result<Self::Value, Error>;
}
```

`WorkUnitKind` (`work_unit.rs`) is the *shape* of a datatype's natural division —
`Region` (2-D rect, Image2D/Mask2D/Fft2D/Features), `Range` (1-D extent, Mask1D/Fft1D),
or `Atomic` (indivisible, Histogram/Scalar/PointList). The typed structs `Region { rect, lod }`,
`Range { start, end }`, `Atomic` and the type-erased `WorkUnit` enum are the payload-carrying
counterparts, used to pass demands across heterogeneous node boundaries
(`GpuOperation::input_demands`).

### `MaterializedValue` — runtime payload

```rust
// backend/gpu/value.rs
pub struct MaterializedValue {
    pub storage: Storage,            // Vram(Arc<GpuBuffer>) | Host(Vec<u8>)
    pub datatype: Arc<dyn DataType>, // no embedded shape tag to re-validate
    pub extent: WorkUnit,            // Region{rect,lod} | Range | Atomic
}

pub enum Storage {
    Vram(Arc<GpuBuffer>),  // image data resident in VRAM
    Host(Vec<u8>),         // histograms, masks, FFTs, scalars, … (or a spilled image)
}
```

`MaterializedValue` replaces `GraphValue::{Image, Raw}` — *every* datatype materializes
the same way (compile the fused DAG, dispatch, land the result in `Storage`); the
typed `TypedData::finish` interprets the bytes through `self`, so there's nothing left
to re-`matches!()`. `MaterializedValue::region()` recovers the `Region { rect, lod }`
for spatially-divisible datatypes (`None` for Range/Atomic).

**Coordinate frames invariant (image buffers):** when `storage` is `Vram`, the buffer may be a
sub-region of a larger merged-fetch/cache buffer; `buffer_coords(image_rect)` maps
image-space coords to buffer-local using `region().rect` vs the requested rect:
```
buffer_x = image_x - source_rect.x + buffer_rect.x
buffer_y = image_y - source_rect.y + buffer_rect.y
```

### `GraphNodeHandle` — the neutral backend handle

```rust
// backend/gpu/handle.rs
#[derive(Clone)]
pub struct GraphNodeHandle {
    pub graph: Arc<Mutex<Graph>>,  // shared mutable DAG
    pub root_id: NodeId,           // where this handle's output sits in the graph
    pub ctx: Arc<GpuContext>,      // wgpu device + queue + pipeline cache + tile cache
}
```

`GraphNodeHandle` is `GpuBackend::Handle` — it is *not* image-shaped. It carries no
width/height/format/color_space fields; those are derived on demand from the root
node's `Arc<dyn DataType>` (downcast to `ImageType` via `Graph::resolve_image_type`).
`Image2D<GpuBackend>` (`backend/gpu/typed/image.rs`) is the typed, host-facing wrapper:
its inherent `width()`/`height()` come from `Graph::node_dims(root_id)`, and
`format()`/`color_space()` delegate to the generic `ColorConversionCapability::pixel_meta()`
blanket impl. `Histogram<GpuBackend>` and other typed handles are likewise thin wrappers
around `GraphNodeHandle`.

Cloning an `Image2D<GpuBackend>` is cheap — all `Arc` fields. All images from the same
source share the same `graph`. Multiple `Image2D` handles can point to different
`root_id`s inside the same graph (e.g. the blurred and original both exist in the
graph simultaneously). `Image2D::fork()` deep-clones the graph (`Graph::clone` +
`salt_fork()`) so subsequent ops on the fork don't pollute the original.

The materialized-tile cache is `GpuContext::cache: RegionCache = Arc<Mutex<TieredCache>>`
(`backend/gpu/cache.rs`) — a unified VRAM→RAM→Disk, content-addressed cache with CLOCK
eviction, shared by every graph on that device. Cache keys are `(content_hash, x, y, w, h)`,
where `content_hash` (`Graph::content_hash`) is a structural hash of the computation
(source identity + each op's kernel/params/datatype + input order) — identical
computations share cache entries across graph forks, independent of `NodeId` churn.

### `GpuSource` — graph leaf, pixel provider

```rust
#[enum_dispatch(AnyGpuSource)]
pub enum GpuSource {
    Image2D(ImageBufferSource),    // pre-existing ImageBuffer (GPU-to-GPU copy on fetch)
    VipsImage(VipsImageSource),    // Image2D<VipsBackend> (upload on demand via region fetch)
}
```

`fetch_region(rect, lod, ctx)` on `VipsImageSource`: prepare a libvips region, fetch bytes to CPU,
`ImageBuffer::upload()` to VRAM. On `ImageBufferSource`: copy/reuse the existing VRAM buffer.
Both return `Arc<ImageBuffer>`. Every `GpuSource` is image-shaped today — `ImageType` is the only
[`Sourceable`](#datatype--the-graph-edge-vocabulary) datatype; non-image datatypes (histograms,
masks, …) are never graph leaves, only node outputs.

### `GpuBuffer` / `ImageBuffer` — VRAM storage units

```rust
// backend/gpu/buffer.rs

/// Payload-agnostic, ref-counted VRAM allocation (STORAGE | COPY_SRC | COPY_DST).
/// Carries no pixel metadata — used directly by histogram/mask/FFT/scalar
/// datatypes via `Storage::Vram`.
pub struct GpuBuffer {
    pub buffer: Arc<wgpu::Buffer>,
    pub byte_len: u64,
}

/// A 2-D image buffer in VRAM — wraps `GpuBuffer` with image metadata.
pub struct ImageBuffer {
    pub buffer: Arc<GpuBuffer>,
    pub width: u32,
    pub height: u32,
    pub meta: PixelMeta,
}
```

Row-major tight-packed layout. Key methods:
- `ImageBuffer::upload(data, w, h, meta, ctx)` — CPU→GPU via `create_buffer_init`
- `ImageBuffer::alloc(w, h, meta, ctx)` — uninitialized VRAM (for kernel output)
- `GpuBuffer::read_to_cpu(ctx)` — GPU→CPU via staging buffer + map_async

### `GpuOperation` / `TypedOperation` traits — graph builder protocol

```rust
// backend/gpu/op.rs
pub trait GpuOperation: Send + Sync + Debug {
    /// Emit this operation into the graph and return the output node id.
    /// `inputs` has no privileged "primary" slot — multi-input ops (composite,
    /// arithmetic, convolution masks) own ALL their inputs.
    fn emit(&self, inputs: &[NodeId], graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId;

    /// Output pixel dimensions. Default: identity. `None` = no spatial output
    /// (histograms, reductions).
    fn output_dims(&self, input_w: u32, input_h: u32) -> Option<(u32, u32)> {
        Some((input_w, input_h))
    }

    /// Map an output WorkUnit to the input WorkUnits this op needs.
    /// `(input_index, work_unit)` pairs. Default: passthrough — input 0 needs
    /// the same WorkUnit as the output.
    fn input_demands(&self, wu: &WorkUnit) -> Vec<(usize, WorkUnit)> {
        vec![(0, wu.clone())]
    }

    /// Scale pixel-space params (e.g. blur sigma) for dispatch at `lod`.
    /// Default: no scaling.
    fn scale_params_for_lod(&self, params: &[Param], lod: Lod) -> Vec<Param> {
        params.to_vec()
    }

    /// Slang wrapper per input slot (index matches `inputs`). Default: every
    /// slot is `InputEncoder::WorkingDecodeRegion` (decode → ACEScg working space).
    fn input_encoders(&self, num_inputs: usize) -> Vec<InputEncoder> {
        vec![InputEncoder::WorkingDecodeRegion; num_inputs]
    }

    /// Slang wrapper for the node's output. Default: `WorkingEncodeRegion { codec: None }`
    /// (inherit color space/format from source/plan). Non-image outputs override
    /// (`HistogramOut`, `RWComplexRegion`, `RWMaskRegion`, …).
    fn output_decoder(&self) -> OutputDecoder {
        OutputDecoder::WorkingEncodeRegion { codec: None }
    }

    /// Which rect sizes this node's compute-shader thread grid. Default `Output`;
    /// reductions (Histogram, Vectorscope) use `Input(0)` — dispatch over the
    /// scanned input, not the (shapeless) output.
    fn dispatch_grid(&self) -> DispatchGrid {
        DispatchGrid::Output
    }
}

/// Statically declares the `DataType` an op's emitted node produces. Kept
/// separate from `GpuOperation` (object-safe, stored as `Arc<dyn GpuOperation>`
/// on every node) since `Output` varies per concrete op.
pub trait TypedOperation: GpuOperation {
    type Output: DataType;
}
```

The key invariant: **`emit()` builds the graph node, it does not execute anything**. The `self_arc`
is stored on the node so `input_demands()` is reachable later during the materialize walk.

Helpers in `backend/gpu/op.rs`:
- `emit_image(graph, input, op_arc, module, function, params)` — single-input, `working_image_type()` output (ACEScg `RgbaF32`)
- `emit_unary(graph, input, op_arc, module, function, params, datatype)` — single-input, any `Arc<dyn DataType>` output
- `splice_sibling(graph, other: &Image2D<GpuBackend>) -> NodeId` — merges another image's subgraph into `graph` (via `Graph::merge_from`) and returns the remapped node id for `other`'s root; same-graph siblings (compositing onto self) are detected via `try_lock` and need no merge. This is how multi-input ops (Composite, Add, Convolution mask, …) pull in their other operands lazily — see [§7](#how-to-add-a-multi-input-gpu-operation-eg-composite-arithmetic).

### The materialization pipeline

`GpuRequest<D: TypedData>::materialize()` (`backend/gpu/request.rs`) is the typed entry point —
`GpuRequest<ImageType>` for image tiles, `GpuRequest<HistogramType>` for histogram reads, etc.
Internally it still drives `GpuRegion::materialize()` / `materialize_batch()` (`backend/gpu/region.rs`,
`materialize.rs`):

1. **Cache check** (`GpuContext::cache: TieredCache`, `cache.rs`) — keyed by `(content_hash, x, y, w, h)` where `content_hash = Graph::content_hash(root)` is a structural hash of the computation (source identity + each op's kernel/params/datatype + input order, independent of `NodeId`). Hit in VRAM/RAM/Disk → return/promote immediately.
2. **Region mapping** (`Graph::materialize`) — `input_demands` walk backwards from the target `WorkUnit` through the op chain, accumulating per-node demands. Produces `MaterializePlan { sources, targets, node_outputs, source_fetches }`.
3. **Region merge** (`merge_overlapping`) — coalesces touching/intersecting source rects into fewer, larger rectangles. One merged region = one Vips fetch = fewer round-trips. GPU amortizes shader setup cost over more pixels.
4. **Graph cut check** (`bfs_find_cuts`) — BFS from root counts how many unique storage buffers would be bound simultaneously. If over the device limit (`max_storage_buffers`, commonly 8), the graph is cut: overflowing sub-trees are pre-materialized and injected as `ImageBufferSource` nodes (`subgraph_with_overrides`). Each pass stays within hardware limits.
5. **IR emission** (`emit_ir_with_layout`) — builds `LayoutPlan` (buffer slot assignment, temp interval coloring via `datatype.needs_fused_temp()`, param layout), then `emit_slang` generates JIT Slang source text. All buffers addressed positionally (not by NodeId) → identical graphs produce identical shader text → pipeline cache reuse.
6. **Parallel compile + fetch** (`rayon::join`) — slangc compilation and source pixel fetch run in parallel.
7. **Encode + dispatch** (`Compiled::encode`) — one `wgpu::CommandEncoder`, one `queue.submit()`. Readback for non-image outputs (histogram, scalar, …) via staging buffer, landing in `MaterializedValue::Storage::Host`.

The `TieredCache` (VRAM → RAM → Disk, CLOCK/second-chance eviction, `cache.rs`) is shared by every
graph on the same `GpuContext` — eviction demotes entries one tier down rather than dropping them
outright, except off Disk.

### LOD (Level of Detail) system

`Lod(n)` = `1/2^n` scale. `Lod(0)` = full resolution. Carried in `WorkUnit::Region { rect, lod }`, not on the handle. `input_demands` receives the `WorkUnit` so halo ops (blur) can scale their radius: `radius / lod.scale_factor()`. Pixel-space params are scaled via `scale_params_for_lod()`, dividing by the scale factor before dispatch.

### JIT shader fusion (`emit.rs`)

The emitter fuses all nodes whose `datatype.needs_fused_temp()` is `true` (today: only `ImageType`) in a pass into **one Slang shader** with **one entry point per needed node**. The key mechanisms:

- **`alloc_temps`** — assigns float4 `RWRegion` temp buffers to `needs_fused_temp()` nodes using interval coloring. Other nodes (Histogram, etc.) bypass temps and write directly to their target via `write_mode()`.
- **Positional naming** — all shader names are index-based (`src_0`, `temp_buf_1`, `target_0`, `u0`, `u1`). Structurally identical graphs produce identical Slang text and hit the Slang cache.
- **`WorkingDecodeRegion`** — wraps each source as a lazy decode-on-read view (no copy). The working-space sandwich (`to_working` on read, `from_working` on write) is baked into the emitted code.
- **`ChainParams`** — all scalar params (source regions, temp regions, target regions, op params) are packed into one std430 struct, one GPU buffer, one binding.

### Bind group layout

```
Group 0 (read-only):   src_0 .. src_N | params_buf
Group 1 (read-write):  temp_0 .. temp_M | target_0 .. target_K
```

Device limit = `max_storage_buffers` (typically 8). BFS cut ensures each pass stays under this. Sources + params occupy group 0, temps + targets occupy group 1.

### `GpuContext` — wgpu device wrapper

`backend/gpu/context.rs`. Holds `device`, `queue`, `arena` (buffer pool), `pipeline_cache` (compiled pipelines keyed by Slang text hash), `max_storage_buffers`. Shared via `Arc` across all images from the same GPU device init.

---

## 4. Operation System

Operations in pixors-engine use one of three implementation patterns:

### Pattern A: VipsOperation — GObject-based (native vips ops)

```rust
pub trait VipsOperation {
    type Output: Runner;
    fn name() -> &'static [u8];
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage);
}
```

A blanket impl bridges `VipsOperation` → `Operation<VipsBackend>`:
```rust
impl<T: VipsOperation> Operation<VipsBackend> for T { ... }
```

Most operations in `src/operation/` use this pattern: arithmetic, filters, geometry, stats — anything libvips provides natively.

### Pattern B: Custom ops — Rust embedded in the vips pipeline

Custom operations run pure Rust code INSIDE the libvips demand-driven pipeline via `vips_image_generate` (for output images) or `vips_sink` (for reductions). This means the operation inherits libvips' threading, I/O scheduling, and memory management — but the pixel logic is pure Rust, not vips GObject calls. No full-image download — work happens region by region.

#### VipsCustomOperation — produces an output Image

```rust
pub trait VipsCustomOperation: Send + Sync + 'static {
    fn generate(&self, out: &mut CustomRegion, input: &CustomRegion) -> Result<(), Error>;
}
```

`generate` is called by vips for each output region. The output has the same geometry and format as the input (a `copy`-style pipeline). `input` is prepared to exactly the output's valid rect.

Usage:
```rust
// Direct API on Image2D<VipsBackend>
let out = img.custom(Invert)?;

// Through unified execute() — requires the Custom<> wrapper
let out = img.execute(&Custom(Invert))?;
```

Internals (`backend/vips/custom.rs`):
- `image.custom(op)` calls `vips_image_new()` + `vips_image_pipelinev()` to copy geometry/format
- Attaches input + boxed op as `CustomHolder` with `vips_image_generate()`
- `generate_tramp` is the C callback: it prepares the input region (`vips_region_prepare`), constructs `CustomRegion` wrappers for both input and output, then calls `op.generate()`
- Drop cleanup via `g_object_set_data_full` with a destroy-notify callback that drops the holder and unrefs the input image

#### VipsCustomSink — produces an arbitrary Rust value (reduction)

```rust
pub trait VipsCustomSink: Send + Sync + 'static {
    type Output;
    type Acc: Default + Send + 'static;

    fn fold(&self, acc: &mut Self::Acc, region: &CustomRegion);
    fn merge(&self, total: &mut Self::Acc, part: Self::Acc);
    fn finish(&self, acc: Self::Acc) -> Self::Output;
}
```

This is how vips' own `avg`, `min`, `stats` work. Vips scans the image across its threadpool — each thread folds regions into its own `Acc`. When a thread finishes, `merge` combines it into the global accumulator. Finally `finish` produces the output value.

Usage:
```rust
// Direct API
let hist = img.sink(HistogramSink)?;   // Histogram { bins: Vec<[u32; 256]> }

// Through unified execute()
let hist = img.execute(&Reduce(HistogramSink))?;
```

Internals:
- `image.sink(sink)` creates a `SinkState<S>` wrapping the sink and a `Mutex<S::Acc>` for the global accumulator
- Calls `vips_sink()` with three trampolines: `sink_start` (allocates a fresh `S::Acc` per thread), `sink_generate` (calls `sink.fold()`), `sink_stop` (calls `sink.merge()` into the global accumulator)
- After `vips_sink` completes, calls `sink.finish()` on the global accumulator

#### CustomRegion — safe Rust view of a vips region

```rust
pub struct CustomRegion {
    ptr: *mut ffi::VipsRegion,
    psize: usize,  // bytes per pixel
}

impl CustomRegion {
    pub fn rect(&self) -> (i32, i32, i32, i32);  // (left, top, width, height)
    pub fn pixel_bytes(&self) -> usize;
    pub fn row(&self, y: i32) -> &[u8];           // read-only row
    pub fn row_mut(&self, y: i32) -> &mut [u8];   // mutable row (output only)
}
```

The row is `width * pixel_bytes` bytes. `row_ptr()` uses `vips_region.bpl` (bytes per line) for stride — NOT `width * psize` — because vips may pad rows for alignment.

#### Coherence wrappers (operation/custom_ops.rs)

Rust coherence rules prevent implementing `Operation<VipsBackend>` for both `VipsOperation` (blanket impl) and `VipsCustomOperation` on the same type. The wrappers solve this:

```rust
// Wraps a VipsCustomOperation so it runs through Image::execute()
pub struct Custom<O>(pub O);
impl<O: VipsCustomOperation + Clone> Operation<VipsBackend> for Custom<O> { ... }

// Wraps a VipsCustomSink so it runs through Image::execute()
pub struct Reduce<S>(pub S);
impl<S: VipsCustomSink + Clone> Operation<VipsBackend> for Reduce<S> { ... }
```

Example mocks in `custom_ops.rs`:
- **`Invert`** — per-band 8-bit invert (`255 - x`), produces output image
- **`HistogramSink`** — per-band 256-bin histogram (only meaningful for u8 formats), `merge` adds counts across threads, `finish` returns the accumulated histogram

### Pattern C: GpuOperation — lazy GPU compute

```rust
pub trait GpuOperation: Send + Sync + Debug {
    fn emit(&self, inputs: &[NodeId], graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId;
    fn output_dims(&self, input_w: u32, input_h: u32) -> Option<(u32, u32)> { ... }
    fn input_demands(&self, wu: &WorkUnit) -> Vec<(usize, WorkUnit)> { ... }
    fn scale_params_for_lod(&self, params: &[Param], lod: Lod) -> Vec<Param> { ... }
    fn input_encoders(&self, num_inputs: usize) -> Vec<InputEncoder> { ... }
    fn output_decoder(&self) -> OutputDecoder { ... }
    fn dispatch_grid(&self) -> DispatchGrid { ... }
}
```

`emit()` only builds the graph node — see [§3](#3-gpubackend-gpu--lazy-tiled) and
[§7](#7-gpu-operation-implementation-guide) for the full protocol and worked examples.

### Example: GaussianBlurOperation (demonstrates patterns A+C)

`GaussianBlurOperation` in `operation/filters.rs` implements BOTH `VipsOperation` AND `GpuOperation`:

**Vips path** — straightforward: sets "in" image, "sigma" double, optional "min_ampl" and "precision", calls `gaussblur`.

**GPU path**:
- `emit`: `emit_image(graph, inputs[0], self_arc, "ops.filters", "gaussian_blur_kernel", vec![Param::F32(self.sigma)])`
- `input_demands`: expands the demanded `Region.rect` by `radius = ceil(3*sigma / lod.scale_factor())` for the Gaussian kernel halo
- `output_dims`: identity (blur preserves dimensions)
- `scale_params_for_lod`: divides `sigma` (param index 0) by `lod.scale_factor()`

The fused emitter handles buffer allocation, color-space sandwich (`WorkingDecodeRegion`/`WorkingEncodeRegion`), coordinate-frame mapping, and dispatch — `emit()` itself never touches wgpu.

### How to choose which pattern

| What you're building | Use |
|---|---|
| Leverage existing libvips op | `VipsOperation` (Pattern A) |
| New pixel-wise op in pure Rust, output is an Image | `VipsCustomOperation` (Pattern B) |
| Scan/reduce image to a Rust value (stats, histograms) | `VipsCustomSink` (Pattern B) |
| GPU-accelerated operation | `GpuOperation` (Pattern C) |
| Operation that runs on both CPU and GPU | `VipsOperation` + `GpuOperation` (Patterns A+C) |

---

## 5. Color System

### Core types

```rust
pub struct ColorSpace {
    primaries: RgbPrimaries,
    white_point: WhitePoint,
    transfer: TransferFn,
}
```

`ColorSpace` is `Copy + Eq + Serialize + Deserialize`. Predefined constants cover all major standards (sRGB, Rec.2020, DCI-P3, ACES, ProPhoto, etc.).

### Matrix3x3

Column-major 3x3 matrix stored as `[[f32; 3]; 3]`. Key operations:
- `mul_vec_simd_x4(r, g, b, a)` — transform 4 pixels at once via `f32x4` SIMD
- `mul(other)` — matrix multiply
- `inverse()` — Gaussian elimination with partial pivoting (returns `Err` if singular)

### Color pipeline (XYZ D50 hub)

The standard color space conversion path uses XYZ D50 as the interchange hub:

```
Source RGB
  → TransferFn::decode()           (de-linearize: sRGB EOTF, PQ, etc.)
  → Color model decode             (if CMYK/YCbCr/Lab → linear RGB)
  → rgb_to_xyz_matrix()            (RGB primaries → XYZ D50)
  → bradford_cat(src_wp, D50)      (if source white point ≠ D50)
  → bradford_cat(D50, dst_wp)      (if destination white point ≠ D50)
  → rgb_to_xyz_matrix().inverse()  (XYZ D50 → destination RGB primaries)
  → Color model encode             (if destination is CMYK/YCbCr/Lab)
  → TransferFn::encode()           (re-linearize)
  → Destination RGB
```

The `rgb_to_rgb_transform()` function composes the full chain in one `Matrix3x3` (excluding transfer functions and model transforms, which are separate).

### Bradford chromatic adaptation

Converts between white points by transforming through the LMS cone response space:
```
B = | 0.8951   0.2664  -0.1614 |
    | -0.7502  1.7135   0.0367 |
    | 0.0389  -0.0685   1.0296 |
```

### Transfer functions

`TransferFn` is `#[repr(u8)]` for direct GPU passthrough. Each variant implements:
- `decode(x) -> f32` — EOTF (electro-optical transfer function)
- `encode(y) -> f32` — OETF (opto-electronic transfer function)
- `is_linear()` — Linear needs no conversion

Special cases:
- sRGB: piecewise (linear < 0.0031308, powered ≥ 0.0031308)
- PQ (ST 2084): constants m1=2610/16384, m2=2523/32, c1=3424/4096, c2=2413/128, c3=2392/128
- HLG: piecewise with ln/exp

### ColorSpace::convert() (GPU)

`Image2D<GpuBackend>::convert(target_meta)` is eager — it materializes the full image, builds the complete A/B matrix chain (source→XYZ D50→target including Bradford CAT if white points differ), creates `ColorConvertParams`, and dispatches the `ColorConvertParamsKernel`. A no-op check returns the source unchanged.

---

## 6. Pixel System

### Pixel trait

```rust
pub trait Pixel: Copy + Pod {
    fn unpack(self) -> [f32; 4];
    fn unpack_x4(s: &[Self]) -> (f32x4, f32x4, f32x4, f32x4);
    fn pack_x4(r: f32x4, g: f32x4, b: f32x4, a: f32x4, mode: AlphaPolicy, out: &mut [Self]);
    fn pack_one(rgba: [f32; 4], mode: AlphaPolicy) -> Self;
}
```

`unpack` converts to straight linear `[r, g, b, a]` — it:
- Normalizes integer types (u8→[0,1], u16→[0,1])
- Unpremultiplies f16/f32 RGBA (divides RGB by alpha)
- Sets alpha=1.0 for RGB types without alpha
- Converts Lab u8/u16 from centered ranges to [-1,1]

`pack` applies `AlphaPolicy`:
- `Straight`: write RGB and alpha as-is
- `PremultiplyOnPack`: multiply RGB by alpha before storing
- `OpaqueDrop`: premultiply RGB, discard alpha channel

### AlphaPolicy

```rust
pub enum AlphaPolicy {
    Straight,           // shader value: 0
    PremultiplyOnPack,  // shader value: 1
    OpaqueDrop,         // shader value: 2
}
```

The shader discriminant order (Straight=0, PremultiplyOnPack=1, OpaqueDrop=2) differs from the Rust enum order. `to_shader()` handles this explicitly — keep in sync with `lib/pixel.slang`.

### Component trait

```rust
pub trait Component: Copy + Pod + Zeroable {
    const ZERO: Self;
    const ONE: Self;
    const MAX_ONE_F32: f32;
    fn to_f32(self) -> f32;
    fn from_f32_clamped(v: f32) -> Self;
}
```

Implemented for `u8`, `u16`, `half::f16`, `f32`. Used by pixel types for type-generic pack/unpack.

### PixelFormat

36 variants covering all supported channel layouts and sample types. Each format has:
- `bytes_per_pixel()` — storage size
- `channel_count()` — number of channels
- `gpu_channel_layout()` — 0=Rgba, 1=Rgb, 2=Gray, 3=GrayA, 4=CmykA (runtime param for shader codec)
- `model_transform()` — `ColorModelTransform` (None, CmykToRgb, LabToRgb, etc.) — tells the shader which color model decode to apply before the RGB matrix

### PixelMeta

```rust
pub struct PixelMeta {
    pub format: PixelFormat,
    pub color_space: ColorSpace,
    pub alpha_policy: AlphaPolicy,
}
```

A compressed descriptor carrying everything needed to interpret raw pixel bytes. `Copy`. Used throughout the pipeline for buffer creation and color space conversion decisions.

---

## 7. GPU Operation Implementation Guide

Operations implement `GpuOperation` (+ `TypedOperation` to declare their output `DataType`)
to **build the graph** (not to execute). At materialize time the emitter reads the graph and
generates a fused shader. Reference implementations: `GaussianBlurOperation` (`operation/filters.rs`),
`AddOperation`/composite (`operation/arithmetic.rs`, `operation/composite.rs`), `HistogramOp`
(`operation/stats.rs`), `GpuColorConvertOperation` (`backend/gpu/op.rs`).

### Case A: Standard image-in / image-out op (most operations)

```rust
use crate::backend::gpu::datatype::ImageType;
use crate::backend::gpu::graph::{Graph, NodeId};
use crate::backend::gpu::op::{GpuOperation, TypedOperation, emit_image};
use crate::backend::gpu::param::Param;
use crate::backend::gpu::work_unit::WorkUnit;
use crate::backend::gpu::Lod;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct MyOperation { pub sigma: f32 }

impl TypedOperation for MyOperation {
    type Output = ImageType;
}

impl GpuOperation for MyOperation {
    fn emit(&self, inputs: &[NodeId], graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_image(graph, inputs[0], self_arc, "ops.my_module", "my_kernel", vec![
            Param::F32(self.sigma),
        ])
    }

    // Override only if output dimensions differ from input.
    fn output_dims(&self, w: u32, h: u32) -> Option<(u32, u32)> {
        Some((w, h))  // identity — this is the default
    }

    // Override if the kernel needs a halo (e.g. blur radius) around its demanded rect.
    fn input_demands(&self, wu: &WorkUnit) -> Vec<(usize, WorkUnit)> {
        match wu {
            WorkUnit::Region { rect, lod } => {
                let radius = (3.0 * self.sigma / lod.scale_factor() as f32).ceil() as i32;
                vec![(0, WorkUnit::Region { rect: rect.expand(radius), lod: *lod })]
            }
            _ => vec![(0, wu.clone())],
        }
    }

    // If sigma is in pixels (not normalized), scale it for reduced LODs:
    fn scale_params_for_lod(&self, params: &[Param], lod: Lod) -> Vec<Param> {
        let mut p = params.to_vec();
        if let Param::F32(sigma) = &mut p[0] {
            *sigma /= lod.scale_factor() as f32;
        }
        p
    }
}
```

`emit_image` creates a `GraphNode` with `eval: NodeEval::Kernel(KernelSpec { module, function })`
and `datatype: working_image_type()` (ACEScg `RgbaF32`). The emitter calls the Slang function
`my_kernel(idx, region_src_0, region_tmp_N, u0)` once per thread.

### Case B: Multi-input op (e.g. composite, arithmetic)

Op structs own ALL their typed inputs. `emit()` receives `inputs` for the implicit "self" image
(`inputs[0]`) and uses `splice_sibling` to lazily fuse each other operand's subgraph:

```rust
use crate::backend::gpu::graph::{Graph, GraphNode, KernelSpec, NodeEval, NodeId};
use crate::backend::gpu::op::{GpuOperation, TypedOperation, emit_image, splice_sibling, working_image_type};
use crate::backend::gpu::work_unit::WorkUnit;
use crate::data::image::Image2D;

#[derive(Debug, Clone)]
pub struct AddOperation<B: crate::backend::Backend> {
    pub right: Image2D<B>,
}

impl TypedOperation for AddOperation<crate::backend::gpu::GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for AddOperation<crate::backend::gpu::GpuBackend> {
    fn emit(&self, inputs: &[NodeId], graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);  // merges `right`'s subgraph if needed
        graph.add_node(GraphNode {
            id: NodeId(0),                    // overwritten by add_node
            inputs: vec![input, right_id],    // 0 = self (lhs), 1 = right
            eval: NodeEval::Kernel(KernelSpec { module: "ops.arithmetic", function: "add_kernel" }),
            params: vec![],
            op: self_arc,
            datatype: working_image_type(),
        })
    }

    fn input_demands(&self, wu: &WorkUnit) -> Vec<(usize, WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]  // both inputs need the same WorkUnit
    }
}
```

`splice_sibling` returns a `NodeId` valid in `graph` for `self.right`'s root — either `right`'s
own `root_id` (if it's already the same graph, e.g. compositing onto self) or the remapped id
after `Graph::merge_from`. The emitter generates `add_kernel(idx, region_src_0, region_src_1, region_tmp_N)`.
`Image2D::execute(op)` (via `Executable::execute` → `GraphBuilder::build`) is the entry point —
no separate "primary image" concept beyond `inputs[0]` being the receiver.

### Case C: Non-image output (e.g. histogram)

Use `emit_unary` with a non-image `Arc<dyn DataType>`. These nodes bypass the float4 temp system
(`needs_fused_temp() == false`) and write directly to their target buffer via `write_mode()`.

```rust
use crate::backend::gpu::datatype::HistogramType;
use crate::backend::gpu::op::{emit_unary, OutputDecoder};
use crate::backend::gpu::work_unit::WorkUnit;

impl TypedOperation for HistogramOp {
    type Output = HistogramType;
}

impl GpuOperation for HistogramOp {
    fn emit(&self, inputs: &[NodeId], graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_unary(graph, inputs[0], self_arc, "ops.histogram", "histogram_kernel",
            vec![Param::U32(self.channel)],
            Arc::new(HistogramType { bins: self.bins }))
    }

    fn output_dims(&self, _w: u32, _h: u32) -> Option<(u32, u32)> {
        None  // no spatial output
    }

    // Histogram is Atomic — dispatch grid comes from the *input* (Input(0)), not this output.
    fn input_demands(&self, _wu: &WorkUnit) -> Vec<(usize, WorkUnit)> {
        vec![(0, WorkUnit::Atomic)]
    }

    fn output_decoder(&self) -> OutputDecoder {
        OutputDecoder::HistogramOut
    }

    fn dispatch_grid(&self) -> DispatchGrid {
        DispatchGrid::Input(0)
    }
}
```

To pull the result, `HistogramTargetCapability::create_histogram(&handle)` calls
`HistogramType::execute(&op, &handle)` (via `Image2D::histogram()`), then `pull_histogram` calls
`HistogramType { bins }.pull(&handle, lod, &Atomic)` — `Targetable::pull` is blanket-implemented for
every `TypedData` and wraps `GpuRequest::materialize()`. See `HistogramOp` in `operation/stats.rs`,
`HistogramTargetCapability for GpuBackend` in `backend/gpu/target.rs`, and `Targetable`/`GpuRequest`
in `datatype/mod.rs`/`request.rs`.

### Case D: Output metadata override (color space conversion)

There is no `dst_meta` side channel. To override the output color space/format, give the emitted
node an `ImageType { color_space, format }` **datatype carrying the target metadata**, paired with
an `output_decoder()` codec for the same target — this is exactly `GpuColorConvertOperation`
(`backend/gpu/op.rs`):

```rust
fn emit(&self, inputs: &[NodeId], graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
    graph.add_node(GraphNode {
        id: NodeId(0),
        inputs: vec![inputs[0]],
        eval: NodeEval::Kernel(KernelSpec { module: "ops.passthrough", function: "passthrough_kernel" }),
        params: vec![],
        op: self_arc,
        datatype: Arc::new(ImageType { color_space: self.dst.color_space, format: self.dst.format }),
    })
}

fn output_decoder(&self) -> OutputDecoder {
    OutputDecoder::WorkingEncodeRegion {
        codec: Some(OutputCodec { color_space: self.dst.color_space, format: self.dst.format }),
    }
}
```

`Graph::resolve_image_type(root_id)` reads the root node's `ImageType` datatype to answer
`Image2D::format()`/`color_space()` (via `ColorConversionCapability::pixel_meta`) — so the
override is visible to callers immediately, with no separate metadata field to keep in sync.

### Shader side (Slang)

Kernel signatures follow the `IRegion` interface pattern established in `shaders/lib/region.slang`:

```slang
// ops/my_module.slang
import "lib/working";

public void my_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float sigma) {
    float4 pixel = input.read(idx);   // decoded to ACEScg linear float4
    // ... process in working space ...
    output.write(idx, pixel);         // stays in working space (emitter adds final encode)
}
```

`input.read()` goes through `WorkingDecodeRegion` (decode + `to_working()` color convert). The final target encode (`from_working()` + codec `encode`) is emitted automatically for the last node in the chain. Intermediate nodes write raw working-space `float4` to their temp buffer — no extra color conversion.

Param values arrive as `ChainParams.u0`, `ChainParams.u1`, etc. (positional, not named). Region descriptors arrive as `ChainParams.inputs_0`, `ChainParams.temp_region_0`, `ChainParams.region_target_0`.

### Key rules

1. **Working space is sRGB linear (not ACEScg).** `GpuPixelEncoding::from_meta` converts to/from sRGB hub. All temp buffers hold sRGB linear `float4`. Color-space-sensitive ops must apply their own transform if they need a different working space.
2. **`emit()` only builds the graph.** Never call wgpu APIs inside `emit()`. All GPU work happens later in `materialize.rs`.
3. **`input_demands()` must be conservative.** Return a `WorkUnit` (rect) that fully covers the source pixels the kernel reads. Over-fetching is safe; under-fetching produces black/clamped pixels at the edges.
4. **LOD-scaled params** — if a param is in pixel units (e.g. blur radius), override `scale_params_for_lod()` to divide it by `lod.scale_factor()` before dispatch.
5. **Cross-backend test is mandatory.** Add a test in `tests/cross_backend.rs` comparing GPU output to the Vips reference within the tolerance guidelines in §8.

---

## 8. Cross-Backend Validation

### Convention

Every new GPU operation MUST pass a cross-backend test in `tests/cross_backend.rs`. The test MUST:
1. Run the operation on both `Image2D<VipsBackend>` and `Image2D<GpuBackend>` with identical parameters and the same source image
2. Compare outputs within a reasonable tolerance

### Tolerance guidelines

- **Float linear ops** (F32 in 0..1 range): RMS < 0.001 typically
- **U8 quantized ops** (0..255): RMS < 2-4 LSBs (quantization accumulates)
- **Round-trips** (convert + inverse): RMS < 2.0 for two quantizations
- **Identity no-ops** (same meta convert): must be EXACT match (RMS = 0.0)
- **Edge pixels**: Excluded from comparison (interior-only RMS). Vips uses EXTEND, GPU uses CLAMP — edge behavior differs

### Known limitations (tracked separately)

- **Sub-word pixel formats** (Rgb8, Gray8): GPU uses `RWStructuredBuffer<uint>` with atomic read-modify-write. Slang `Atomic<uint>.and()/.or()` does not emit real `OpAtomic` (no store), so sub-word packing is broken. Rgba8 (4 bytes = whole word) works fine.

### Common color space matrix

For ops that work in a canonical space (blur, composite), the transform chain is always:
```
source color space → ACEScg (AP1 linear) → source color space
```
The forward and inverse matrices should be inverses of each other (modulo floating-point error). A blur on sRGB data and the same blur on Rec.2020 data should produce visually identical results in their respective spaces.

---

## 9. Directory Reference — where to put things

| What | Where |
|---|---|
| New pixel format | `pixel/format.rs` (add variant), `pixel/<name>.rs` (impl Pixel), update `PixelFormat::from()` in `dispatch.rs` |
| New vips operation | `operation/<category>.rs`, implement `VipsOperation`, add pub use in `operation/mod.rs` |
| New vips-only operation | Same as above, no `GpuOperation` impl needed |
| New GPU operation | `operation/<category>.rs` (or new file), implement BOTH `VipsOperation` + `GpuOperation`, add cross-backend test |
| New GPU kernel params | `pixors-shader` crate — define params struct with `#[kernel]` attribute, shader code in `pixors-shader/shaders/` |
| New color space constant | `color/space.rs` |
| New transfer function | `color/transfer.rs` (add variant + decode/encode impls) |
| New primaries/white point | `color/primaries.rs` (add variant + chromaticities) |
| New generator | `generator.rs` — implement `GenerateOperation` |
| New draw operation | `draw.rs` — implement `DrawOperation` |
| New backend | `backend/<name>/mod.rs` with `Backend` impl + capability traits |
| Backend-specific enum mapping | `vips.rs` (for Vips), `<backend>/mod.rs` (for others) |

---

## 10. Edge Cases and Invariants

### Memory safety

- **VipsHandle Drop**: `g_object_unref` must be called exactly once per reference. Clone increments, Drop decrements. Never call free/g_free on a VipsImage.
- **VipsRegion fetch buffer**: `vips_region_fetch` returns a pointer that must be freed with `g_free`. The Region wrapper handles this.
- **GraphNodeHandle is Send+Sync**: Both `Graph` (`Arc<Mutex<Graph>>`) and `GpuContext` (`Arc<GpuContext>`) are thread-safe.

### VipsGObject invariants

- **Output extraction BEFORE unref_outputs**: Always extract GObject properties before calling `vips_object_unref_outputs`.
- **Thread-local VIPS_THREAD**: `VipsGObject` creates a `VIPS_THREAD` guard that registers `vips_thread_shutdown` on drop. This ensures thread-local cleanup.

### GPU invariants

- **Format consistency**: All GpuOperations currently assume input and output have the same `PixelFormat`. Heterogeneous format ops are TODO.
- **Context sharing**: All images derived from the same source share the same `Arc<GpuContext>`. Cross-context operations are not supported.
- **Cache keying**: `GpuContext::cache` (`TieredCache`) is keyed by `(content_hash, x, y, w, h)`, where `content_hash = Graph::content_hash(root_id)` is a structural hash of the computation — independent of `NodeId`. Two handles at different `root_id`s that compute the same thing (e.g. after `fork()`) share cache entries; semantically different computations never collide even if `root_id`s coincide across graphs.
- **Buffer word alignment**: `ImageBuffer::alloc()` rounds size up to u32 alignment because shaders address buffers as `RWStructuredBuffer<uint>`.
- **Graph mutation is Mutex-guarded**: The `Graph` inside `GraphNodeHandle` is behind `Arc<Mutex<Graph>>`. Lock it for both read and write. `emit()` is called with the lock held. `splice_sibling`/`GraphBuilder::build` use `try_lock`/`Arc::ptr_eq` to detect same-graph siblings and avoid self-deadlock.
- **`GpuOperation` is stored on the node**: `GraphNode.op` keeps the op alive for `input_demands` calls during the materialize walk. Always pass `self_arc` into the node when implementing `emit()`.

### Error handling

- All GPU errors use `Error::Vips(String)`. This is historical — should eventually become `Error::Gpu(String)` or similar.
- `fetch_region()` on a Vips source can fail if the requested rect is outside image bounds. Op chains should clamp rects before passing to source fetch.
- Pipeline compile failures (missing SPIR-V, layout mismatches) are not gracefully handled — they panic via unwrap on the lock.

---

## 11. Slang C FFI and RAII layer (`backend/gpu/slang.rs`)

### Current C API (slang_wrapper.cpp/.h)

| Function | Description |
|---|---|
| `slangw_create_global_session()` | Creates the process-level global session (thread-unsafe without guard) |
| `slangw_global_session_release(void* gs)` | Explicit release of the global session |
| `slangw_create_session(gs, paths, count, SlangwOptLevel opt_level)` | Creates a per-compilation session; takes `SlangwOptLevel` (always use `SLANGW_OPT_MAXIMAL`) |
| `slangw_compile_to_spirv(session, name, source, source_len, target_idx, out_code, out_size, out_diag)` | Load module + compile to SPIR-V in one call; module stays internal to C++ — never crosses the FFI boundary |
| `slangw_free_buffer(void*)` | Frees a SPIR-V output buffer |
| `slangw_free_string(char*)` | Frees diagnostic strings (different allocator from SPIR-V buffer) |
| `slangw_release(void*)` | Releases an `ISlangUnknown` — sessions only; **NOT modules** |
| `slangw_shutdown()` | Global Slang shutdown |

**Removed (do not use):**
- `slangw_create_session_spirv(gs, paths, count)` — replaced by `slangw_create_session`
- `slangw_load_module_from_source(session, name, src, len)` — exposed borrowed module pointer unsafely; removed
- `slangw_load_module(session, name, src, src_len, out_diag)` — same problem; removed
- `slangw_get_spirv(module, target_idx, out_code, out_size, out_diag)` — replaced by `slangw_compile_to_spirv`
- `slangw_module_find_and_get_entry_point_code(...)` — legacy; removed

### `SlangwOptLevel` enum

```c
typedef enum {
    SLANGW_OPT_NONE    = 0,
    SLANGW_OPT_DEFAULT = 1,
    SLANGW_OPT_HIGH    = 2,
    SLANGW_OPT_MAXIMAL = 3,
} SlangwOptLevel;
```

Default level used everywhere: **`SLANGW_OPT_MAXIMAL`**.

### Rust RAII wrappers (`backend/gpu/slang.rs`)

```rust
struct GlobalSession(NonNull<c_void>);  // Drop: slangw_global_session_release
struct Session(NonNull<c_void>);         // Drop: slangw_release
struct SpirvBuf(*const c_void, usize);  // Drop: slangw_free_buffer
struct DiagString(*mut c_char);          // Drop: slangw_free_string
// Module struct REMOVED — module is a borrowed pointer, must not be released
```

**Global session storage:** `static GLOBAL_SESSION: Mutex<Option<GlobalSession>>` — process singleton, lazy init. Slang session creation is not thread-safe without the mutex guard.

**`SlangCompiler` session:** holds `session: Mutex<Option<Session>>` — persistent session per compiler instance, lazy init. Created once with `shader_dir` + `out_dir` as search paths and reused across all compilations (eliminates ~10–50 ms per call after the first).

**Module naming:** module name = `format!("{hash_val:016x}")` (hex hash of the source, NOT `"main"`). This ensures the Slang session module cache never returns a stale cached module for a different source.

### Memory management invariant (critical)

`IModule*` returned by `loadModuleFromSource` is a **borrowed pointer owned by the session** — it is NOT addref'd for the caller. Calling `release()` on it decrements the refcount to 0, freeing it while the session still holds a reference → use-after-free → heap corruption. The fix: the module never crosses the FFI boundary. `slangw_compile_to_spirv` handles load + compile entirely in C++.

```rust
// DO NOT DO THIS — module is borrowed, not owned:
slangw_release(module_ptr);  // UAF! session heap corruption

// Correct: module stays internal to slangw_compile_to_spirv
```

### Performance

- Session creation savings: ~10–50 ms eliminated per compile after the first call (session is reused).
- Benchmark labels (via `crate::utils::Stopwatch` from `pixors-engine/src/utils.rs`):
  - `gpu.compile.slang` — time inside `slangw_compile_to_spirv` (holds session lock)
  - `gpu.compile.opt` — time for spirv-tools optimization (outside lock, parallelisable)

### Error messages

Compilation failures include the actual Slang diagnostic text (e.g. `"undefined variable 'x' at line 42"`) instead of opaque `rc=-3` codes. The `DiagString` wrapper owns the string and frees it on Drop.

---

## 12. Dependencies (minimal external surface)

- **libvips** (via FFI/bindgen): VipsBackend only. No vips dependency in color/pixel/generator/draw modules
- **wgpu 26** (spirv feature): GpuBackend only. Used by `GpuContext`, `GpuBuffer`/`ImageBuffer`, `dispatch_kernel`
- **pixors-shader** (workspace): Provides `GpuKernel` trait, kernel types (`BlurParamsKernel`, `ComposeParamsKernel`, `ColorConvertParamsKernel`), and shader param structs (`BufferRegion`, `ColorSpace`, `Matrix3`)
- **bytemuck**: For Pod/Zeroable on params structs passed to GPU
- **rayon**: Parallel iteration in `apply_ops()` for multi-input ops
- **wide**: `f32x4` SIMD in color matrix and pixel pack/unpack
- **pollster**: Blocking async for `GpuContext::new()` (wgpu adapter request)

No dependency on any other Pixors crate.
