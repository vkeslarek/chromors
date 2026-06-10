# AGENTS.md — pixors-engine

## Build

```
cargo build
cargo test
```

## Requirements

- libvips >= 8.6 (dev headers)
- pkg-config
- Rust 1.94+
- GPU: Vulkan/Metal/DX12 (wgpu 26)

## Architecture

```
src/
├── lib.rs              # init(), re-exports, ffi include!
├── error.rs            # Error enum
├── backend/
│   ├── mod.rs          # Backend, Operation, SourceInput, TargetOutput, ImageTargetCapability, ColorConversionCapability
│   ├── gpu/
│   │   ├── mod.rs      # GpuBackend, GraphNodeHandle, Lod, RegionCache, re-exports
│   │   ├── datatype/   # DataType, TypedData, Sourceable, Targetable, Executable
│   │   │   ├── mod.rs      # the cross-cutting traits (see "The datatype model")
│   │   │   ├── image.rs    # ImageType
│   │   │   ├── histogram.rs# HistogramType
│   │   │   ├── mask.rs     # Mask1dType, Mask2dType
│   │   │   ├── fft.rs      # Fft1dType, Fft2dType
│   │   │   └── reduction.rs# ScalarType, PointListType, FeaturesType
│   │   ├── typed/      # host-facing typed wrappers around GraphNodeHandle
│   │   │   ├── image.rs     # Image2D<GpuBackend> inherent impls
│   │   │   └── histogram.rs # Histogram<GpuBackend> inherent impls
│   │   ├── value.rs    # MaterializedValue, Storage, WriteMode
│   │   ├── work_unit.rs# Region, Range, Atomic, AnyWorkUnit, WorkUnitKind, WorkUnit
│   │   ├── graph.rs    # Graph, GraphNode, SourceNode, KernelSpec, NodeId, NodeEval, content_hash
│   │   ├── op.rs       # GpuOperation/TypedOperation traits, InputEncoder/OutputDecoder/DispatchGrid, emit_image/emit_unary/splice_sibling
│   │   ├── builder.rs  # GraphBuilder::build — standalone multi-input node construction
│   │   ├── request.rs  # GpuRequest<D: TypedData> — typed materialization request
│   │   ├── cache.rs    # TieredCache (VRAM→RAM→Disk, content-hash keyed, CLOCK eviction)
│   │   ├── emit.rs     # JIT Slang emitter, LayoutPlan, alloc_temps, emit_slang
│   │   ├── compile.rs  # Compiled (SPIR-V + wgpu pipelines)
│   │   ├── materialize.rs # materialization pipeline, execute_batch
│   │   ├── region.rs   # GpuRegion (prepare/materialize/materialize_batch)
│   │   ├── buffer.rs   # GpuBuffer (payload-agnostic VRAM), ImageBuffer (image-shaped wrapper)
│   │   ├── context.rs  # GpuContext (wgpu device + queue + pipeline cache + RegionCache)
│   │   ├── source.rs   # GpuSource (Image2D | VipsImage via enum_dispatch), AnyGpuSource, fetch_region
│   │   ├── pass.rs     # CutFinder (device storage-buffer-limit enforcement)
│   │   ├── param.rs    # Param enum, GpuPixelEncoding
│   │   └── arena.rs    # buffer pool
│   └── vips/
│       ├── mod.rs      # VipsBackend, VipsHandle, IntoVipsEnum
│       ├── gobject.rs  # VipsGObject, Runner trait
│       ├── region.rs   # Region (vips_region_prepare/fetch)
│       ├── source.rs   # Source (vips_source_new_from_file/blob)
│       ├── target.rs   # Target (vips_target_new_to_file/memory)
│       └── custom.rs   # VipsCustomOperation, VipsCustomSink
├── color/
│   ├── space.rs        # ColorSpace { primaries, white_point, transfer }
│   ├── matrix.rs       # Matrix3x3, rgb_to_rgb_transform, bradford_cat
│   ├── primaries.rs    # RgbPrimaries + chromaticities
│   └── transfer.rs     # TransferFn (SRGB, PQ, HLG, …) + decode/encode
├── data/
│   ├── image.rs        # Image2D<B: Backend> — phantom-typed handle
│   └── histogram.rs    # Histogram<B> handle, HistogramResult
├── target.rs           # ImageTarget, HistogramTarget, MaterializedImage, MaterializedHistogram
├── operation/
│   ├── composite.rs    # Composite2Operation (vips) + multi-input GPU composite
│   ├── filters.rs      # GaussianBlurOperation (vips + GPU reference impl)
│   ├── stats.rs        # HistogramOp (GPU non-image output reference impl)
│   ├── arithmetic.rs   # AddOperation etc. (multi-input GPU reference impl via splice_sibling)
│   └── …               # bands, geometry, icc, opacity, edge, …
└── pixel/
    ├── format.rs       # PixelFormat (38 variants)
    └── meta.rs         # PixelMeta { format, color_space, alpha_policy }
```

## Key design rules

### The datatype model

Every `GraphNode` carries `datatype: Arc<dyn DataType>` — the open vocabulary that
describes what a node produces. There is no closed enum to edit when adding a
new datatype; write a struct in `backend/gpu/datatype/<name>.rs` and implement
the traits below.

```rust
pub trait DataType: Send + Sync + Debug + 'static {
    fn as_any(&self) -> &dyn std::any::Any;
    fn needs_fused_temp(&self) -> bool { false }            // only ImageType -> true
    fn write_mode(&self) -> WriteMode { WriteMode::Positional } // HistogramType -> AtomicAccumulate{count}
    fn byte_size(&self, w: u32, h: u32, image_format: PixelFormat) -> u64;
    fn work_unit_kind(&self) -> WorkUnitKind;               // Region | Range | Atomic
}

pub trait TypedData: DataType + Sized {
    type Value: Clone + Send + Sync;
    type WorkUnit: AnyWorkUnit;
    fn finish(&self, value: &MaterializedValue, lod: Lod, wu: &Self::WorkUnit, ctx: &GpuContext)
        -> Result<Self::Value, Error>;
}
```

Concrete datatypes (`backend/gpu/datatype/*.rs`): `ImageType { color_space, format }`
(the only one with `needs_fused_temp() == true` and `work_unit_kind() == Region`),
`HistogramType { bins }` (`work_unit_kind() == Atomic`, `write_mode() ==
AtomicAccumulate{count: bins}`), `Mask1dType`/`Mask2dType`, `Fft1dType`/`Fft2dType`,
`ScalarType`/`PointListType`/`FeaturesType`.

Capability traits layer on top of `DataType`:

- `Sourceable: DataType` — can be a graph leaf (`fetch_region` from a `GpuSource`).
  Today only `ImageType` implements it.
- `Targetable: TypedData + Clone` — blanket-impl'd for every `TypedData`; `pull(&self,
  node, lod, wu)` wraps `GpuRequest::new(...).materialize()`.
- `Executable: DataType + Sized` — blanket-impl'd; `execute::<O>(op, node)` calls
  `GraphBuilder::build(op, &[node])`.

`WorkUnitKind` (`work_unit.rs`) is the *shape* of a datatype's natural division —
`Region` (2-D rect + `Lod`), `Range` (1-D `[start, end)`), or `Atomic` (indivisible).
The typed structs `Region`/`Range`/`Atomic` implement `AnyWorkUnit` to convert to/from
the erased `WorkUnit` enum used wherever the graph crosses heterogeneous node boundaries.

### MaterializedValue — runtime payload

```rust
pub struct MaterializedValue {
    pub storage: Storage,            // Vram(Arc<GpuBuffer>) | Host(Vec<u8>)
    pub datatype: Arc<dyn DataType>,
    pub extent: WorkUnit,
}
```

No embedded `kind` re-validation — the typed `GpuRequest<D>` already knows `D`, so
`TypedData::finish` decodes `storage` directly via `self`.

### The node evaluation model

`GraphNode.eval: NodeEval` is the evaluation strategy:
```rust
pub enum NodeEval {
    Kernel(KernelSpec),  // fused Slang function call — the common case today
    // Future: View, Reduction, Host
}
```

Breaking the old "every node == one kernel" coupling into an enum makes it possible to add no-dispatch view nodes (channel-level fusion) and CPU host-op nodes (feature extraction, alignment) without changing the core emit logic.

### Backends are generic, models are concrete

- `Image2D<B: Backend>` = phantom-typed handle, only holds `B::Handle`
- `Image2D<VipsBackend>` has vips-specific methods; `Image2D<GpuBackend>` has GPU-specific methods (`width`/`height`/`format`/`color_space`/`graph`/`root_id`/`fork`/`execute`, see `backend/gpu/typed/image.rs`)
- For `GpuBackend`, `B::Handle = GraphNodeHandle` — a neutral `{ graph, root_id, ctx }` triple with no image-specific fields. Image metadata is derived on demand from the root node's `Arc<dyn DataType>` (downcast to `ImageType`)
- Capability traits gate which methods compile: `OpenFile`, `OpenBuffer`, `SourceInput`, `TargetOutput`, `ImageTargetCapability`, `HistogramTargetCapability`, `ColorConversionCapability`. `HistogramTargetCapability::pull_histogram` is a thin wrapper over `Targetable::pull` (`HistogramType::pull`) — new non-image datatypes follow this same wrap pattern only if they need a `data/<name>.rs` handle exposed off `Image2D`
- `Operation<Input> { type Output; }` — typed output, not hardcoded to Image. The Vips path already returns `f64`, `Bounds`, `Histogram`, etc. via this mechanism

### Vips is the reference

Every GPU operation MUST produce the same result as the equivalent vips operation within the tolerances in §8 of CLAUDE.md. Cross-backend tests live in `tests/cross_backend.rs`. If GPU and vips diverge, fix the GPU path.

## How to add a new Image→Image GPU operation

1. **Define op struct** in `operation/<category>.rs`:
   ```rust
   #[derive(Debug, Clone)]
   pub struct MyOperation { pub amount: f32 }
   ```

2. **Implement VipsOperation** (CPU reference):
   ```rust
   impl VipsOperation for MyOperation {
       type Output = crate::Image2D<VipsBackend>;
       fn name() -> &'static [u8] { b"my_op_name\0" }
       fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
           op.set_image("in", image);
           op.set_double("amount", self.amount as f64);
       }
   }
   ```

3. **Implement GpuOperation** (GPU graph builder):
   ```rust
   impl TypedOperation for MyOperation {
       type Output = ImageType;
   }

   impl GpuOperation for MyOperation {
       fn emit(&self, inputs: &[NodeId], graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
           emit_image(graph, inputs[0], self_arc, "ops.my_module", "my_kernel",
               vec![Param::F32(self.amount)])
       }
       // If `amount` is in pixel units (e.g. blur sigma), scale it for LOD:
       fn scale_params_for_lod(&self, params: &[Param], lod: Lod) -> Vec<Param> {
           let mut p = params.to_vec();
           if let Param::F32(v) = &mut p[0] {
               *v /= lod.scale_factor() as f32;
           }
           p
       }
   }
   ```

4. **Write Slang shader** in `pixors-shader/shaders/ops/my_module.slang`:
   ```slang
   import "lib/working";
   public void my_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float amount) {
       float4 c = input.read(idx);  // decoded to linear sRGB float4
       // ... transform c using amount ...
       output.write(idx, c);
   }
   ```

5. **Add cross-backend test** in `tests/cross_backend.rs` (compare GPU vs Vips within tolerance).

## How to add a non-image output operation (e.g. histogram, scalar stat)

1. Define op struct. Implement `VipsOperation` returning the typed result via `Runner`.
2. Implement `GpuOperation::emit` using `emit_unary` with a non-image `DataType`, and override
   `output_dims`/`input_demands`/`output_decoder`/`dispatch_grid` as needed:
   ```rust
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
       fn input_demands(&self, _wu: &WorkUnit) -> Vec<(usize, WorkUnit)> {
           vec![(0, WorkUnit::Atomic)]  // histogram needs the whole input
       }
       fn output_decoder(&self) -> OutputDecoder {
           OutputDecoder::HistogramOut
       }
       fn dispatch_grid(&self) -> DispatchGrid {
           DispatchGrid::Input(0)  // thread grid covers the input being scanned
       }
   }
   ```
3. Expose a typed handle (e.g. `Histogram<B>`) whose backend capability (e.g.
   `HistogramTargetCapability::pull_histogram`) wraps `Targetable::pull` (blanket-impl'd for any
   `TypedData`) — `GpuRequest::new(...).materialize()` drives `HistogramType::finish` to
   decode the `MaterializedValue`. See `HistogramOp` in `operation/stats.rs`,
   `HistogramTargetCapability for GpuBackend` in `backend/gpu/target.rs`, and `HistogramBuffer`
   in `backend/gpu/typed/histogram.rs`.

## How to add a multi-input GPU operation (e.g. composite, arithmetic)

The op struct owns its extra inputs as `Image2D<B>` fields; `emit` receives `inputs: &[NodeId]`
(no privileged "primary" — `inputs[0]` is whichever handle the caller passed first) and splices
sibling subgraphs with `splice_sibling`:
```rust
pub struct AddOperation<B: Backend> {
    pub right: Image2D<B>,
}

impl TypedOperation for AddOperation<GpuBackend> {
    type Output = ImageType;
}

impl GpuOperation for AddOperation<GpuBackend> {
    fn emit(&self, inputs: &[NodeId], graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let input = inputs[0];
        let right_id = splice_sibling(graph, &self.right);
        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: vec![input, right_id],
            eval: NodeEval::Kernel(KernelSpec { module: "ops.arithmetic", function: "add_kernel" }),
            params: vec![],
            op: self_arc,
            datatype: working_image_type(),
        })
    }
    fn input_demands(&self, wu: &WorkUnit) -> Vec<(usize, WorkUnit)> {
        vec![(0, wu.clone()), (1, wu.clone())]  // both inputs, same work unit
    }
}
```
`splice_sibling` merges `self.right`'s graph into the host graph via `Graph::merge_from` (or reuses
`right.root_id()` directly if it's the same graph, e.g. compositing an image onto itself — detected
via `try_lock`). For an op with no privileged host input at all, use `GraphBuilder::build(op,
&[handle_a, handle_b])` instead of `Image2D::execute`.

## How to override output metadata (color space / format)

Emit a node whose `datatype` is an `ImageType { color_space, format }` carrying the *target*
metadata, with `output_decoder()` returning `WorkingEncodeRegion { codec: Some(OutputCodec{...}) }`
so the final `from_working()` encode converts to that space. There is no separate override field on
`GraphNode` — the datatype IS the override. See `GpuColorConvertOperation` in `backend/gpu/op.rs`,
used by `Image2D<GpuBackend>::convert(meta)` (via `ColorConversionCapability`).

## How to add a new typed data handle (e.g. `Features<B>`)

Pattern mirrors `Histogram<B>` in `data/histogram.rs` / `backend/gpu/typed/histogram.rs`:
1. Add `FeaturesType { channels: u32 }` in `backend/gpu/datatype/reduction.rs` implementing
   `DataType` + `TypedData` (and `Sourceable` only if it can be a graph leaf).
2. Define `pub struct Features<B: Backend>(B::Handle, PhantomData<B>)` in `data/features.rs`.
3. Add inherent `pull()`/accessors in `backend/gpu/typed/features.rs` — for `GpuBackend`,
   `Targetable::pull` (blanket-impl'd) handles materialize + `FeaturesType::finish` decode.
4. Add `FeaturesTarget<B>` + `MaterializedFeatures<B>` in `target.rs` if a `TargetOutput` impl is needed.

## GPU optimization mechanisms

### Graph fusion (no per-op dispatch)
All nodes whose `datatype.needs_fused_temp()` is true (today: `ImageType`) are fused into **one Slang shader, one `queue.submit()`**. Intermediate `float4` results live in temp buffers (`RWRegion`) with slot reuse via interval coloring (`alloc_temps`). Non-image nodes (histograms, …) write directly to their target with no temp. No round-trips to CPU between ops.

### Source fetch coalescing (region merge)
`merge_overlapping(source_rects)` in `materialize.rs` merges all source rects needed for a batch into fewer, larger rectangles before calling into Vips. One merged fetch = one libvips region prepare+fetch = one CPU→GPU upload. GPU efficiency scales with payload size, so batching pixels amortizes the transfer cost.

### Device buffer-limit cuts (staging)
`CutFinder` in `pass.rs` walks depth-first from the root node, counting how many storage buffer bindings each pass would require. When the count would exceed the device's storage-buffer limit, it marks cut nodes as `StagingCuts`. These are pre-materialized and re-injected as `ImageBufferSource` nodes, keeping each shader pass within hardware limits. The cut adds a GPU→staging→GPU round-trip only at the cut boundary, not for every node.

### Pipeline caching (no re-compile)
`GpuContext.pipeline_cache` (keyed by Slang text hash) stores compiled `wgpu::ComputePipeline`s. The emitter uses **positional names** (not NodeId-based) so structurally identical graphs produce identical Slang text → cache hit even after the graph has grown with new operations upstream.

### LOD scaling (cheap preview without re-parameterization)
Pixel-space params (blur sigma, etc.) are scaled by `GpuOperation::scale_params_for_lod(params, lod)` (default no-op) before dispatch — typically dividing by `lod.scale_factor()`. The graph structure is re-used at every LOD; only param bytes change.

### Content-hash caching + coordinate frames
`TieredCache` (`cache.rs`, VRAM→RAM→Disk, CLOCK eviction) keys entries on `(Graph::content_hash(root), x, y, w, h)` — a structural hash of source identity + each node's kernel/params/output decoder/datatype, independent of `NodeId`. Identical computations hit the cache across graph forks and sessions. A cached `MaterializedValue`'s VRAM storage may be a sub-region of a larger buffer (from a merged fetch or cache hit); `WorkUnit::resolve(w, h)` plus `BufferRegion { stride, x, y, width, height }` params let shaders work in buffer-local coords without repacking.

## Three Vips operation styles

**VipsOperation** — wraps libvips native ops through the GObject API. Blanket impl of `Operation<VipsBackend>` via `Runner` trait. Use for anything libvips provides natively.

**VipsCustomOperation** — pure Rust code inside the libvips demand-driven pipeline. `generate(out, input)` called per output region. No full-image download. Use for custom pixel-wise transforms.

**VipsCustomSink** — reduction via `vips_sink`. `fold(acc, region)` per region (from vips threadpool), `merge(total, part)` to combine, `finish(acc) -> Output`. Use for stats, histograms, any reduction. Wrap in `Reduce<S>` for `Image::execute` compatibility.

## Directory reference

| What | Where |
|---|---|
| New datatype | `backend/gpu/datatype/<name>.rs` — new struct implementing `DataType` (+ `TypedData`, `Sourceable` if it's a graph leaf), re-export from `datatype/mod.rs` |
| New GPU operation | `operation/<category>.rs` — implement `VipsOperation` + `GpuOperation`/`TypedOperation`, add to `operation/mod.rs`, add cross-backend test |
| New Slang shader | `pixors-shader/shaders/ops/<name>.slang`, trigger compile with `cargo build -p pixors-shader` |
| New typed data handle | `data/<name>.rs` (handle struct), `backend/gpu/typed/<name>.rs` (inherent `pull`/accessors via `Targetable`), `target.rs` (Target + Materialized wrapper if `TargetOutput` needed) |
| New pixel format | `pixel/format.rs` (variant + `channel_count`, `bytes_per_pixel`), `pixel/<name>.rs` (impl Pixel) |
| New color space | `color/space.rs` (constant), `color/primaries.rs` / `color/transfer.rs` if new primaries/TF needed |
| New Vips custom op | `operation/custom_ops.rs` — implement `VipsCustomOperation` or `VipsCustomSink` |

## Slang C FFI (backend/gpu/slang.rs)

The Slang compilation layer uses a thin C wrapper (`slang_wrapper.cpp/.h`) and Rust RAII types.

**Current C API:**
- `slangw_create_global_session()` — process-level singleton
- `slangw_global_session_release(gs)` — explicit release (needed for clean shutdown)
- `slangw_create_session(gs, paths, count, opt_level)` — per-compilation session; takes `SlangwOptLevel` (always use `SLANGW_OPT_MAXIMAL`)
- `slangw_compile_to_spirv(session, name, source, source_len, target_idx, out_code, out_size, out_diag)` — load module + compile to SPIR-V in one call; module stays internal to C++ and never crosses the FFI boundary
- `slangw_free_buffer(void*)` — free SPIR-V buffer
- `slangw_free_string(char*)` — free diagnostic string (different allocator from SPIR-V buffer)
- `slangw_release(void*)` — release session only — **NOT modules**
- `slangw_shutdown()` — global shutdown

**Rust RAII wrappers** in `backend/gpu/slang.rs`:
```rust
struct GlobalSession(NonNull<c_void>);  // Drop → slangw_global_session_release
struct Session(NonNull<c_void>);         // Drop → slangw_release
struct SpirvBuf(*const c_void, usize);  // Drop → slangw_free_buffer
struct DiagString(*mut c_char);          // Drop → slangw_free_string
// Module struct REMOVED — module is borrowed, must not be released
```

**Critical memory invariant:** `IModule*` from `loadModuleFromSource` is a borrowed pointer owned by the session — NOT addref'd for the caller. Releasing it decrements the refcount to 0, freeing memory the session still holds → use-after-free. Fix: `slangw_compile_to_spirv` handles load + compile entirely in C++; the module never crosses the FFI boundary.

**Session persistence:** `SlangCompiler` holds `session: Mutex<Option<Session>>` — created once with `shader_dir` + `out_dir` as search paths and reused across all compilations (~10–50 ms saved per call after the first). Module name = `format!("{hash_val:016x}")` (hex hash of source) — prevents stale cache hits.

`GLOBAL_SESSION: Mutex<Option<GlobalSession>>` guards process-level singleton creation (Slang is not thread-safe during init). Error messages from failed compilations include the Slang diagnostic text instead of opaque `rc=-N` codes.

**Removed (do not use):** `slangw_create_session_spirv`, `slangw_load_module_from_source`, `slangw_load_module`, `slangw_get_spirv`, `slangw_module_find_and_get_entry_point_code`.

## Code style

- `cargo fmt --all` before commit
- `cargo clippy --workspace` before push
- No comments unless the WHY is non-obvious
- No extra abstractions beyond what the task requires
