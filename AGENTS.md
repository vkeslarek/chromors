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
│   ├── mod.rs          # Backend, Operation, SourceInput, TargetOutput, ImageTargetCapability, HistogramTargetCapability
│   ├── gpu/
│   │   ├── mod.rs      # GpuBackend, GpuHandle, GraphNodeHandle, Lod, RegionCache
│   │   ├── value.rs    # ValueKind, NodeEval, GraphValue (aka MaterializedBuffer)
│   │   ├── graph.rs    # Graph, GraphNode, SourceNode, KernelSpec, NodeId
│   │   ├── op.rs       # GpuOperation trait, OutputSpec
│   │   ├── ops.rs      # emit_image(), emit_unary() helpers
│   │   ├── emit.rs     # JIT Slang emitter, LayoutPlan, alloc_temps, emit_slang
│   │   ├── compile.rs  # Compiled (SPIR-V + wgpu pipelines)
│   │   ├── materialize.rs # 7-step materialization pipeline, execute_batch
│   │   ├── region.rs   # GpuRegion (prepare/materialize/materialize_batch)
│   │   ├── buffer.rs   # GpuBuffer (VRAM storage, upload/alloc/read_to_cpu)
│   │   ├── context.rs  # GpuContext (wgpu device + queue + pipeline cache)
│   │   ├── source.rs   # GpuSource (Buffer | Vips), fetch_region
│   │   ├── pass.rs     # bfs_find_cuts (device buffer-limit enforcement)
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
│   ├── image.rs        # Image<B: Backend> — phantom-typed handle
│   └── histogram.rs    # Histogram<B> handle, HistogramResult
├── target.rs           # ImageTarget, HistogramTarget, MaterializedImage, MaterializedHistogram
├── operation/
│   ├── composite.rs    # Composite2Operation (vips) + multi-input GPU composite
│   ├── filters.rs      # GaussianBlurOperation (vips + GPU reference impl)
│   ├── stats.rs        # HistogramOp (GPU non-image output reference impl)
│   ├── misc.rs         # ColorConvertOp (dst_meta override reference impl)
│   └── …               # arithmetic, bands, geometry, icc, opacity, edge, …
└── pixel/
    ├── format.rs       # PixelFormat (38 variants)
    └── meta.rs         # PixelMeta { format, color_space, alpha_policy }
```

## Key design rules

### The typed value model

Every graph edge carries a `ValueKind` — the shape tag that describes what a node produces:

```rust
pub enum ValueKind {
    Image,                       // 2-D pixel buffer (any PixelFormat)
    Histogram { bins: u32 },    // uint atomic accumulator
    PointList { capacity: u32 },// (x,y) append list
    Scalar,                      // single f32
    Features { channels: u32 }, // multi-channel feature map
}
```

At materialize time `ValueKind` drives buffer allocation size (`compile.rs`) and whether the node gets a float4 temp buffer (`alloc_temps` — only `Image` nodes do). Non-image nodes write directly to their target.

The runtime payload is `GraphValue` (re-exported as `MaterializedBuffer`):
```rust
pub enum GraphValue {
    Image { buffer: Arc<GpuBuffer>, buffer_rect: Rect, source_rect: Rect },
    Raw   { bytes: Vec<u8>, kind: ValueKind, source_rect: Rect },
}
```

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

- `Image<B: Backend>` = phantom-typed handle, only holds `B::Handle`
- `Image<VipsBackend>` has vips-specific methods; `Image<GpuBackend>` has GPU-specific methods
- Capability traits gate which methods compile: `OpenFile`, `OpenBuffer`, `SourceInput`, `TargetOutput`, `ImageTargetCapability`, `HistogramTargetCapability`
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
       type Output = crate::Image;
       fn name() -> &'static [u8] { b"my_op_name\0" }
       fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
           op.set_image("in", image);
           op.set_double("amount", self.amount as f64);
       }
   }
   ```

3. **Implement GpuOperation** (GPU graph builder):
   ```rust
   impl GpuOperation for MyOperation {
       fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
           emit_image(graph, input, self_arc, "ops.my_module", "my_kernel",
               vec![Param::F32(self.amount)])
       }
       // If sigma is in pixel units, scale with LOD:
       fn lod_scale_param_indices(&self) -> &'static [usize] { &[0] }
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
2. Implement `GpuOperation::emit` using `emit_unary` with the appropriate `ValueKind`:
   ```rust
   fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
       emit_unary(graph, input, self_arc, "ops.histogram", "histogram_kernel",
           vec![Param::U32(self.channel)],
           ValueKind::Histogram { bins: self.bins })
   }
   fn output_spec(&self, _w: u32, _h: u32) -> OutputSpec {
       OutputSpec::Histogram { bins: self.bins }
   }
   fn inverse_map(&self, _rect: Rect, w: u32, h: u32, lod: Lod) -> Vec<(usize, Rect)> {
       // Histogram needs the whole image
       let s = lod.scale_factor();
       vec![(0, Rect::new(0, 0, (w as f64 / s).ceil() as i32, (h as f64 / s).ceil() as i32))]
   }
   ```
3. Expose a typed handle (e.g. `HistogramHandle`) with a `pull()` method that creates a `GpuRegion`, calls `materialize()`, and extracts bytes from `GraphValue::Raw`. See `HistogramOp::apply` + `HistogramHandle` in `operation/stats.rs` and the `HistogramTargetCapability` impl in `mod.rs`.

## How to add a multi-input GPU operation (e.g. composite, bandjoin)

Build `GraphNode` directly to specify multiple inputs:
```rust
fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
    let overlay_id = graph.add_source(Arc::new(/* GpuSource for overlay */));
    graph.add_node(GraphNode {
        id: NodeId(0),
        inputs: vec![input, overlay_id],   // 0 = primary, 1 = overlay
        eval: NodeEval::Kernel(KernelSpec { module: "ops.compose", function: "compose_kernel" }),
        params: vec![Param::U32(self.mode as u32)],
        op: self_arc,
        dst_meta: None,
        output: ValueKind::Image,
    })
}
fn inverse_map(&self, rect: Rect, _w: u32, _h: u32, _lod: Lod) -> Vec<(usize, Rect)> {
    vec![(0, rect), (1, rect)]  // both inputs, same rect
}
```
The emitter passes both source regions to the kernel: `compose_kernel(idx, region_src_0, region_src_1, region_tmp_N, u0)`.

## How to override output metadata (color space / format)

Set `dst_meta: Some(PixelMeta { ... })` on the `GraphNode`. The emitter uses the last `dst_meta` in the chain to build the `dst_cs` shader constant. This is how `ColorConvertOp` (`operation/misc.rs`) works — it emits a passthrough kernel but overrides the output color space so the final `from_working()` encode converts to the target space.

## How to add a new typed data handle (e.g. `Features<B>`)

Future work (Phase D). Pattern mirrors `Histogram<B>` in `data/histogram.rs`:
1. Define `pub struct Features<B: Backend>(GraphNodeHandle, PhantomData<B>)`.
2. Add `FeaturesTargetCapability` in `backend/mod.rs` with `create_features` + `pull_features`.
3. Implement for `GpuBackend`: `pull_features` creates `GpuRegion`, materializes, extracts `GraphValue::Raw { kind: ValueKind::Features { .. }, .. }`.
4. Add `FeaturesTarget<B>` in `target.rs`.
5. Add `MaterializedFeatures<B>` with the decoded typed payload.

## GPU optimization mechanisms

### Graph fusion (no per-op dispatch)
All `Image`-producing nodes in a pass are fused into **one Slang shader, one `queue.submit()`**. Intermediate `float4` results live in temp buffers (`RWRegion`) with slot reuse via interval coloring (`alloc_temps`). No round-trips to CPU between ops.

### Source fetch coalescing (region merge)
`merge_overlapping(source_rects)` in `materialize.rs` merges all source rects needed for a batch into fewer, larger rectangles before calling into Vips. One merged fetch = one libvips region prepare+fetch = one CPU→GPU upload. GPU efficiency scales with payload size, so batching pixels amortizes the transfer cost.

### Device buffer-limit cuts (BFS staging)
`bfs_find_cuts` in `pass.rs` does a BFS from the root node, counting how many storage buffer bindings each pass would require. When the count would exceed `ctx.max_storage_buffers` (device limit, commonly 8), it marks cut nodes. These are pre-materialized and injected as `BufferSource` nodes via `subgraph_with_overrides`, keeping each shader pass within hardware limits. The cut adds a GPU→staging→GPU round-trip only at the cut boundary, not for every node.

### Pipeline caching (no re-compile)
`GpuContext.pipeline_cache` (keyed by Slang text hash) stores compiled `wgpu::ComputePipeline`s. The emitter uses **positional names** (not NodeId-based) so structurally identical graphs produce identical Slang text → cache hit even after the graph has grown with new operations upstream.

### LOD scaling (cheap preview without re-parameterization)
Pixel-space params (blur sigma, etc.) listed in `lod_scale_param_indices()` are automatically divided by `lod.scale_factor()` before dispatch. The graph structure is re-used at every LOD; only param bytes change.

### Coordinate frame separation (`buffer_rect` / `source_rect`)
A `GraphValue::Image` may be a sub-region of a larger VRAM buffer (from a merged fetch or cache hit). `buffer_coords(image_rect)` maps image-space to buffer-local. Shaders always work in buffer-local coords via the `BufferRegion { stride, x, y, width, height }` params — no wasted copies to tightly-pack sub-rects before dispatch.

## Three Vips operation styles

**VipsOperation** — wraps libvips native ops through the GObject API. Blanket impl of `Operation<VipsBackend>` via `Runner` trait. Use for anything libvips provides natively.

**VipsCustomOperation** — pure Rust code inside the libvips demand-driven pipeline. `generate(out, input)` called per output region. No full-image download. Use for custom pixel-wise transforms.

**VipsCustomSink** — reduction via `vips_sink`. `fold(acc, region)` per region (from vips threadpool), `merge(total, part)` to combine, `finish(acc) -> Output`. Use for stats, histograms, any reduction. Wrap in `Reduce<S>` for `Image::execute` compatibility.

## Directory reference

| What | Where |
|---|---|
| New typed value kind | `backend/gpu/value.rs` — add variant to `ValueKind`, handle in `compile.rs` buffer sizing and `emit.rs` `alloc_temps` |
| New GPU operation | `operation/<category>.rs` — implement `VipsOperation` + `GpuOperation`, add to `operation/mod.rs`, add cross-backend test |
| New Slang shader | `pixors-shader/shaders/ops/<name>.slang`, trigger compile with `cargo build -p pixors-shader` |
| New typed data handle | `data/<name>.rs` (handle struct), `backend/mod.rs` (capability trait), `target.rs` (Target + Materialized wrapper) |
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
