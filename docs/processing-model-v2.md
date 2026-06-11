# Processing model v2

Status: **design doc**, refining. A lazy, multi-backend graph where adding a
new datatype or operation is *additive* — new structs and impls, never an
edit to a central enum or a `match`.

**Two halves, cleanly split.** The *description* of a computation (what data,
how it divides, what depends on what) is **backend-agnostic** and written
once. The *lowering* of that description onto a concrete engine — Vips
(libvips, CPU reference, used heavily) or GPU (Slang JIT fusion) — is a small
per-backend trait, written only for the backends a thing actually supports.
You never write the structural half twice; you only write the part that is
genuinely different between Vips and GPU.

The model is seven nouns:

| Noun | What it is | Half |
|------|------------|------|
| **Kind** | A category of data + its metadata + how its work divides (`Image2D`, `FeatureSet`). | agnostic |
| **WorkUnit** | The slice of a Kind being asked for (`Region`, `Range`, `Atomic`). | agnostic |
| **Operation** | A node: owns its typed inputs, declares output Kind + per-input demand. | agnostic |
| **Backend** | The engine that runs a DAG (`VipsBackend`, `GpuBackend`): owns context, buffer, and lowering. | — |
| **Lower\<B\>** | How one Operation/Kind realizes on backend `B` (Slang entry+view, or a vips op). | per-backend |
| **Buffer\<B\>** | A materialized result on backend `B` + the Kind that tags it. | per-backend |
| **Source / Target** | The doors in/out. Source produces a `Buffer<B>`; Target is the only exit (download, decode, write). | per-backend |

---

## Kind — the agnostic datatype, plus per-backend lowering

A `Kind` value is the datatype's **metadata** (an image's format + color
space, a point list's capacity). It tags every node's output and every
Buffer, and is **backend-agnostic** — the same `Image2D` Kind describes an
image whether it ends up on Vips or the GPU. Split object-safe core
(`AnyKind`) from the typed surface (`Kind`):

```rust
/// Object-safe, backend-agnostic. No `view`/`params` here — those are
/// Slang-specific and live in the GPU lowering (see `GpuView`).
pub trait AnyKind: Send + Sync + Debug + 'static {
    fn as_any(&self) -> &dyn Any;
    fn shape(&self) -> Shape;                  // Region | Range | Atomic
    fn byte_size(&self, wu: &WorkUnit) -> u64;
    /// Identity into a hasher (cache key, `Cached` adapter). Raw bits, not a
    /// `Debug` string. `AnyOperation` carries the same for op identity.
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

/// Typed surface: just the WorkUnit it divides into. No `residency`, no
/// `Value`/`decode` — turning a result into a host value is a Target's job.
pub trait Kind: AnyKind + Clone + Sized {
    type WorkUnit: WorkUnitFor;                 // Region | Range | Atomic
}
```

The backend-specific knowledge a Kind needs is a **per-backend capability
trait**, implemented only for the backends that Kind supports:

```rust
/// How this Kind is wrapped in a Slang slot + the std430 geometry its wrapper
/// reads. Image + mask Kinds impl it; a GPU-only point list impls it; a Kind
/// that never touches the GPU does not.
pub trait GpuView: Kind {
    fn view(&self, role: Role) -> View;        // "Codec<Rgba8,Srgb>", "PointListView<N>"
    fn params(&self, wu: &WorkUnit) -> ParamBlock;
}

/// How this Kind maps onto a libvips band format. Image-like Kinds impl it;
/// a GPU-only Kind (e.g. a fused point list) simply does not — and so it
/// can't be used on `VipsBackend`, enforced at compile time.
pub trait VipsBand: Kind {
    fn band_format(&self) -> VipsFormat;
}
```

`Data<K, B>` only compiles when `K` supports `B` (the backend's
materialize bounds on `K: GpuView` / `K: VipsBand`). A `FeatureSet` that only
impls `GpuView` is simply never a valid `Data<FeatureSetKind, VipsBackend>` —
no runtime "unsupported on this backend" error.

`Shape` and `Role` are the only closed enums in the model. Both are tiny,
shared by all Kinds, and structural (3 materialize strategies; 3 shader
slots) — they have not grown across datatypes and aren't expected to.

`WorkUnit` is the erased slice enum; `WorkUnitFor` is its typed counterpart
(`Region`/`Range`/`Atomic` each implement it):

```rust
pub enum Shape { Region, Range, Atomic }
pub enum Role  { Input, Output, Temporary }

pub enum WorkUnit { Region(Region), Range(Range), Atomic }
pub trait WorkUnitFor: Clone + Send + Sync + 'static {
    fn erase(&self) -> WorkUnit;
    fn typed(wu: &WorkUnit) -> Option<Self>;
}
```

---

## Backend — the engine a DAG runs on

A `Backend` is a unit struct naming an execution engine. It owns three
associated types and the materialize entry point:

```rust
pub trait Backend: Sized + Send + Sync + 'static {
    type Ctx: Send + Sync;       // GpuBackend → GpuContext; VipsBackend → VipsContext
    type Payload: Send + Sync;   // GpuBackend → GpuBuffer;  VipsBackend → VipsRegion
    type Builder;                // lowering accumulator: GpuEmitter / VipsPipeline

    /// Walk the agnostic DAG, lower each node into a `Builder`, run it, return
    /// the result. GPU: emit one fused Slang module + dispatch. Vips: build a
    /// libvips demand-driven pipeline + sink the region.
    fn materialize<K: Kind>(ctx: &Self::Ctx, root: &Arc<Node<Self>>, wu: &WorkUnit)
        -> Result<Buffer<Self>, Error>;
}

pub struct GpuBackend;
pub struct VipsBackend;
```

A materialized result is `Buffer<B>` — the backend's payload + the Kind tag:

```rust
pub struct Buffer<B: Backend> {
    payload: Arc<B::Payload>,     // GpuBuffer (VRAM) | VipsRegion (CPU)
    spec: Arc<dyn AnyKind>,
}
```

`spec` is `Arc<dyn AnyKind>` (not `Arc<dyn Any>`) so any backend reads
`shape`/`byte_size`/`dyn_hash` without downcasting; the concrete Kind is
recovered only for type-specific fields (and the backend's own `*View`/`*Band`
capability).

There is **no `residency` and no `Host` variant.** A `Buffer<GpuBackend>` is
VRAM; a `Buffer<VipsBackend>` is a CPU region. Either way the *only* hop to a
plain host value is a `Target` — `materialize` itself never downloads.

---

## View — the decode/process/encode sandwich (GPU lowering)

> This whole section is **GPU-backend lowering** — `View`/`ParamBlock` are
> `GpuView`'s vocabulary. The Vips backend has no `View`; it lowers a Kind to
> a band format and lets libvips stream. Nothing here touches the agnostic
> core.

A Buffer always carries a *real* Kind — including fused-internal edges (an
op's output is a real `Image2D { format, color_space }`, e.g. ACEScg f16,
never a "raw" placeholder). So a kernel's three slots are unconditional:

- `view(Input)`  → "decode from the upstream Buffer's encoding into my
  working representation."
- `view(Output)` → "encode my working representation into my own
  `output_spec()`'s encoding."
- `view(Temporary)` → an op's private scratch layout (no codec — written and
  read by the same dispatch).

```rust
pub struct View {
    pub slang: Cow<'static, str>,   // "Codec<Rgba8, Srgb>", "PointListView<4096>", …
    pub binding: Binding,
}
// built via a small typed builder, not hand-concatenated strings:
//   View::structured("PointListView").param(capacity)
//   View::codec(format, color_space)
```

The codec lives **entirely in Slang** — a generic library parameterised by
the Spec's metadata does the conversion on `.get()`/`.set()`. When the
upstream encoding already equals the working representation, the codec is an
identity (still emitted, just cheap). There is no "raw vs working vs output"

## View — the decode/process/encode sandwich

A Buffer always carries a *real* Kind — including fused-internal edges (an
op's output is a real `Image2D { format, color_space }`, e.g. ACEScg f16,
never a "raw" placeholder). So a kernel's three slots are unconditional:

- `view(Input)`  → "decode from the upstream Buffer's encoding into my
  working representation."
- `view(Output)` → "encode my working representation into my own
  `output_spec()`'s encoding."
- `view(Temporary)` → an op's private scratch layout (no codec — written and
  read by the same dispatch).

```rust
pub struct View {
    pub slang: Cow<'static, str>,   // "Codec<Rgba8, Srgb>", "PointListView<4096>", …
    pub binding: Binding,
}
// built via a small typed builder, not hand-concatenated strings:
//   View::structured("PointListView").param(capacity)
//   View::codec(format, color_space)
```

The codec lives **entirely in Slang** — a generic library parameterised by
the Spec's metadata does the conversion on `.get()`/`.set()`. When the
upstream encoding already equals the working representation, the codec is an
identity (still emitted, just cheap). There is no "raw vs working vs output"
state in Rust: **changing a buffer's encoding is itself an `Operation`**
(`ColorConvert`'s `output_spec()` returns a different `Image2D`), so no
fusion step can silently reinterpret bytes.

---

## Operation — agnostic structure + per-backend `Lower<B>`

An op is generic over its backend through its input edges (`Input<K, B>`
points at an upstream `Node<B>`). Its **structure** — inputs, demand, output
Kind — is written *once*, generic over `B`. Its **execution** — the genuinely
different Slang-emit vs. vips-setup — is a separate `Lower<B>`, written once
per backend the op supports:

```rust
pub struct Input<K: Kind, B: Backend> { pub src: Arc<Node<B>>, pub spec: Arc<K> }

/// Structural half — one generic impl per op, shared by every backend.
pub trait Operation<B: Backend>: Lower<B> + 'static {
    type Output: Kind;

    /// Structural edges, in slot order (wiring + content hash).
    fn inputs(&self) -> Vec<&dyn AnyInput<B>>;

    /// Per-input slice demanded to produce `out` (halos, full-image
    /// reductions). One entry per `inputs()`, same order. `None` **prunes**
    /// that input for this region — not fetched, not bound, not sampled (an
    /// opaque overlay fully covering `out` returns `None` for the base).
    fn demand(&self, out: &<Self::Output as Kind>::WorkUnit) -> Vec<Option<WorkUnit>>;

    /// Output metadata, derived from `self`'s own typed input specs.
    fn output_spec(&self) -> Self::Output;
}

/// Execution half — the ONLY part that differs between backends. Written once
/// per (op, backend) pair; the bodies are irreducibly different (emit a Slang
/// entry + params, vs. construct + wire a libvips operation).
pub trait Lower<B: Backend> {
    fn lower(&self, cx: &mut B::Builder);
}
```

So a blur is: one struct (`Blur<B>`), one *generic* `impl<B: Backend>
Operation<B> for Blur<B>` (inputs/demand/output_spec — never duplicated), then
`impl Lower<GpuBackend> for Blur<GpuBackend>` (Slang `blur_main` + radius
param) and `impl Lower<VipsBackend> for Blur<VipsBackend>` (`vips_gaussblur`
setup). Drop the vips impl and the blur is simply GPU-only — there is no
"unsupported backend" branch to write, the type system removes it.

`Input<K, B>` erases to `dyn AnyInput<B> { fn src(&self) -> &Arc<Node<B>>; fn
spec(&self) -> &dyn AnyKind; }`. The `Arc<Node<B>>` *is* the upstream edge —
no arena, no `NodeId` (see "No Graph").

---

## Authoring — a multi-backend datatype + operation

### A. An image datatype (runs on both backends)

Metadata struct + agnostic impls + one capability impl *per backend it
supports*. `Image2D` supports both, so it impls `GpuView` **and** `VipsBand`:

```rust
#[derive(Clone, Debug)]
pub struct ImageKind { pub format: PixelFormat, pub color_space: ColorSpace }

impl AnyKind for ImageKind {
    fn as_any(&self) -> &dyn Any { self }
    fn shape(&self) -> Shape { Shape::Region }
    fn byte_size(&self, wu: &WorkUnit) -> u64 { /* w*h*bpp from wu */ }
    fn dyn_hash(&self, s: &mut dyn Hasher) { self.format.hash(s); self.color_space.hash(s); }
}
impl Kind for ImageKind { type WorkUnit = Region; }

impl GpuView for ImageKind {                                   // GPU lowering
    fn view(&self, _: Role) -> View { View::codec(self.format, self.color_space) }
    fn params(&self, wu: &WorkUnit) -> ParamBlock { /* BufferRegion */ }
}
impl VipsBand for ImageKind {                                  // Vips lowering
    fn band_format(&self) -> VipsFormat { self.format.into() }
}

pub type Image2D<B = GpuBackend> = Data<ImageKind, B>;
```

### B. An operation on both backends — `Blur`

Structure once (generic over `B`); execution once per backend:

```rust
pub struct Blur<B: Backend> { input: Input<ImageKind, B>, radius: f32 }

impl<B: Backend> Operation<B> for Blur<B> where Blur<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.expanded(self.radius.ceil() as i32)))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }   // passthrough
}

impl Lower<GpuBackend> for Blur<GpuBackend> {
    fn lower(&self, cx: &mut GpuEmitter) {                     // fuses into the Slang chain
        cx.kernel("blur_main").param("radius", self.radius);
    }
}
impl Lower<VipsBackend> for Blur<VipsBackend> {
    fn lower(&self, cx: &mut VipsPipeline) {                   // libvips streams it
        cx.op("gaussblur").set("sigma", self.radius as f64);
    }
}

// First-party ops (in the engine crate) attach sugar as an inherent method.
// A DOWNSTREAM op cannot (`impl Data` outside its crate is E0116) — it uses an
// extension trait. Same call site either way.
impl<B: Backend> Image2D<B> where Blur<B>: Lower<B> {
    pub fn blur(&self, radius: f32) -> Image2D<B> { self.push(Blur { input: self.as_input(), radius }) }
}
// downstream form:
// pub trait BlurExt<B> { fn blur(&self, r: f32) -> Image2D<B>; }
// impl<B: Backend> BlurExt<B> for Image2D<B> where Blur<B>: Lower<B> { … }
```

`img.blur(2.0)` now works identically on `Image2D<VipsBackend>` and
`Image2D<GpuBackend>`. Drop the `Lower<VipsBackend>` impl and `blur` simply
vanishes from vips images — no "unsupported" branch, the bound removes it.

> **Verified in the POC** (`poc/tests/smoke.rs`, a separate crate = a downstream
> consumer): the whole chain — `ImageKind` (`AnyKind`+`Kind`+`GpuView`),
> `Blur<B>` (one generic `Operation<B>` + one `Lower<GpuBackend>`), the erased
> `AnyOperation<B>` bridge, and `img.blur(2.0).blur(4.0)` — type-checks and
> compiles. The orphan rule forced the extension-trait form above; everything
> else matched the model as written.

### C. A GPU-only datatype — `FeatureSet` (detect corners)

A point set has no libvips equivalent: it impls `GpuView` and **not**
`VipsBand`, so it's GPU-only by construction. Its op fixes the backend to
`GpuBackend`:

```rust
pub struct FeatureSetKind { pub capacity: u32 }
impl AnyKind for FeatureSetKind { /* shape = Atomic, byte_size, dyn_hash */ }
impl Kind for FeatureSetKind { type WorkUnit = Atomic; }
impl GpuView for FeatureSetKind {
    fn view(&self, _: Role) -> View { View::structured("PointListView").param(self.capacity) }
    fn params(&self, _: &WorkUnit) -> ParamBlock { ParamBlock::scalar("capacity", "uint", self.capacity) }
}
// no VipsBand → `Data<FeatureSetKind, VipsBackend>` does not type-check.

pub type FeatureSet = Data<FeatureSetKind, GpuBackend>;

pub struct DetectCorners { input: Input<ImageKind, GpuBackend>, threshold: f32 }
impl Operation<GpuBackend> for DetectCorners {
    type Output = FeatureSetKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<GpuBackend>> { vec![&self.input] }
    fn demand(&self, _: &Atomic) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(Region::full(self.input.spec.dims())))]   // whole image
    }
    fn output_spec(&self) -> FeatureSetKind { FeatureSetKind { capacity: 4096 } }
}
impl Lower<GpuBackend> for DetectCorners {
    fn lower(&self, cx: &mut GpuEmitter) { cx.kernel("detect_corners").param("threshold", self.threshold); }
}

impl Image2D<GpuBackend> {
    pub fn detect_corners(&self, t: f32) -> FeatureSet {
        self.push(DetectCorners { input: self.as_input(), threshold: t })
    }
}
```

`detect_corners` exists *only* on `Image2D<GpuBackend>` — a vips image has no
such method, again enforced by the type, not a runtime check.

### D. The exit Target (host download lives here, not on the Kind)

```rust
pub struct PointsHost;
impl Target<FeatureSetKind, GpuBackend> for PointsHost {
    type Out = Vec<(f32, f32)>;
    fn extract(&self, buf: &Buffer<GpuBackend>, _: &Atomic, ctx: &GpuContext) -> Result<Self::Out, Error> {
        let raw = buf.download(ctx)?;           // the one GPU→host hop
        let count = u32::from_le_bytes(raw[..4].try_into()?) as usize;
        Ok(bytemuck::cast_slice(&raw[4..])[..count].to_vec())
    }
}
```

> **Naming:** the metadata struct is `*Kind` (author-only); the user name is a
> `pub type Image2D<B> = Data<…Kind, B>` alias. The user sees one concrete
> type; the author writes the metadata + a one-line alias. (Earlier "Handle"
> is gone — the generic core is `Data<K, B>`.)

---

## Using — processing something

The caller picks a backend once (at the source) and the whole pipeline is
that backend. Everything is lazy until a `Target`. `materialize` is
**internal** — the only public exits run a Target, so a result never silently
lands on the host:

```rust
let img  = vips::open("in.jpg")?;                   // Image2D<VipsBackend>
let edit = img.blur(4.0).exposure(1.0);            // lazy, same methods as GPU
edit.into(&JpegFile::new("out.jpg"), Region::full(edit.dims()))?;   // Target, Out = ()

let gpu  = gpu::open("in.jpg", &ctx)?;             // Image2D<GpuBackend>
let tile = gpu.blur(4.0).into(&ViewportTile, Region::tile(0, 0))?;  // Target, Out = Buffer<Gpu> (no download)
let pts  = gpu.detect_corners(0.2).into(&PointsHost, Atomic)?;      // Target, Out = Vec<(f32,f32)>
```

The generic core (never named by users; friendly names are aliases) is one
struct over `(Kind, Backend)`:

```rust
pub struct Data<K: Kind, B: Backend> { root: Arc<Node<B>>, ctx: Arc<B::Ctx>, spec: Arc<K>, _m: PhantomData<(K,B)> }

impl<K: Kind, B: Backend> Data<K, B> {
    /// Internal — exposing this invites `.download()` and breaks the invariant.
    pub(crate) fn materialize(&self, wu: K::WorkUnit) -> Result<Buffer<B>, Error> {
        B::materialize::<K>(&self.ctx, &self.root, &wu.erase())
    }

    /// The single public terminal. Host targets download; a disk target
    /// writes (`Out = ()`); a viewport target clones the buffer (no download).
    pub fn into<T: Target<K, B>>(&self, t: &T, wu: K::WorkUnit) -> Result<T::Out, Error> {
        t.extract(&self.materialize(wu.clone())?, &wu, &self.ctx)
    }

    fn as_input(&self) -> Input<K, B> { Input { src: self.root.clone(), spec: self.spec.clone() } }
    fn push<Op: Operation<B, Output = K2>, K2: Kind>(&self, op: Op) -> Data<K2, B> {
        Data { root: Arc::new(Node::Op(Arc::new(op))), ctx: self.ctx.clone(), spec: Arc::new(op.output_spec()), _m: PhantomData }
    }
}
```

`push` wraps a **new** root `Arc`; the old DAG is shared, never mutated.
`clone()` is `Arc::clone`. No `Mutex`, no arena — concurrent handles are
lock-free.

`width()`/`height()` and Region-only ergonomics live on `impl<B> Image2D<B>`
(or a `Spatial` extension) — `FeatureSet` never has them, enforced by the
type.

---

## Source / Target — the doors, per backend

`demand()` recursion bottoms out at **Source** leaves; **Target** is the sole
exit. Both carry the backend, and a Source carries its (single) output Kind as
an associated type:

```rust
pub trait Source<B: Backend>: Send + Sync + 'static {
    type Kind: Kind;
    fn spec(&self) -> Arc<Self::Kind>;
    fn fetch(&self, ctx: &B::Ctx, wu: &<Self::Kind as Kind>::WorkUnit) -> Result<Buffer<B>, Error>;
    fn lower(&self, cx: &mut B::Builder);     // inject this leaf's view/config (concrete Kind known)
    fn dyn_hash(&self, s: &mut dyn Hasher);   // leaf identity for the cache key
}

pub trait Target<K: Kind, B: Backend>: Send + Sync {
    type Out;                                 // () = write to disk; Vec<…> = host value; Buffer<B> = stay resident
    fn extract(&self, buf: &Buffer<B>, wu: &K::WorkUnit, ctx: &B::Ctx) -> Result<Self::Out, Error>;
}
```

`Target::extract` owns the **one** download (and any host decode) — that's why
neither `Kind` nor the model carries a `residency` or `decode`. `VipsSource:
Source<VipsBackend, Kind = ImageKind>`, `GpuSource: Source<GpuBackend, Kind =
ImageKind>`; `JpegFile: Target<ImageKind, B>` works for whichever `B` produced
the image. A GPU-resident exit (viewport) is just `Target<_, GpuBackend, Out =
Buffer<GpuBackend>>` that clones the Arc — every exit is a typed Target, so
there is no raw download path to misuse.

### Data caching is the user's call — a Source adapter, not a builtin

The engine does **not** own a tile/output value cache. Whether a fetch or a
materialized region is remembered is a policy only the caller knows:
interactive editing wants it (the same tiles are re-asked every frame as the
user drags a slider); a batch export does not (each tile is touched once;
caching is pure memory waste). So caching is **opt-in, by wrapping a Source**:

```rust
pub struct Cached<S> { inner: S, store: Mutex<HashMap<(WorkUnit-key), Buffer>> }
impl<K: Kind, S: Source<K>> Source<K> for Cached<S> {
    fn spec(&self) -> Arc<K> { self.inner.spec() }
    fn fetch(&self, wu: &K::WorkUnit) -> Result<Buffer, Error> {
        // hit → clone the GPU Buffer Arc; miss → inner.fetch + insert
    }
}

let src = VipsSource::open("in.jpg")?;
let img = if interactive { Image2D::from(src.cached()) } else { Image2D::from(src) };
```

Same adapter pattern can wrap any node to memoize a *materialized* sub-result,
not just a leaf fetch — but it stays an explicit caller decision, never an
implicit engine cost. (Contrast: today's `materialize.rs` bakes in a
`RegionCache` everyone pays for.)

The **compilation cache is different and stays engine-owned** — see "No
Graph" below. That caches *shaders*, not data; it's always on because
compiling is the dominant cost and shader reuse is never a waste.

---

## Params — one std430 buffer, two contributors *(GPU lowering)*

> GPU-backend only. The fused Slang kernel needs every Kind's geometry +
> every op's scalars in one buffer. The Vips backend has no `ParamBlock` — it
> sets typed properties on libvips operations instead.

The fused kernel takes a *single* storage params struct (`ChainParams`, an
SSBO — see "ChainParams is an SSBO"). Two parties write into it, in a fixed
order, with positional field names so structurally-identical graphs produce
byte-identical layouts (and hit the pipeline cache):

```rust
pub struct ParamBlock {
    pub fields: Vec<(String, &'static str)>,  // (name, slang type) — e.g. ("u3", "float")
    pub bytes: Vec<u8>,                       // std430 little-endian payload
}
```

Both contributions are **injected by the node's `lower`** (the concrete-type
site), never pulled by the materializer — see "The materializer is type-blind"
below for why this is the only thing that compiles.

1. **Kind params** — inside a node's `Lower<GpuBackend>::lower`, the op/source
   calls `GpuView::params(wu)` on its (statically-known) input/output/temp
   Kinds and pushes the result into the builder. For images that's the
   `BufferRegion { stride, x, y, w, h }` the `view` wrapper indexes with; for a
   point list it's the `capacity` and atomic-counter offset. Named
   `inputs_{i}` / `temp_region_{b}` / `region_target_{i}` by slot index.

2. **Operation params** — the same `lower` appends the op's scalars (`u{n}`
   fields) via the `GpuBuilder`. The builder records each node's base field
   index so the generated call site reads `g_params[0].u{base+k}`. LOD scaling
   happens *here* (a blur radius shrinks at coarser MIPs) — baked in before
   serialisation, the shader sees a plain scalar.

This closes the gap the old model left implicit: both the Kind's
buffer-geometry constants **and** the Operation's config are first-class,
serialised the same way, into the same buffer — neither is special-cased in
the codegen. Adding a Kind or an Op that needs new params is just a longer
`ParamBlock`, never a new field hardcoded in the emitter.

*(Today's `Param::{Struct, Region}` are still `unimplemented!` in the JIT
emitter — a `ParamBlock`-of-bytes model subsumes them: a struct/region param
is just more named fields + bytes, no enum branch.)*

---

## Materialization — `Backend::materialize`, per backend

`Data::materialize(wu)` just calls `B::materialize`. The two backends share
the **agnostic front half** (the demand walk over `Operation::demand`) but
diverge in how they *run* the lowered DAG.

**Vips backend** (the heavily-used reference): walk the DAG, call each op's
`Lower<VipsBackend>::lower` to construct + wire a libvips operation, attach
the leaf `Source`s, and let libvips's own demand-driven engine stream the
requested region into a `VipsRegion`. No Slang, no `ParamBlock`, no manual
fusion — libvips fuses internally. `demand()` informs tiling; the heavy
lifting is libvips's.

**GPU backend**: the JIT-fused path below — one Slang module, chained entry
points, one dispatch. This is the rest of this section.

### GPU `materialize`, step by step

Lazy until here; the walk runs fresh every call (cheap, and what enables
per-region specialization).

**1. Demand walk (inverse map).** *(shared front half.)* From the root
`WorkUnit`, pop a
`(node, WorkUnit)`, resolve it to a concrete rect, and for each input call
`Operation::demand(out_wu)`. A `Some(wu)` propagates to that input; a `None`
**prunes** it (region-specific — an opaque overlay drops the base). Push onto
upstream nodes, bounding-box-accumulating where reached by multiple paths.
Stop at `Source` leaves. Result: every live node's required rect, for *this*
region.

**2. Source fetches.** For each reachable `Source`, merge overlapping demanded
rects, tile-align (256px), clamp to source bounds. `Source::fetch` runs later,
in parallel. If the caller wrapped the Source in `Cached` (its choice), a
re-fetch hits; otherwise it re-uploads.

**3. Layout + lower (the type-blind walk).** Post-order over the DAG, the
materializer allocates a slot per node — sizing it with the **agnostic**
`AnyKind::byte_size(wu)` (the one thing it *can* read from an erased Kind) —
then calls that node's `lower(&mut GpuBuilder)`. Slot kinds:
   - **Sources** → group-0; **Temporaries** → group-1, **liveness-reused**
     (a freed temp is handed to a later node of identical dims, bounding VRAM);
     **Output(s)** → group-2.
   - Each node's `lower` **injects** its kernel + scalar params + input/output
     `View`s into the builder (concrete-type site — see below). The
     materializer never reads a `View`.

   If `sources + 1` or `temps + outputs` exceeds the device's storage-buffer
   limit, a **cut finder** splits the graph: materialize a subgraph to its own
   buffer, swap it in as a synthetic `Source`, and recurse.

**4. Kernel fusion / emit.** The builder now holds the whole reachable
subgraph as **one Slang module** — each node a compute entry point
(`entry_0`, `entry_1`, … in topo order) that bounds-checks its slot, reads
inputs through the injected `view(Input)` (`region_tmp_{b}` for an upstream
temp — working-space, no re-decode; or a source's decode codec), calls the
op's registered Slang function with `(idx, inputs…, params…)`, and writes via
`view(Output)` into its temp (or, for the final node, the output buffer).

   **Fusion = chained entry points sharing temp buffers in one command
   encoder, no round-trip between ops.** The IR text is positional, so its
   hash is the pipeline-cache key. Because step 1 pruned per region, the IR —
   and thus the cache entry — is specialized to this region's live set.

### The materializer is type-blind — Views are *injected*, not pulled

The materializer walks a DAG of `Arc<dyn AnyOperation<B>>` / `Arc<dyn
AnySource<B>>`; every edge yields only `Arc<dyn AnyKind>`. It **cannot** ask
that erased Kind for its GPU `View`/`params`: Rust has no cross-cast from
`dyn AnyKind` to `dyn GpuView`, and downcasting to concrete Kinds
(`downcast_ref::<ImageKind>()`) would be the exact central `match` the model
exists to avoid. So the rule:

> **`GpuView`/params are never read by the materializer. Each node injects its
> own views into the builder inside `lower`,** where the concrete Kinds
> (`Input<ImageKind, _>`, `Self::Output`) are statically known and
> `K: GpuView` is provable.

This is why `lower` lives on **both** ops *and* sources (`AnySource<B>` gained
`lower` for exactly this — a leaf injects its own decode view), and why the
builder grew `input_view`/`output_view`. The materializer stays a pure,
type-blind orchestrator: walk, allocate by `byte_size`, call `lower`, compile,
dispatch. **Verified to compile** in `poc` (`GpuBackend::materialize`'s
`collect` walk + `smoke.rs`'s `Blur`/`ImageSource` lowering).

**5. Compile + fetch (parallel).** `slangc` → SPIR-V → pipeline, **looked up
in the engine-owned pipeline cache by IR hash** (the dominant cost; this is
the one cache that's always on). `Source::fetch` uploads on other threads. A
pass with zero entry points (pure passthrough of a fetched source)
short-circuits to the fetched buffer.

**6. Encode + submit.** Bind groups (0: sources+params, 1: temps+outputs),
dispatch each entry point in order, submit one encoder, free scratch.

**7. Result.** The output buffer is returned as a GPU `Buffer`, Kind-tagged —
stays GPU-resident for the next op, the viewport, or a Target. No readback, no
host copy, no decode happens here. A caller who wants it cached across frames
wraps the node (or the Source) in `Cached`; a caller who wants it on the host
runs a `Target`.

---

## No `Graph` — the DAG is the `Arc<Node<B>>` tree (decided)

**Decided: no persistent `Graph`.** `Input<K, B>` holds an `Arc<Node<B>>` to
its *producing node*, so the Operation tree **is** the DAG — navigate by
pointer-chasing inputs. Shared sub-expressions are shared `Arc`s; a diamond
(A feeds B and C) is natural. The DAG is backend-typed: a vips pipeline and a
gpu pipeline are different types and can't be mixed by accident.

```rust
pub enum Node<B: Backend> {
    Op(Arc<dyn AnyOp<B>>),       // erased Operation<B>: inputs / demand_erased / output_kind / lower / dyn_hash
    Source(Arc<dyn AnySource<B>>),
}
```

`AnyOp<B>` / `AnySource<B>` are the object-safe erased mirrors of
`Operation<B>` / `Source<B>` (the typed traits aren't object-safe: associated
`Output: Kind`, typed `WorkUnit`). A blanket impl bridges every typed op/source
to its erased form, so the pointer-walking materializer drives the DAG without
knowing concrete Kinds.

What today's `Arc<Mutex<Graph>>` + `NodeId` arena actually buys, and whether
it survives:

| Graph gives us | Without it |
|---|---|
| Stable `NodeId` for positional Slang names / pipeline-cache key | Assign indices during the one compile-time DFS; identity for in-DAG dedup = `Arc::as_ptr`. A structural hash (for a user `Cached` adapter's key) is just another tree walk. |
| In-place append by a shared builder | `push` returns a **new** root `Arc` wrapping the old one — immutable, structurally shared, no `Mutex`. More idiomatic Rust. |
| Reverse edges for liveness (who consumes node N) | One DFS builds the consumer map before layout — it was being walked anyway. |
| Cut-finder rewrites a node → synthetic source | Rebuild the spine from the cut point up (re-wrap `Arc`s); the rest is shared untouched. |
| `subgraph_with_overrides` / `merge_from` | Pointer substitution on an immutable tree — same operation, no arena bookkeeping. |

So the persistent `Graph` collapses into: **(a)** an immutable `Arc<Node>`
DAG the user builds and forks for free, and **(b)** a *transient* `Plan`/
`Layout` (indices, bindings, param offsets, liveness) built by one walk per
materialize and thrown away after. The arena, the `Mutex`, `get_node`/
`get_source` indirection, and NodeId plumbing all go away.

### The one thing that *must* persist: the compilation cache *(GPU backend)*

Walking the DAG per materialize is cheap. **Compiling a shader is the single
most expensive operation in the GPU backend** — far costlier than any
dispatch. So the one structure that outlives a GPU materialize is a
process-wide cache keyed by the emitted **IR hash** (the Vips backend has no
such compile step; libvips caches its own operations):

```
IR text (positional, deterministic)  ──hash──►  compiled pipeline (SPIR-V + wgpu::ComputePipeline)
```

Two materializes whose walk produces byte-identical IR — same ops, same Kinds,
same layout — share the pipeline. This is the **only** engine-owned cache.
Data (fetched tiles, materialized outputs) is *not* cached by the engine —
that's the caller's `Cached` adapter (see Source/Target). The pipeline cache,
by contrast, is always on: compiling dominates wall-clock and shader reuse is
never waste.

### Per-region specialization is a feature, not a problem

Because the demand walk + layout runs **per requested region**, the same DAG
can legitimately emit **different IR for different regions** — and that is
healthy:

- A `Normal` composite of an opaque RGB8 overlay over a base: in a region
  where the overlay fully covers the output, the walk can prune the base
  input entirely — `demand()` returns nothing for it, it's not fetched, not
  bound, not sampled. That region's IR has one source; a region where the
  overlay only partially covers has two. Different IR → different pipeline →
  two cache entries, each optimal.
- Coarser MIPs bake different LOD-scaled params (and can drop sub-pixel ops
  entirely), again specializing the IR.

This is the upside of re-walking per region: optimizations that depend on
*which* pixels are asked for fall out naturally, and the IR-hash cache keeps
each distinct specialization compiled exactly once. A persistent `Graph`
would have fought this (one baked structure per DAG); the transient walk
embraces it.

---

## Multi-pass operations — multi-node, not multi-dispatch-per-op

A separable Gaussian (X then Y), an FFT (log₂N butterfly stages), a tree
reduction — none of these are a single GPU dispatch. The model handles them
**without** adding anything to `Operation`, because fusion *is already*
multi-dispatch: a chain of N nodes emits N entry points run back-to-back in
one encoder, sharing temps, no host round-trip. So a multi-pass algorithm is
just **multiple nodes**, composed in the ergonomic wrapper:

```rust
impl Image2D {
    // fixed pass count → fixed nodes
    pub fn gaussian(&self, r: f32) -> Image2D { self.blur_x(r).blur_y(r) }

    // data-dependent pass count → loop at BUILD time (dims are known now)
    pub fn fft(&self) -> Image2D {
        let mut n = self.clone();
        for stage in 0..self.width().ilog2() { n = n.butterfly(stage); }
        n
    }
}
```

`node = exactly one dispatch` stays invariant. The win over a
multi-dispatch-per-op escape hatch: each pass is an independent node, so each
is independently cacheable, prunable, and fusable with its neighbours. An op
that needs scratch *within* its single dispatch (a workgroup-local block
reduction behind a barrier) uses `temps()` + writes the loop in its own
`entry()` Slang — the model never sees it.

**Known limit:** a pass count that depends on *runtime data* (not build-time
dims) can't be unrolled at build. Vanishingly rare in image processing (FFT
size, bin count, blur radius are all known when the DAG is built). If it ever
arises, that op spills to its own multi-stage Slang kernel; it does not change
the model.

---

## Boundaries, memory, and resolved questions

### Engine DAG vs. document — serialization lives a layer up

Killing the persistent `Graph` does **not** touch save/load, undo, or the UI
node tree. Those are `pixors-document`'s job: `Document` / `LayerNode` /
`compile_document()` own stable IDs, history, and serialization. The engine's
`Arc<Node>` DAG is the *transient* thing `compile_document()` **produces** per
render — you serialize the document, not the fused DAG. Inspecting the raw DAG
(debug, a pipeline preset) is one walk with `Arc::as_ptr → id`; shared `Arc`s
collapse to one id, so diamonds round-trip correctly. No persistent arena
needed for any of it.

### VRAM is RAII, not GC

No garbage collector, no central node manager. Two independent drops:
- **`Arc<Node>` chains** are tiny host structs (params + pointers). When the UI
  drops a handle, refcount hits zero and they free. A slider drag that spawns
  parallel chains cleans itself up.
- **Output `Buffer`s** are `Arc<GpuBuffer>`. The engine keeps **no** output
  cache, so a result is owned by whoever asked for it; on drop the `GpuBuffer`
  frees its VRAM (or returns to the allocator pool). Abandoned materializes
  reclaim themselves.

Bounded VRAM pressure exists in exactly two opt-in/owned places, each with a
clear policy: the caller's `Cached` adapter (CLOCK / `TieredCache`), and the
engine's **pipeline cache, which needs its own LRU bound** (rapid slider drags
mint many IR specializations — compiled pipelines must evict, even though data
isn't cached).

### `ChainParams` is an SSBO

The fused params struct binds as a **storage buffer** (`std430`), never a
uniform. A long JIT-fused chain concatenates many `ParamBlock`s and would blow
the ~64 KB UBO cap on limited devices; SSBOs lift that. (Already the case in
the current backend — `compile.rs` binds params as `BufferBindingType::Storage`.)

### Resolved design questions

1. **`output_spec` stays explicit.** Most ops write `(*self.input.spec).clone()`.
   We keep the one-liner — **no default impl, no derive macro.** Less magic
   beats less typing; an explicit `output_spec` never hides a surprising
   "which input is primary" rule.

2. **Identity by `dyn_hash`, not `Debug`.** `AnyKind` and `AnyOperation` each
   get `fn dyn_hash(&self, state: &mut dyn Hasher)`; the content/identity hash
   folds those directly. Drops the current `format!("{:?}")`-then-hash waste
   (string formatting just to feed a hasher).

3. **`View` via a small builder.** `View::structured("PointListView").param(cap)`
   centralises the Slang-type formatting instead of hand-built strings at each
   call site — keeps the open set open while giving `cargo check` one place to
   guard.
