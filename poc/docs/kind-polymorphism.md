# Kind Polymorphism — Reusing Operations Across Datatypes

Status: **design proposal — ready for implementation**
Companion to `docs/core-simplification.md` (depends on nothing from it except one
trivial builder addition, noted in §6; reads better after C4's `ViewAdapter` but does
not require it).

## 1. The problem, concretely

Suppose we add **video** to the engine. A `VideoFrame` is a pixel plane plus timing
metadata:

```text
VideoFrame = { pixel plane (w×h, PixelFormat, ColorSpace), pts, duration, frame_index }
```

We want **all ~200 image operations** (blur, exposure, composite, geometry, …) to work
on frames *without writing a frame variant of any of them*, and without the core ever
learning what a frame is. Concretely this should compile:

```rust
let frame: VideoFrame2D<GpuBackend> = video.frame_at(pts)?;
let processed: VideoFrame2D<GpuBackend> =
    frame.map_image(|img| img.exposure(0.5, 0.0).blur(1.2));
// `processed` still carries pts/duration/frame_index — and pulls through
// a video Target later.
```

The key observation that makes this cheap: **image operations never touch the input
Kind at lowering time.** On the GPU, a kernel reads its input through whatever
`IRegion` wrapper the *source leaf* declared (`GpuView::input` on the leaf's Kind);
the op contributes only its kernel call. On vips, ops consume an opaque `VipsHandle`.
So if a frame's buffer *is* a valid image plane, an image pipeline can run on it
unmodified — the only missing piece is a typed way to say so in the graph.

## 2. Existing precedents (what the engine already does)

The codebase already has two of the three casting mechanisms; this design adds the
third and names the taxonomy:

| Mechanism | Runtime cost | Kind | Buffer layout | Existing example |
|---|---|---|---|---|
| `ViewAdapter` (alias/swizzle/remap) | zero — no kernel, no buffer | unchanged | unchanged | `ExtractBand` reads channel 2 of an RGBA decode |
| **`Reinterpret` (this doc)** | **zero — no kernel, no buffer** | **changes** | **identical bytes** | Frame → Image, Image → Frame |
| Cross-Kind `Operation` | one kernel | changes | changes | `HistogramOp` (Image → Histogram) |

`Reinterpret` is the missing middle: a *typed cast between Kinds whose payloads are
byte-identical*. Everything heavier than that is just a normal cross-Kind operation
(already supported, see `HistogramOp`); everything lighter is a `ViewAdapter`.

## 3. Options considered

### Option A — handle composition, no new Kind

```rust
pub struct VideoFrame<B: Backend> {
    pub image: Image2D<B>,
    pub meta: FrameMeta, // pts, duration, frame_index — plain host data
}
```

Frames never enter the graph; metadata rides next to the handle in host code.

- **Pro:** zero engine changes; maximally simple; composition-over-inheritance.
- **Con (fatal):** the frame is not a graph value, so nothing downstream can be typed
  on it — no `Target<VideoFrameKind, B>` (video encoder sink), no
  `Source` producing frames, no future temporal op (`Operation` consuming
  `Input<VideoFrameKind>` × N for motion blur / temporal denoise), no caching keyed on
  frame identity. Every consumer must hand-carry `meta` and the pairing is by
  convention, not by type.

Rejected as the *general* model, but note: nothing forbids a thin struct like this in
app code on top of Option C. The engine just shouldn't be limited by it.

### Option B — make image ops generic over an `ImageLike` trait

```rust
pub struct Blur<K: ImageLike, B: Backend> { input: Input<K, B>, sigma: f32 }
```

- **Con (fatal):** infects every operation struct, every `impl Operation`, every
  `impl Lower`, and every ergonomic method with an extra type parameter (~200 ops);
  monomorphizes the whole op library per consuming Kind; and still doesn't answer how
  the *output* gets frame metadata back. Maximum churn, minimum leverage.

Rejected.

### Option C — `ReinterpretAs` + a single generic `Reinterpret` node ✅

The user-stated idea, formalized: *the Kind itself declares, in Rust, how its data is
interpreted as another Kind*. One declarative trait + one generic, datatype-agnostic
graph node that costs nothing at runtime. Image ops stay exactly as they are; the
frame enters their world through a typed cast node and leaves through another.

Chosen. Detailed below.

## 4. The design

### 4.1 The capability trait (core, `operation/reinterpret.rs` — new file)

```rust
/// `Self` declares that its payload bytes are a valid `T` payload, and how to
/// derive the `T` spec from its own. A pure metadata statement — no compute,
/// no buffer transformation. Both Kinds must divide into the same WorkUnit
/// shape (enforced by the bound: you cannot reinterpret a Region-shaped value
/// as an Atomic one).
pub trait ReinterpretAs<T>: Kind
where
    T: Kind<WorkUnit = Self::WorkUnit>,
{
    fn reinterpret_spec(&self) -> T;
}
```

Notes:

- The `T: Kind<WorkUnit = Self::WorkUnit>` bound makes shape-mismatched casts a
  **compile error**, not a runtime check. (Frame and Image are both `Region`-shaped.)
- "Byte-identical payload" is the impl's contract. It is not memory-unsafe to get it
  wrong (buffers are bounds-checked views), but it is semantically wrong — the doc
  comment must say so, and `byte_size` equality is debug-asserted at the node (4.2).

### 4.2 The generic cast node (core, same file)

```rust
/// A zero-cost typed cast in the graph: output Kind differs, payload is the
/// input's payload, untouched. Lowering forwards the input — no kernel on the
/// GPU, handle passthrough on vips.
pub struct Reinterpret<K: Kind, T: Kind, B: Backend> {
    pub input: Input<K, B>,
    pub spec: T,
}

impl<K, T, B> Operation<B> for Reinterpret<K, T, B>
where
    K: Kind,
    T: Kind<WorkUnit = K::WorkUnit>,
    B: Backend,
    Self: Lower<B>,
{
    type Output = T;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &T::WorkUnit) -> Vec<Option<WorkUnit>> {
        // Same shape, same extent: identity passthrough.
        let wu = out.erase();
        debug_assert_eq!(
            self.input.spec.byte_size(&wu),
            self.spec.byte_size(&wu),
            "Reinterpret requires byte-identical payloads"
        );
        vec![Some(wu)]
    }

    fn output_spec(&self) -> T {
        self.spec.clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        self.spec.dyn_hash(state);
    }
}
```

Note the struct does **not** require `K: ReinterpretAs<T>` — the trait is only the
*safe constructor* path (4.4). This keeps an explicit-spec escape hatch (4.5, rewrap)
without a second node type.

### 4.3 Lowering — generic over all Kinds, written once

**GPU.** The node adds no step; its consumers must resolve to the node's input. That
is one new builder method (the same node-resolution map `alias` already uses):

```rust
// backend/gpu/mod.rs
impl GpuBuilder {
    /// The current node's value IS its single input's value — no kernel, no
    /// temp, no adapter. Consumers resolving this node get its input instead.
    pub fn forward(&mut self) -> &mut Self {
        let Some(input) = self.cur_inputs.first().cloned() else {
            self.fail(Error::Backend("forward: node has no input".into()));
            return self;
        };
        if let Some(k) = self.cur_node {
            self.forwarded.insert(k, input); // consulted in enter()'s resolution
        }
        self
    }
}
```

(`enter`'s input-resolution chain checks `forwarded` first, before
`alias`/`source_of`/`last_step_of`. After `core-simplification.md` C4 this is the
adapter map with `adapter: None`. ~10 lines either way.)

```rust
impl<K: Kind, T: Kind, B> Lower<GpuBackend> for Reinterpret<K, T, GpuBackend> { … }
// body:
fn lower(&self, cx: &mut GpuBuilder) {
    cx.forward();
}
```

One subtlety: if the `Reinterpret` is the **DAG root** (user pulls the cast itself,
e.g. `frame.as_image().pull(...)`), there is no consumer and no output registration.
Two cases:

- Root cast *into* a Kind whose `GpuView::output` is an encode sandwich (Image): the
  zero-step encode path already handles "root is a plain source read"; `forward`
  resolves the root to that source/step, and the cast's `T: GpuView` provides the
  output wrap. `Reinterpret::lower` therefore also registers the output when it is
  root — same pattern every op uses: `cx.output(self.spec.output())`. Since `forward`
  marks the node as produced-by-its-input, the encode reads the right value. No
  special emitter code.
- In practice the dominant path is cast → image ops → cast back → frame Target, where
  the casts are interior nodes and this case never fires; but it must work for tests.

**Vips.**

```rust
impl<K: Kind, T: Kind> Lower<VipsBackend> for Reinterpret<K, T, VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let h = cx.input(self.input.src());
        cx.emit(h); // same handle, new Kind — vips never looks at the Kind
    }
}
```

Both impls are datatype-blind → they live in core legitimately (invariant: core speaks
`Kind`, never a concrete datatype — satisfied).

### 4.4 The user-facing surface (core, `node.rs`)

```rust
impl<K: Kind, B: Backend> Data<K, B> {
    /// Zero-cost typed cast, derived from the Kind's own declaration.
    pub fn reinterpret<T>(&self) -> Data<T, B>
    where
        K: ReinterpretAs<T>,
        T: Kind<WorkUnit = K::WorkUnit>,
        Reinterpret<K, T, B>: Lower<B>,
    {
        let spec = self.spec.reinterpret_spec();
        self.push(Reinterpret { input: self.as_input(), spec })
    }

    /// Zero-cost cast with an explicit target spec — the caller asserts byte
    /// compatibility (used for the rewrap direction, where the target spec
    /// carries data the source Kind doesn't have, e.g. frame timing).
    pub fn reinterpret_with<T>(&self, spec: T) -> Data<T, B>
    where
        T: Kind<WorkUnit = K::WorkUnit>,
        Reinterpret<K, T, B>: Lower<B>,
    {
        self.push(Reinterpret { input: self.as_input(), spec })
    }
}
```

### 4.5 The worked example — `data/video_frame.rs` (future module)

Everything below is *additive*: one new file, zero core edits.

```rust
/// Frame = image plane + timing. The plane's layout is literally ImageKind —
/// composition, so reinterpretation is definitionally byte-identical.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoFrameKind {
    pub image: ImageKind,
    pub pts: i64,
    pub duration: i64,
    pub frame_index: u64,
}

impl AnyKind for VideoFrameKind {
    fn as_any(&self) -> &dyn Any { self }
    fn byte_size(&self, wu: &WorkUnit) -> u64 { self.image.byte_size(wu) } // plane only
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        self.image.dyn_hash(state);
        state.write_i64(self.pts);
        state.write_u64(self.frame_index);
    }
}

impl Kind for VideoFrameKind {
    type WorkUnit = Region; // same shape as the image plane
}

/// GPU representation: a frame *is* its plane on the GPU — delegate wholesale.
/// This is "the Frame tells the engine how to read its bytes", declared once.
impl GpuView for VideoFrameKind {
    fn input(&self) -> View { self.image.input() }
    fn output(&self) -> OutputWrap { self.image.output() }
}

/// The declaration that unlocks the entire image-op library:
impl ReinterpretAs<ImageKind> for VideoFrameKind {
    fn reinterpret_spec(&self) -> ImageKind { self.image.clone() }
}

pub type VideoFrame2D<B> = Data<VideoFrameKind, B>;

impl<B: Backend> VideoFrame2D<B>
where
    Reinterpret<VideoFrameKind, ImageKind, B>: Lower<B>,
    Reinterpret<ImageKind, VideoFrameKind, B>: Lower<B>,
{
    /// View this frame's plane as an image. Zero cost; lazy like everything.
    pub fn as_image(&self) -> Image2D<B> {
        self.reinterpret()
    }

    /// Run an image pipeline over the plane, keep the timing. The image spec
    /// may legitimately change (crop, resize) — the rewrapped frame adopts it.
    pub fn map_image(&self, f: impl FnOnce(Image2D<B>) -> Image2D<B>) -> VideoFrame2D<B> {
        let out = f(self.as_image());
        let spec = VideoFrameKind { image: (*out.spec).clone(), ..(*self.spec).clone() };
        out.reinterpret_with(spec)
    }
}
```

Resulting graph for the §1 snippet — note the casts are *nodes* but cost nothing
(GPU: the whole chain still fuses into **one** kernel pass; vips: same handle chain):

```text
VideoFileSource ─ Reinterpret(→Image) ─ Exposure ─ Blur ─ Reinterpret(→Frame) ─ [video Target]
                  no kernel                kernel   kernel  no kernel
```

Sources and sinks round it out (sketch, not part of this change):

```rust
/// One leaf per decoded frame — pts is source config, not WorkUnit payload,
/// so the core's WorkUnit vocabulary is untouched.
pub struct VideoFrameSource { /* demuxer handle, pts */ }
impl Source<GpuBackend> for VideoFrameSource { type Kind = VideoFrameKind; … }

pub struct VideoEncoderTarget { /* muxer */ }
impl Target<VideoFrameKind, GpuBackend> for VideoEncoderTarget {
    type Out = (); // side effect: encode + mux, reading pts/duration from the spec
    …
}
```

## 5. Decisions and their rationale

1. **Metadata lives in the Kind, never in the payload buffer.** The Kind already
   flows through the graph (`output_spec`, `Buffer.spec`, `Data.spec`) and is the
   typed metadata channel; buffers stay raw planes that any backend can bind blindly.
   If metadata were in the buffer, `Reinterpret` would need offsets and every image
   kernel would read garbage. This single decision is what makes the cast free.
2. **Shape equality is a trait bound, not a runtime check.**
   `T: Kind<WorkUnit = Self::WorkUnit>` — casting a Region-shaped frame to an
   Atomic-shaped histogram doesn't compile. Byte-size equality (which can depend on
   runtime spec values) is a `debug_assert` in `demand`.
3. **The rewrap direction is `reinterpret_with`, not `impl ReinterpretAs<VideoFrameKind> for ImageKind`.**
   An image alone cannot derive pts/duration — the target spec needs information only
   the caller (the original frame) has. So `ReinterpretAs` covers the
   information-losing direction; the information-adding direction takes an explicit
   spec. `map_image` packages the round trip so users never touch either primitive.
4. **`Reinterpret` does not require `ReinterpretAs`** — the trait gates only the
   convenience constructor. One node type serves both directions; no `Rewrap` twin.
5. **When the layouts *don't* match, this model degrades gracefully into the existing
   one.** A 10-bit planar YUV frame is not byte-compatible with any `ImageKind` —
   then you simply don't implement `ReinterpretAs`, and the bridge is an ordinary
   cross-Kind `Operation` with a real conversion kernel
   (`YuvToImage { input: Input<VideoFrameKind, B> } → Output = ImageKind`), exactly
   the `HistogramOp` pattern. Reinterpret is an optimization for the
   layout-compatible case, not the only door.
6. **Frame identity participates in hashing.** `dyn_hash` includes pts/frame_index,
   so two casts of different frames through identical image pipelines produce
   different node hashes — any future content-addressed cache stays correct.
7. **No `Shape`/`WorkUnit` extension for time.** Frame selection (`pts`) is source
   configuration (`video.frame_at(t)` creates a leaf bound to `t`), not a work-unit
   dimension. Extending `WorkUnit` with time would ripple through every demand walk
   for zero benefit to spatial ops. Revisit only if/when a *temporal* op needs to
   demand "frames t-1..t+1" from one source node — that's the cross-Kind-operation
   tier anyway.

## 6. Required modifications (implementation checklist)

Core (datatype-agnostic, ~90 lines total):

1. `src/operation/reinterpret.rs` — `ReinterpretAs<T>`, `Reinterpret<K, T, B>`,
   `impl Operation<B>`, generic `impl Lower<GpuBackend>` + `impl Lower<VipsBackend>`.
2. `src/node.rs` — `Data::{reinterpret, reinterpret_with}`.
3. `src/backend/gpu/mod.rs` — `GpuBuilder::forward()` + `forwarded` map consulted in
   `enter`'s resolution chain (before `source_of`/`last_step_of`; merges into the
   adapter map if `core-simplification.md` C4 lands first).
4. `src/operation/mod.rs` — `pub mod reinterpret; pub use reinterpret::*;`.

Proof tests (use **existing** Kinds — no video needed to validate the mechanism):

5. `tests/gpu_probe.rs` — define a local test Kind `TaggedImageKind { image: ImageKind, tag: u32 }`
   (delegating `AnyKind`/`GpuView`, `ReinterpretAs<ImageKind>`) and assert:
   - `tagged.reinterpret::<ImageKind>().invert().pull(...)` equals
     `plain.invert().pull(...)` byte-for-byte (cast is transparent to compute);
   - the emitted Slang for cast→invert→cast-back contains exactly one kernel call
     (cast adds no step);
   - root-cast pull works (`tagged.reinterpret::<ImageKind>().pull(...)`).

Future, when video actually lands (zero core changes by construction):

6. `src/data/video_frame.rs` as in §4.5 + demuxer/encoder source/target.

## 7. Relationship to `core-simplification.md`

Independent but synergistic:

- `forward()` is the only shared touchpoint; with C4 it's the adapter map's
  `adapter: None` entry, without C4 it's a sibling map to `alias_swizzles`.
- C5's explicit dispatch domain makes the root-cast case cleaner (the cast registers
  output + dispatch like any op) but is not required.
- This doc *adds* the third row of the casting taxonomy (§2); C4 cleans up the first
  row. Together they form the complete polymorphism story: **view within a Kind
  (adapter), cast between layout-equal Kinds (reinterpret), convert between
  layout-different Kinds (cross-Kind op)** — all three datatype-blind in the core,
  all three declared by the datatype modules.
