# Chromors Node Editor — Detailed Implementation Refinement

> **Audience:** an implementer (human or AI) who will write the code with **zero
> room for interpretation**. Every type, field, algorithm, and coordinate
> convention is spelled out. Where a decision could go two ways, this doc picks
> one and states it as a RULE.
>
> **Prerequisite reading:** `CLAUDE.md` (the engine contract) and
> `docs/ARCHITECTURE.md`. This document NEVER changes the engine's rules. The
> node editor is a **consumer** of the engine, living entirely in the
> `chromors-viewer` crate. It builds the engine's immutable `Arc<Node<B>>` DAG;
> it never reaches inside it.

---

## 0. Glossary — two words that both mean "node"

This is the single most important distinction in the whole document. Read it
twice.

| Term used in this doc | What it is | Mutable? | Lives in |
|---|---|---|---|
| **EditorNode** | A box on the canvas the user drags around (e.g. a "Blur" box). UI model. | Yes | `chromors-viewer` |
| **engine node** (`Node<B>`) | An immutable `Arc<Node<B>>` in the lazy DAG (`src/node.rs`). | No (Arc, rebuilt) | `poc` crate |

> **RULE G0.** Never store an `Arc<Node<B>>` as the identity of an EditorNode.
> The editor graph is the source of truth; the engine DAG is a **compiled
> artifact** regenerated from it. One EditorNode may compile to several engine
> nodes (e.g. a "Load" EditorNode compiles to a `VipsImageSource` + a GPU bridge
> + a `Convert`).

When this doc says "node" unqualified, it means **EditorNode**.

---

## 1. Goals & non-goals

### 1.1 Goals
1. A Blender/Chainner-style **node graph editor** where the user composes the
   engine's operations visually.
2. **Multiple live viewports**, each bound to any node's output socket, updating
   as the graph changes.
3. A **left palette** of draggable node types (Chainner style): drag a palette
   entry onto the canvas to instantiate an EditorNode.
4. **Vello-drawn** canvas (nodes, wires, sockets, widgets, text). No external
   immediate-mode GUI toolkit.
5. **Idiomatic Rust**: data-oriented, `SlotMap`-keyed graph, explicit interaction
   state machine, no `Rc<RefCell<...>>` spaghetti, no global mutable singletons.
6. **Incremental & lazy**: changing one parameter recompiles only the affected
   subgraph and re-pulls only the affected viewports.

### 1.2 Non-goals (explicitly out of scope for v1)
- Undo/redo (designed-for in §13.4 but not implemented in v1).
- Serialization to disk (graph save/load) — interfaces reserved in §13.3.
- Node groups / subgraphs / macros.
- Animation / timeline.
- Touch / multi-touch gestures.

---

## 2. Dependencies

Add to `chromors-viewer/Cargo.toml`:

```toml
slotmap = "1.0"       # stable keys for nodes/edges/panels
parley  = "0.2"       # text layout for canvas labels & widgets
# vello, wgpu, tao, poc, chromors-viewport, bytemuck, pollster already present
```

> **RULE D1.** `slotmap` is mandatory. Do **not** use `Vec<EditorNode>` +
> indices (indices shift on removal) nor `HashMap<u64, _>` with hand-rolled ids.
> `SlotMap` gives stable, generational keys that survive deletion — exactly what
> wires need to reference endpoints safely.

> **RULE D2.** Text is rendered through `parley` (layout) → `vello::Scene`
> glyph run. All text drawing goes through the helper in §10.6 — node code never
> touches `parley` directly.

---

## 3. Crate / module layout

All new code is under `chromors-viewer/src/`. The existing `app.rs`, `gpu.rs`,
`main.rs` are extended/replaced as noted. `chromors-viewport` is **not modified**
except where §9.3 explicitly says so.

```
chromors-viewer/src/
  main.rs                  # entry point (mostly unchanged: logging + App::run)
  gpu.rs                   # GpuState (unchanged)
  app.rs                   # SHRINKS: window/event plumbing only, delegates to Editor

  editor/
    mod.rs                 # pub use; the top-level `Editor` struct (§12)
    graph.rs               # NodeGraph, EditorNode, Edge, ids, mutations (§4)
    types.rs               # DataType, PortValue, PortId, socket model (§5)
    registry.rs            # NodeKind, NodeDescriptor, NODE_REGISTRY (§6)
    descriptors/           # one file per group of node descriptors (§6.4)
      mod.rs               #   registers all of them
      sources.rs           #   Load, Constant, Gradient...
      color.rs             #   Exposure, Brightness, Saturation, Gamma, Invert...
      filters.rs           #   Blur, Sharpen...
      geometry.rs          #   Crop, Resize, Flip, Rotate...
      sinks.rs             #   Viewer (output)
    compile.rs             # editor graph -> engine DAG; eval & memo (§7)
    params.rs              # ParamValue, ParamSpec, widget model (§8)

  ui/
    mod.rs                 # pub use
    layout.rs              # DockLayout: panel tree, hit-testing regions (§9)
    panel.rs               # Panel enum + trait; routing (§9.2)
    canvas.rs              # NodeCanvas: graph-space camera, draw, interaction (§10,§11)
    palette.rs             # PalettePanel: draggable node list (§11.5 + §12.4)
    inspector.rs           # InspectorPanel: selected node's params (§8.4)
    viewport_panel.rs      # ViewportPanel: wraps chromors_viewport::ViewportRenderer (§9.3)
    theme.rs               # colors, sizes, fonts — single source of visual constants (§10.1)
    text.rs                # parley+vello text helper (§10.6)
    widgets.rs             # slider/number/dropdown/color drawn in Vello + hit model (§8.3)
    input.rs               # InputEvent, ModifierState, MouseButton normalization (§11.1)
```

> **RULE M1.** `editor/` knows nothing about pixels-on-screen (no wgpu, no
> Vello, no panel bounds). It is the *model + engine-compilation* half. `ui/`
> knows nothing about how the engine works (no `poc::operation::*`); it talks to
> `editor/` through `PortValue` and the registry. This mirrors the engine's own
> "two halves" rule and keeps the AI from tangling them.

---

## 4. The graph model (`editor/graph.rs`)

### 4.1 Keys

```rust
use slotmap::{SlotMap, new_key_type};

new_key_type! {
    pub struct NodeKey;   // identifies an EditorNode
    pub struct EdgeKey;   // identifies a wire
}
```

### 4.2 Sockets — addressing

A socket is addressed by `(NodeKey, side, index)`.

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Side { In, Out }

/// A fully-qualified socket address. `index` is the ordinal within that side,
/// matching the order in the node's `NodeDescriptor` (§6).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PortAddr {
    pub node: NodeKey,
    pub side: Side,
    pub index: u16,
}
```

> **RULE S1.** Socket order is fixed by the descriptor and never reordered at
> runtime. `index` is therefore a stable identity for a socket *within a node of
> a given kind*. An edge stores two `PortAddr`s.

### 4.3 EditorNode

```rust
use crate::editor::params::ParamValue;
use crate::editor::registry::NodeKindId;

pub struct EditorNode {
    /// Which descriptor in the registry this is an instance of.
    pub kind: NodeKindId,
    /// Canvas position of the node's top-left corner, in GRAPH SPACE units
    /// (see §10.2). NOT screen pixels.
    pub pos: glam::Vec2,
    /// Current parameter values, indexed identically to the descriptor's
    /// `params` Vec. Length == descriptor.params.len() at all times.
    pub params: Vec<ParamValue>,
    /// User-editable display title; defaults to descriptor.title.
    pub title: String,
    /// Collapsed nodes draw only the title bar + sockets (no widgets).
    pub collapsed: bool,
    /// Cached, recomputed layout (socket positions, body height). See §10.4.
    /// `None` means "dirty, recompute before drawing/hit-testing".
    pub layout_cache: Option<NodeLayout>,
}
```

> Use `glam` for `Vec2`/`Affine2`? **RULE 4A.** Use `vello::kurbo` types
> (`kurbo::Point`, `kurbo::Vec2`, `kurbo::Affine`) everywhere for geometry, so no
> conversion is needed when drawing. Replace `glam::Vec2` above with
> `kurbo::Point` (position) / `kurbo::Vec2` (offsets). `kurbo` is already a
> transitive dep via vello and is re-exported as `vello::kurbo`. Do **not** add
> `glam`.

Corrected field: `pub pos: vello::kurbo::Point,`.

### 4.4 Edge

```rust
pub struct Edge {
    pub from: PortAddr, // must be Side::Out
    pub to:   PortAddr, // must be Side::In
}
```

### 4.5 NodeGraph

```rust
pub struct NodeGraph {
    pub nodes: SlotMap<NodeKey, EditorNode>,
    pub edges: SlotMap<EdgeKey, Edge>,
    /// Monotonic counter bumped on ANY structural or param change. Viewports
    /// compare against their last-seen value to know they must recompile. See §7.5.
    pub revision: u64,
}
```

#### 4.5.1 Invariants enforced by the mutation API

> **RULE G1 (single-driver inputs).** An input socket accepts **at most one**
> incoming edge. Connecting a second edge to an occupied input first removes the
> existing one.

> **RULE G2 (no self/cycle).** `connect` must reject an edge that would create a
> cycle. Detection: see §7.2 (the topo sort doubles as the cycle check, but
> `connect` does a cheap reachability test first).

> **RULE G3 (type match).** `connect` must reject an edge whose output
> `DataType` is not accepted by the input socket (`DataType::accepts`, §5.3).

#### 4.5.2 Mutation API (the ONLY way to change the graph)

```rust
impl NodeGraph {
    pub fn new() -> Self { /* empty, revision 0 */ }

    pub fn add_node(&mut self, kind: NodeKindId, pos: Point) -> NodeKey;
    pub fn remove_node(&mut self, key: NodeKey);   // also removes incident edges

    /// Returns Err with a reason if any of G1/G2/G3 is violated. On success,
    /// enforces G1 by evicting a pre-existing edge into `to`.
    pub fn connect(&mut self, from: PortAddr, to: PortAddr) -> Result<EdgeKey, ConnectError>;
    pub fn disconnect(&mut self, edge: EdgeKey);

    pub fn set_param(&mut self, node: NodeKey, index: usize, value: ParamValue);
    pub fn move_node(&mut self, node: NodeKey, new_pos: Point);

    /// The edge whose `to == addr`, if any (inputs are single-driver, G1).
    pub fn incoming(&self, addr: PortAddr) -> Option<(EdgeKey, PortAddr /*from*/)>;
    /// All edges whose `from == addr` (outputs fan out freely).
    pub fn outgoing(&self, addr: PortAddr) -> impl Iterator<Item = (EdgeKey, PortAddr /*to*/)>;
}
```

> **RULE G4.** *Every* method that changes structure or params bumps
> `self.revision += 1` and invalidates the affected nodes' `layout_cache`
> (`move_node` invalidates only that node's wires, not layout; param changes that
> alter widget count — none in v1 — would invalidate layout). `add_node`,
> `remove_node`, `connect`, `disconnect`, `set_param` all bump revision.

`ConnectError`:

```rust
pub enum ConnectError {
    TypeMismatch { out: DataType, in_: DataType },
    WouldCycle,
    NotAnOutput,    // from.side != Out
    NotAnInput,     // to.side != In
    SameNode,       // from.node == to.node
}
```

---

## 5. The socket type system (`editor/types.rs`)

This is the editor-side type layer that decides which wires are legal and what
flows through them. It is allowed to be a closed enum **because it lives in the
app, not the engine** (the engine's "no central enum" rule, CLAUDE.md §6, binds
only `poc`). Adding a new flowing type = one new variant here + handling it in
the descriptors that use it.

### 5.1 DataType — the static socket type

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DataType {
    Image,      // poc::data::image::Image2D<GpuBackend>
    Mask,       // single-channel image (Image with 1-band layout); see §5.4
    Scalar,     // f64 (knob output, e.g. a "Number" node)
    Color,      // [f32; 4] linear RGBA
    // future: Histogram, Vectorscope, Points, Vector(vello::Scene)
}
```

### 5.2 PortValue — the runtime value carried on a wire during evaluation

```rust
use poc::backend::gpu::GpuBackend;
use poc::data::image::Image2D as GpuImage; // = Data<ImageKind, GpuBackend>

#[derive(Clone)]
pub enum PortValue {
    Image(GpuImage<GpuBackend>),
    Mask(GpuImage<GpuBackend>),   // same handle type; semantic 1-band contract
    Scalar(f64),
    Color([f32; 4]),
}

impl PortValue {
    pub fn data_type(&self) -> DataType { /* match */ }
    /// Typed extractors used by descriptor build closures. They PANIC on
    /// mismatch — by the time a build closure runs, the type system (§7.1) has
    /// already guaranteed the variant. (Panicking is correct here: a mismatch
    /// is a registry bug, not a user error.)
    pub fn image(&self) -> &GpuImage<GpuBackend>;
    pub fn scalar(&self) -> f64;
    pub fn color(&self) -> [f32; 4];
}
```

> **RULE T1.** `PortValue::Image` clones are cheap — `Data<K,B>` is `Arc`-backed
> (see `src/node.rs::Data: Clone`). Clone freely; never try to share by
> reference across the graph.

### 5.3 Acceptance rule

```rust
impl DataType {
    /// Can an output of type `self` feed an input declared as `other`?
    /// v1: exact match only, EXCEPT a Mask may feed an Image input
    /// (a 1-band image is a valid image). Image may NOT feed a Mask input.
    pub fn accepts(self, producer: DataType) -> bool {
        self == producer
            || (self == DataType::Image && producer == DataType::Mask)
    }
}
```
Here `self` is the **input** socket's declared type and `producer` is the
**output** socket's type.

### 5.4 Mask contract
A `Mask` is physically an `Image2D<GpuBackend>` whose `ImageKind.layout` has
`model == ColorModel::Gray` and one band. Descriptors that emit masks must
guarantee that layout. Consumers that need a mask but receive an `Image` (legal
per §5.3 only in the Image←Mask direction, not here) — not applicable in v1;
masks only originate from mask-producing nodes.

---

## 6. Node descriptor registry (`editor/registry.rs`)

This is **how a new node type is added** — the single extension point. The AI
adds a `NodeDescriptor` to a file under `descriptors/`; everything else
(palette entry, sockets, inspector widgets, compilation) is automatic.

### 6.1 Identity

```rust
/// Stable string id, e.g. "color.exposure". Used as the registry key and (later)
/// for serialization. MUST be unique and stable across versions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeKindId(pub &'static str);
```

### 6.2 Socket & param specs

```rust
use crate::editor::types::DataType;
use crate::editor::params::ParamSpec;

pub struct SocketSpec {
    pub name: &'static str,   // shown next to the socket
    pub ty: DataType,
}

pub struct NodeDescriptor {
    pub id: NodeKindId,
    pub title: &'static str,      // default node title + palette label
    pub category: Category,       // palette grouping (§6.3)
    pub inputs:  Vec<SocketSpec>,
    pub outputs: Vec<SocketSpec>, // v1: most nodes have exactly 1 output
    pub params:  Vec<ParamSpec>,  // ordered; EditorNode.params mirrors this
    /// The compile closure: given resolved input values + current params,
    /// produce each output value. See §7.4 for the exact contract.
    pub build: BuildFn,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category { Source, Color, Filter, Geometry, Composite, Sink }
```

### 6.3 The build function signature

```rust
use poc::backend::gpu::GpuContext;
use std::sync::Arc;

/// Inputs are in socket order. A `None` means "input socket is unconnected".
/// `params` is the node's current param values (== descriptor.params order).
/// `ctx` is the shared GPU context (needed by source nodes to bridge from Vips).
///
/// Returns one PortValue per OUTPUT socket, in order. On user error
/// (e.g. required input missing) return Err(BuildError) — the node renders in
/// an error state (§10.5) and downstream evaluation of that branch is skipped.
pub type BuildFn = fn(
    inputs: &[Option<PortValue>],
    params: &[ParamValue],
    ctx: &Arc<GpuContext>,
) -> Result<Vec<PortValue>, BuildError>;

pub struct BuildError(pub String);
```

> **RULE B1.** A `build` closure is **pure** with respect to the engine: it only
> *constructs* lazy `Data` handles (calls like `.blur()`, `.exposure()`); it must
> never call `.pull()` / `.materialize()` / download. Evaluation stays lazy until
> a viewport pulls (§9.3). The one allowed exception is a **Source** node, which
> may open a file via Vips and bridge to a GPU source — still lazy on the GPU
> side (`Data::from_source`).

### 6.4 The registry

```rust
use std::collections::HashMap;
use std::sync::OnceLock;

pub struct Registry {
    by_id: HashMap<NodeKindId, NodeDescriptor>,
    ordered: Vec<NodeKindId>, // palette display order == insertion order
}

impl Registry {
    pub fn get(&self, id: NodeKindId) -> &NodeDescriptor;
    pub fn iter(&self) -> impl Iterator<Item = &NodeDescriptor>;
    pub fn iter_category(&self, c: Category) -> impl Iterator<Item = &NodeDescriptor>;
}

/// Global, built once. All descriptors/*.rs register into this.
pub fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(build_registry)
}

fn build_registry() -> Registry {
    let mut r = Registry::default();
    crate::editor::descriptors::sources::register(&mut r);
    crate::editor::descriptors::color::register(&mut r);
    crate::editor::descriptors::filters::register(&mut r);
    crate::editor::descriptors::geometry::register(&mut r);
    crate::editor::descriptors::sinks::register(&mut r);
    r
}
```

### 6.5 Worked descriptor examples (COPY THESE PATTERNS EXACTLY)

`descriptors/color.rs`:

```rust
pub fn register(r: &mut Registry) {
    // ── Exposure ──────────────────────────────────────────────────────────
    r.add(NodeDescriptor {
        id: NodeKindId("color.exposure"),
        title: "Exposure",
        category: Category::Color,
        inputs:  vec![SocketSpec { name: "image", ty: DataType::Image }],
        outputs: vec![SocketSpec { name: "out",   ty: DataType::Image }],
        params: vec![
            ParamSpec::float("stops",    -5.0, 5.0, 0.0),
            ParamSpec::float("preserve",  0.0, 1.0, 0.0),
        ],
        build: |inputs, params, _ctx| {
            let img = inputs[0].as_ref()
                .ok_or(BuildError("exposure: 'image' not connected".into()))?
                .image().clone();
            let stops    = params[0].float() as f32;
            let preserve = params[1].float() as f32;
            Ok(vec![PortValue::Image(img.exposure(stops, preserve))])
        },
    });

    // ── Invert (no params) ────────────────────────────────────────────────
    r.add(NodeDescriptor {
        id: NodeKindId("color.invert"),
        title: "Invert",
        category: Category::Color,
        inputs:  vec![SocketSpec { name: "image", ty: DataType::Image }],
        outputs: vec![SocketSpec { name: "out",   ty: DataType::Image }],
        params: vec![],
        build: |inputs, _p, _ctx| {
            let img = inputs[0].as_ref()
                .ok_or(BuildError("invert: 'image' not connected".into()))?
                .image().clone();
            Ok(vec![PortValue::Image(img.invert())])
        },
    });
}
```

`descriptors/sources.rs` — the Source node (the ONLY place file I/O + Vips→GPU
bridge happens):

```rust
pub fn register(r: &mut Registry) {
    r.add(NodeDescriptor {
        id: NodeKindId("source.load"),
        title: "Load Image",
        category: Category::Source,
        inputs:  vec![],
        outputs: vec![SocketSpec { name: "image", ty: DataType::Image }],
        params: vec![ParamSpec::path("file")],
        build: |_inputs, params, ctx| {
            use poc::data::image::{Image2D, VipsImageSource};
            use poc::backend::vips::VipsBackend;
            use poc::node::Data;
            use std::sync::Arc;

            let path = params[0].path()
                .ok_or(BuildError("load: no file selected".into()))?;
            let vips = Image2D::<VipsBackend>::open(path)
                .map_err(|e| BuildError(format!("open failed: {e:?}")))?;
            let src = VipsImageSource::new(vips);
            let gpu_img = Data::from_source(Arc::new(src), ctx.clone());
            Ok(vec![PortValue::Image(gpu_img)])
        },
    });
}
```

`descriptors/sinks.rs` — the Viewer node (graph sink that a viewport binds to):

```rust
pub fn register(r: &mut Registry) {
    r.add(NodeDescriptor {
        id: NodeKindId("sink.viewer"),
        title: "Viewer",
        category: Category::Sink,
        inputs:  vec![SocketSpec { name: "image", ty: DataType::Image }],
        outputs: vec![], // a sink produces nothing for the graph
        params: vec![],
        // A Viewer's build just PASSES THROUGH its input so compile.rs can read
        // the value at its input socket. (See §7.4: sinks are evaluated like any
        // node; the bound ViewportPanel reads PortValue at the viewer's input.)
        build: |inputs, _p, _ctx| {
            let img = inputs[0].as_ref()
                .ok_or(BuildError("viewer: nothing connected".into()))?
                .clone();
            Ok(vec![img]) // expose the (passed-through) value as output[0]
        },
    });
}
```

> **RULE R1.** The Viewer descriptor declares 0 outputs in `outputs` (no wire
> drags out of it) but its `build` returns a 1-element Vec. This is intentional:
> the compiler still records `output[0]` internally so a `ViewportPanel` can read
> it (§7.4 stores values per `(NodeKey, out_index)` regardless of whether the
> descriptor exposes that socket for wiring). The canvas simply never draws an
> output socket for a node whose `descriptor.outputs` is empty.

---

## 7. Compilation & evaluation (`editor/compile.rs`)

This turns the editor graph into evaluated `PortValue`s by running each node's
`build` in dependency order. It is the bridge into the engine's lazy DAG.

### 7.1 Static type validation (done at connect time AND before eval)

Connection legality (G3) is checked in `connect`. The compiler additionally
treats a required-but-unconnected input as a build error at eval time (the
`build` closure returns `Err`).

### 7.2 Topological order & cycle detection

```rust
/// Returns nodes in dependency order (inputs before consumers), or Err with the
/// node that closes a cycle. Standard Kahn's algorithm over the editor graph:
///   in_degree(n) = number of connected INPUT sockets of n that have an edge.
///   adjacency: edge from A.out -> B.in contributes A -> B.
pub fn topo_order(g: &NodeGraph) -> Result<Vec<NodeKey>, NodeKey>;
```

`connect`'s cheap pre-check (§4.5.1 G2): before inserting edge `from.node ->
to.node`, do a DFS from `to.node` following outgoing edges; if it reaches
`from.node`, reject `WouldCycle`.

### 7.3 The evaluation cache

```rust
use std::collections::HashMap;

/// Output values keyed by (producing node, output socket index).
pub struct EvalCache {
    values: HashMap<(NodeKey, u16), PortValue>,
    /// errors per node, for rendering the error state.
    errors: HashMap<NodeKey, BuildError>,
    /// the graph revision this cache was built from.
    built_revision: u64,
}
```

> **RULE E1 (lazy memo at the handle level, not the value level).** Because every
> `PortValue::Image` is a lazy `Data` handle, re-running a whole `build` pass is
> cheap (it only re-assembles `Arc`s; no pixels move). v1 therefore recomputes the
> **entire** `EvalCache` whenever `graph.revision` changes. Do NOT prematurely
> implement per-node incremental invalidation; the laziness already makes a full
> recompile O(nodes) of pointer work. (§13.1 notes the future incremental path.)

### 7.4 The eval pass

```rust
pub fn evaluate(g: &NodeGraph, ctx: &Arc<GpuContext>) -> EvalCache {
    let mut cache = EvalCache::new(g.revision);
    let order = match topo_order(g) {
        Ok(o) => o,
        Err(_cycle) => return cache, // leave empty; UI shows cycle warning
    };
    for key in order {
        let node = &g.nodes[key];
        let desc = registry().get(node.kind);

        // Gather inputs in socket order.
        let mut inputs: Vec<Option<PortValue>> = Vec::with_capacity(desc.inputs.len());
        let mut upstream_failed = false;
        for in_idx in 0..desc.inputs.len() as u16 {
            let addr = PortAddr { node: key, side: Side::In, index: in_idx };
            match g.incoming(addr) {
                Some((_edge, from)) => {
                    match cache.values.get(&(from.node, from.index)) {
                        Some(v) => inputs.push(Some(v.clone())),
                        None => { upstream_failed = true; inputs.push(None); }
                    }
                }
                None => inputs.push(None),
            }
        }
        if upstream_failed {
            cache.errors.insert(key, BuildError("upstream error".into()));
            continue;
        }

        match (desc.build)(&inputs, &node.params, ctx) {
            Ok(outs) => {
                for (i, v) in outs.into_iter().enumerate() {
                    cache.values.insert((key, i as u16), v);
                }
            }
            Err(e) => { cache.errors.insert(key, e); }
        }
    }
    cache
}
```

### 7.5 When does eval run?
- The `Editor` (§12) holds `Option<EvalCache>`.
- Each frame, if `cache.is_none()` or `cache.built_revision != graph.revision`,
  call `evaluate` and store it.
- After (re)eval, for every `ViewportPanel` bound to a `(NodeKey, out_index)`,
  fetch the `PortValue::Image` from the cache and push it to the panel's
  `ViewportRenderer` (§9.3). Skip panels whose bound value is missing/errored
  (keep showing the previous frame — `replace_image`/`update_composite` already
  does soft-swap, §9.3).

---

## 8. Parameters & widgets (`editor/params.rs`, `ui/widgets.rs`)

### 8.1 ParamSpec — declared in the descriptor

```rust
pub enum ParamSpec {
    Float { name: &'static str, min: f64, max: f64, default: f64 },
    Int   { name: &'static str, min: i64, max: i64, default: i64 },
    Bool  { name: &'static str, default: bool },
    Choice{ name: &'static str, options: &'static [&'static str], default: usize },
    Color { name: &'static str, default: [f32; 4] },
    Path  { name: &'static str }, // file picker; default = None
}

impl ParamSpec {
    pub fn float(name: &'static str, min: f64, max: f64, default: f64) -> Self { /* */ }
    pub fn int(/* */) -> Self;
    pub fn choice(/* */) -> Self;
    pub fn color(/* */) -> Self;
    pub fn path(name: &'static str) -> Self;
    pub fn name(&self) -> &'static str;
    pub fn default_value(&self) -> ParamValue;
}
```

### 8.2 ParamValue — stored per node instance

```rust
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Choice(usize),       // index into ParamSpec::Choice.options
    Color([f32; 4]),
    Path(Option<String>),
}

impl ParamValue {
    pub fn float(&self) -> f64;       // panics if not Float (registry bug)
    pub fn int(&self) -> i64;
    pub fn bool(&self) -> bool;
    pub fn choice(&self) -> usize;
    pub fn color(&self) -> [f32; 4];
    pub fn path(&self) -> Option<&str>;
}
```

> **RULE P1.** `EditorNode.params[i]` corresponds 1:1 to
> `descriptor.params[i]`. On `add_node`, initialize `params` from each spec's
> `default_value()`. The two Vecs always have equal length.

### 8.3 Widget model (drawn in Vello — `ui/widgets.rs`)

Widgets are **immediate-mode draw + retained hit regions**. A widget is drawn
into a `Scene` at a rect; the same pass records a `WidgetHit` so the interaction
layer can route clicks/drags.

```rust
/// One interactive control's screen-space rect + how to interpret a drag on it.
pub struct WidgetHit {
    pub rect: kurbo::Rect,            // SCREEN space (already transformed)
    pub target: WidgetTarget,
}

pub enum WidgetTarget {
    /// param at (node, index); drag maps horizontal delta -> value within min..max
    ParamDrag  { node: NodeKey, index: usize },
    ParamToggle{ node: NodeKey, index: usize },           // bool
    ParamCycle { node: NodeKey, index: usize, options: usize }, // choice (click cycles / opens menu)
    ParamColor { node: NodeKey, index: usize },           // opens color popup
    ParamPath  { node: NodeKey, index: usize },           // opens file dialog
    Socket(PortAddr),                                      // wire drag start/end
    NodeBody(NodeKey),                                     // node drag / select
    NodeTitle(NodeKey),                                    // node drag + double-click rename
    NodeCollapse(NodeKey),                                 // collapse toggle
}
```

> **RULE W1.** Each frame the canvas rebuilds a fresh `Vec<WidgetHit>` while it
> draws (painter's order). Hit-testing iterates this Vec **in reverse** (topmost
> first) so overlapping widgets resolve to the front-most. The Vec is the single
> source of truth for "what is under the cursor"; there is no separate retained
> hit tree.

### 8.4 Inspector panel (`ui/inspector.rs`)
Shows the params of the **single selected** node (multi-select shows nothing in
v1) as a vertical list of labelled widgets. The same `WidgetTarget` model is
reused; the inspector contributes its own `WidgetHit`s into the active panel's
hit list. Editing a slider here calls `graph.set_param(...)` exactly like editing
the inline widget on the node body.

### 8.5 Float drag math (so two implementers produce identical feel)

```text
On ParamDrag start: record start_value, start_cursor_x.
On drag move to cursor_x:
    span      = max - min
    px_range  = 200.0                      // full range crossed over 200 screen px
    delta     = (cursor_x - start_cursor_x) / px_range * span
    if shift_held { delta *= 0.1 }         // fine adjust
    new = clamp(start_value + delta, min, max)
    graph.set_param(node, index, ParamValue::Float(new))
Int params: same, then round to nearest integer before clamping.
```

---

## 9. Window layout & multiple viewports (`ui/layout.rs`, `ui/panel.rs`)

### 9.1 DockLayout — a binary split tree

```rust
pub enum DockNode {
    Split {
        dir: SplitDir,         // Horizontal = children side-by-side; Vertical = stacked
        ratio: f32,            // 0..1 fraction given to `first`
        first: Box<DockNode>,
        second: Box<DockNode>,
    },
    Leaf(PanelId),
}

#[derive(Clone, Copy, PartialEq)] pub enum SplitDir { Horizontal, Vertical }

new_key_type! { pub struct PanelId; }
```

`DockLayout` owns the root `DockNode` and a `SlotMap<PanelId, Panel>`.

```rust
pub struct DockLayout {
    pub root: DockNode,
    pub panels: SlotMap<PanelId, Panel>,
}

impl DockLayout {
    /// Compute each panel's pixel rect by walking the tree against the window
    /// rect. Returns (PanelId, Rect) for every leaf. Splitter gutter = 4px
    /// (subtracted from both sides of a split). Called every frame and on resize.
    pub fn solve(&self, window: kurbo::Rect) -> Vec<(PanelId, kurbo::Rect)>;

    /// Panel under a screen point (topmost leaf containing it), for event routing.
    pub fn hit(&self, layout: &[(PanelId, kurbo::Rect)], p: kurbo::Point) -> Option<PanelId>;
}
```

> **RULE L1.** `solve` is pure and cheap; call it once per frame, cache the
> `Vec<(PanelId, Rect)>` for that frame, and reuse it for both event routing and
> drawing. Splitter dragging (resizing panels) mutates the corresponding
> `Split.ratio`; v1 may ship with **fixed ratios** (no draggable splitters) and
> add dragging later — but the tree/solve design already supports it.

### 9.2 Panel — the four kinds

```rust
pub enum Panel {
    Canvas(NodeCanvas),         // the node graph editor (§10)
    Viewport(ViewportPanel),    // a live image preview bound to a viewer (§9.3)
    Palette(PaletteState),      // draggable node list (§11.5)
    Inspector(InspectorState),  // selected node params (§8.4)
}
```

Default v1 layout (built in `Editor::new`):

```text
            window
   ┌──────────┬───────────────────────┬───────────┐
   │ Palette  │      Node Canvas       │ Inspector │
   │ (Source) │  (DockNode::Leaf)      │           │
   │  ...     ├───────────────────────┤           │
   │          │   Viewport (Viewer A) │           │
   └──────────┴───────────────────────┴───────────┘
```
Encoded as nested splits:
`Split(H, 0.15, Palette, Split(H, 0.82, Split(V, 0.6, Canvas, Viewport), Inspector))`.

Adding a **second viewport** = insert another `Viewport` leaf bound to a
different Viewer node (e.g. split the existing Viewport leaf). The user triggers
this via a command (§12.4) "Add Viewport for selected Viewer".

### 9.3 ViewportPanel — wraps the existing renderer

```rust
use chromors_viewport::{ViewportRenderer, ViewportBounds};

pub struct ViewportPanel {
    pub renderer: ViewportRenderer,
    /// Which graph output this panel shows.
    pub bound: Option<(NodeKey, u16)>,
    /// Tracks the last revision pushed so we only re-push on change.
    last_pushed_revision: u64,
    has_image: bool,
}

impl ViewportPanel {
    pub fn new(device, queue, format, bounds: ViewportBounds) -> Self;

    /// Called after eval (§7.5). Pulls the bound image out of the cache and
    /// pushes it to the renderer using the SOFT-SWAP path so the preview never
    /// blinks while editing.
    pub fn sync(&mut self, cache: &EvalCache, revision: u64) {
        let Some((node, out)) = self.bound else { return };
        if revision == self.last_pushed_revision { return; }
        if let Some(PortValue::Image(img)) = cache.values.get(&(node, out)) {
            let display = img.convert(display_layout()); // §9.4
            if self.has_image {
                self.renderer.update_composite(display); // soft replace_image
            } else {
                self.renderer.attach_image(display);
                self.renderer.fit();
                self.has_image = true;
            }
            self.last_pushed_revision = revision;
        }
    }
}
```

> **RULE V1.** Each `ViewportPanel` owns its **own** `ViewportRenderer` (its own
> camera, atlas, fetcher). They share the wgpu `Device`/`Queue` and the
> `GpuContext`. Two viewports of the same Viewer have independent pan/zoom — this
> is the point of multiple viewports.

> **RULE V2.** `ViewportRenderer::bounds` (a `ViewportBounds { x, y, w, h }`) is
> set from the panel's solved pixel rect **every frame before draw** so the image
> renders only inside the panel. The existing `draw(enc, view, clip_x, clip_y,
> clip_w, clip_h)` scissor params are also set to the panel rect so nothing
> bleeds outside it (see `renderer.rs:715` `draw` — it already supports a clip
> rect and viewport bounds).

> **RULE V3 (single surface, many panels).** There is ONE wgpu surface (the
> window). Each frame, after clearing, iterate panels and have each draw into the
> shared surface view within its scissor rect. The node canvas draws via Vello
> (render to an offscreen texture sized to the panel, then blit into the panel
> rect — reuse the `VelloOverlay` two-phase pattern in
> `chromors-viewport/src/vello_overlay.rs`). The viewport panels draw via their
> `ViewportRenderer::draw`. Order: clear whole surface → viewports → canvas →
> palette/inspector (Vello) → drag ghost overlay.

### 9.4 display_layout()
Identical to the current `app.rs::display_layout()` (straight-alpha sRGB
RGBA8 U8). Move it into `ui/viewport_panel.rs` and reuse.

---

## 10. Drawing the node canvas (`ui/canvas.rs`)

### 10.1 Theme constants (`ui/theme.rs`)
Single source of every size/color. (Values are defaults; tune freely.)

```rust
pub const NODE_WIDTH: f64 = 180.0;          // graph-space units
pub const NODE_TITLE_H: f64 = 26.0;
pub const NODE_ROW_H: f64 = 22.0;           // per socket row / widget row
pub const NODE_CORNER: f64 = 6.0;
pub const SOCKET_R: f64 = 5.0;
pub const SOCKET_HIT_R: f64 = 10.0;         // generous hit radius
pub const WIRE_WIDTH: f64 = 2.0;
pub const GRID_STEP: f64 = 32.0;

pub const COL_BG:        Color = Color::from_rgb8(30, 30, 34);
pub const COL_GRID:      Color = Color::from_rgb8(40, 40, 46);
pub const COL_NODE_BODY: Color = Color::from_rgb8(52, 52, 60);
pub const COL_NODE_SEL:  Color = Color::from_rgb8(255, 180, 40);
pub const COL_WIRE:      Color = Color::from_rgb8(200, 200, 210);
pub const COL_TEXT:      Color = Color::from_rgb8(230, 230, 235);
// per-DataType socket colors:
pub fn socket_color(t: DataType) -> Color { /* Image=blue, Mask=gray, Scalar=green, Color=pink */ }
// per-Category title-bar colors:
pub fn category_color(c: Category) -> Color { /* Source=teal, Color=purple, ... */ }
```

### 10.2 Coordinate spaces (CRITICAL — get this exactly right)

Three spaces:

1. **Graph space** — where node positions live. Units are arbitrary "world"
   units. Independent of zoom.
2. **Panel space** — pixels within the canvas panel, origin at the panel's
   top-left. `(0,0)` .. `(panel_w, panel_h)`.
3. **Screen space** — pixels in the window. `panel_space + panel_rect.origin`.

The canvas has a camera identical in spirit to
`chromors_viewport::Camera` (reuse the same math, §10.3):

```rust
pub struct GraphCamera {
    pub pan: kurbo::Vec2, // graph-space point shown at panel's top-left
    pub zoom: f64,        // graph units -> panel pixels multiplier
}
impl GraphCamera {
    // graph -> panel
    pub fn g2p(&self, p: Point) -> Point {
        Point::new((p.x - self.pan.x) * self.zoom, (p.y - self.pan.y) * self.zoom)
    }
    // panel -> graph (for hit-testing cursor)
    pub fn p2g(&self, p: Point) -> Point {
        Point::new(p.x / self.zoom + self.pan.x, p.y / self.zoom + self.pan.y)
    }
    pub fn affine(&self) -> Affine {
        Affine::scale(self.zoom) * Affine::translate(-self.pan.to_vec2())
    }
}
```

> **RULE C1.** Draw the whole graph scene in graph space using `affine()` as the
> scene transform (pass it to `scene.append(&local, Some(affine))` exactly like
> `vello_overlay.rs:149`). Widget hit rects (§8.3), however, are stored in
> **screen** space (`affine` applied + panel origin added) so the interaction
> layer compares them directly against the raw cursor position.

### 10.3 Reuse from chromors-viewport
The pan/zoom-about-cursor and zoom spring logic in
`chromors-viewport/src/controller.rs` (`on_scroll`, `update_physics`) is exactly
what the canvas camera wants. **RULE C2.** Copy that math into a
`CanvasController` (don't depend on `ViewportController`, which is image-specific
and references `ViewportRenderer`). Zoom anchors at the cursor: the graph point
under the cursor stays fixed across a zoom step (same formula as
`controller.rs:132-137`).

### 10.4 NodeLayout — computed geometry (cached on EditorNode)

```rust
pub struct NodeLayout {
    pub size: kurbo::Size,            // graph-space w x h
    pub title_rect: kurbo::Rect,      // graph space, relative to node.pos
    pub input_sockets:  Vec<Point>,   // graph-space socket CENTERS (abs)
    pub output_sockets: Vec<Point>,
    pub widget_rows: Vec<kurbo::Rect>,// graph-space rect per param row (abs)
}
```

Layout algorithm (deterministic — both implementers must produce identical
geometry):

```text
let d = descriptor(node.kind);
let n_in  = d.inputs.len();
let n_out = d.outputs.len();
let n_rows_sockets = max(n_in, n_out);
let n_rows_widgets = if node.collapsed { 0 } else { d.params.len() };

height = NODE_TITLE_H
       + n_rows_sockets as f64 * NODE_ROW_H
       + n_rows_widgets as f64 * NODE_ROW_H
       + 8.0; // bottom padding
size = (NODE_WIDTH, height)

title_rect = Rect::new(0, 0, NODE_WIDTH, NODE_TITLE_H)            // node-relative

// Inputs on the LEFT edge, outputs on the RIGHT edge, one per row,
// starting just below the title:
for i in 0..n_in:
    y = NODE_TITLE_H + (i + 0.5) * NODE_ROW_H
    input_sockets[i]  = node.pos + (0.0, y)            // x = left edge
for o in 0..n_out:
    y = NODE_TITLE_H + (o + 0.5) * NODE_ROW_H
    output_sockets[o] = node.pos + (NODE_WIDTH, y)     // x = right edge

// Widget rows below the socket rows:
base_y = NODE_TITLE_H + n_rows_sockets * NODE_ROW_H
for k in 0..n_rows_widgets:
    widget_rows[k] = Rect at node.pos + (8, base_y + k*NODE_ROW_H),
                     width NODE_WIDTH-16, height NODE_ROW_H-4
```

Recompute when `layout_cache.is_none()`. Invalidate on collapse toggle, title
change, kind change (never changes), and (defensively) on first draw.

### 10.5 Draw order for one node
1. Body rounded-rect (`COL_NODE_BODY`; if selected, stroke `COL_NODE_SEL` 2px; if
   `cache.errors` has this node, stroke red and tint title bar red).
2. Title bar (rounded top, `category_color`), title text (`text::draw`, §10.6).
3. Collapse chevron at title's right (records `NodeCollapse` hit).
4. For each input socket: filled circle `socket_color(ty)`, label to its right.
5. For each output socket: filled circle, label to its left.
6. If not collapsed: each param widget via `widgets::draw_*` into its
   `widget_rows[k]` rect (records the appropriate `WidgetTarget`).
7. Record `NodeBody`/`NodeTitle` hits (title rect first so it wins for dragging).
8. Socket hits recorded LAST (so sockets win over body, since hit list is scanned
   in reverse — §8.3 W1; sockets are appended after body → scanned first).

### 10.6 Text helper (`ui/text.rs`)

```rust
/// Lay out and draw a single line of text into `scene` at `origin` (the text
/// baseline-left, in the scene's current coordinate space) with `size` (graph
/// units) and `color`. Internally caches a parley FontContext + a shaped-line
/// LRU keyed by (string, size_bucket).
pub fn draw_line(scene: &mut Scene, origin: Point, text: &str, size: f64, color: Color);
/// Measured advance width of `text` at `size`, for centering / truncation.
pub fn measure(text: &str, size: f64) -> f64;
```

> **RULE TX1.** Embed one font (e.g. `Inter` or any redistributable TTF) via
> `include_bytes!` so there is no system-font dependency. Register it once in a
> `OnceLock<FontContext>`. Truncate labels with `…` when wider than their slot
> (use `measure`).

### 10.7 Wires
For each edge, get `from` output-socket center and `to` input-socket center
(graph space, from the two nodes' `NodeLayout`). Draw a cubic Bézier — reuse the
exact handle-offset convention so all wires look consistent:

```text
let dx = (to.x - from.x).abs().max(40.0) * 0.5;
c1 = from + (dx, 0)
c2 = to   - (dx, 0)
BezPath: move_to(from); curve_to(c1, c2, to);
stroke width WIRE_WIDTH, color COL_WIRE (or socket_color of the output type).
```
This is the same primitive as `chromors-viewport/src/vector.rs::BezierGraphic` —
you may render wires by constructing `BezierGraphic`s, or inline the path. Inline
is simpler here since wires are part of the canvas scene, not the overlay.

### 10.8 Background grid
Draw before nodes: vertical+horizontal lines every `GRID_STEP` graph units,
color `COL_GRID`, within the visible graph-space rect (`p2g` of the panel
corners). Snap line positions to multiples of `GRID_STEP`.

---

## 11. Interaction (`ui/canvas.rs` state machine, `ui/input.rs`)

### 11.1 Normalized input

```rust
pub struct ModifierState { pub ctrl: bool, pub shift: bool, pub alt: bool }

pub enum InputEvent {
    PointerMoved { screen: Point },
    PointerDown  { screen: Point, button: PtrButton },
    PointerUp    { screen: Point, button: PtrButton },
    Scroll       { screen: Point, delta_y: f32 },
    Key          { code: KeyCode, pressed: bool },
}
pub enum PtrButton { Left, Middle, Right }
```

`app.rs` translates `tao` events into `InputEvent` + maintains `ModifierState`,
then routes to `Editor::on_input` (§12.3).

### 11.2 Canvas interaction state machine

```rust
pub enum CanvasInteraction {
    Idle,
    PanningCanvas { last: Point },                       // middle-drag or space-drag
    DraggingNodes { anchor_graph: Point, last_graph: Point }, // moves all selected
    DraggingWire  { from: PortAddr, cursor: Point },     // rubber-band a new wire
    DraggingParam { node: NodeKey, index: usize, start_val: f64, start_x: f64 },
    BoxSelect     { start: Point, current: Point },      // screen space
}
```

Selection set lives on the canvas: `pub selected: HashSet<NodeKey>`.

### 11.3 Transition table (EXHAUSTIVE — implement exactly)

Resolve the cursor's hit via the current frame's `Vec<WidgetHit>` (reverse scan).
Let `hit` = topmost `WidgetTarget` under the pointer (or `None`).

**State Idle:**
| Event | Condition | Action → New state |
|---|---|---|
| PointerDown Left | hit = Socket(out) | begin `DraggingWire { from: out }` |
| PointerDown Left | hit = Socket(in) that is occupied | detach its edge, begin `DraggingWire { from: the freed output }` (re-wire gesture) |
| PointerDown Left | hit = Socket(in) empty | begin `DraggingWire`? **No** — wires only start from outputs. Treat as no-op. |
| PointerDown Left | hit = NodeTitle/NodeBody(n) | if `!shift`: set selection = {n}; else toggle n in selection. begin `DraggingNodes` |
| PointerDown Left | hit = NodeCollapse(n) | toggle `n.collapsed`, invalidate layout, stay Idle |
| PointerDown Left | hit = ParamDrag(n,i) | begin `DraggingParam` (record start value/x) |
| PointerDown Left | hit = ParamToggle | flip bool param, `set_param`, stay Idle |
| PointerDown Left | hit = ParamCycle | advance choice (mod options), `set_param`, stay Idle |
| PointerDown Left | hit = ParamPath | open `rfd` file dialog (async, §12.5), stay Idle |
| PointerDown Left | hit = ParamColor | open color popup (§8.3; v1 may stub to a fixed cycle), stay Idle |
| PointerDown Left | hit = None | clear selection, begin `BoxSelect { start }` |
| PointerDown Middle | any | begin `PanningCanvas { last }` |
| Scroll | any | zoom about cursor (§10.3 anchored), stay Idle |
| Key Delete | selection non-empty | `remove_node` each selected, clear selection |
| Key F | one node selected with image output | (optional) bind nearest viewport to it |

**State DraggingWire { from, cursor }:**
| Event | Action |
|---|---|
| PointerMoved | update `cursor`; draw rubber-band Bézier from `from` socket to cursor |
| PointerUp Left over Socket(in) | attempt `graph.connect(from, in)`; on Err flash the input red briefly; → Idle |
| PointerUp Left elsewhere | → Idle (wire discarded) |

**State DraggingNodes { anchor, last }:**
| Event | Action |
|---|---|
| PointerMoved | `delta = p2g(cursor) - last`; for each selected node `move_node(n, pos+delta)`; update `last` |
| PointerUp Left | → Idle |

**State DraggingParam:**
| Event | Action |
|---|---|
| PointerMoved | apply §8.5 drag math, `set_param` |
| PointerUp Left | → Idle |

**State PanningCanvas { last }:**
| Event | Action |
|---|---|
| PointerMoved | `camera.pan -= (cursor - last)/zoom`; update last |
| PointerUp Middle | → Idle |

**State BoxSelect { start, current }:**
| Event | Action |
|---|---|
| PointerMoved | update `current`; live-highlight nodes whose graph-space bbox intersects the rect |
| PointerUp Left | commit selection = nodes intersecting rect (additive if shift); → Idle |

> **RULE I1.** Param edits and structural edits go **only** through `NodeGraph`'s
> mutation API (§4.5.2). The state machine never writes `node.params[i]`
> directly — it calls `graph.set_param`, which bumps `revision` so the next frame
> re-evals and the bound viewports refresh.

### 11.4 Hover feedback
On every `PointerMoved` in Idle, recompute the topmost hit and store
`hovered: Option<WidgetTarget>`; draw hovered sockets/nodes with a subtle
highlight. (No retained dirty-region machinery needed — the canvas redraws each
frame; see §14 perf note.)

### 11.5 Palette → canvas drag (Chainner-style)
The palette panel (§12.4) lists registry descriptors grouped by `Category`. The
drag crosses panels, so it is owned by the **Editor**, not a single panel:

```rust
pub struct PaletteDrag {
    pub kind: NodeKindId,
    pub cursor: Point,      // screen space
}
```

- PointerDown on a palette row → `Editor.palette_drag = Some(PaletteDrag{kind, cursor})`.
- PointerMoved → update `cursor`; draw a translucent ghost node at the cursor
  (drawn last, on top of everything, in §9.3 V3 order).
- PointerUp:
  - if cursor is over the Canvas panel → `let g = canvas.camera.p2g(cursor -
    canvas_panel_origin); graph.add_node(kind, g);` select the new node.
  - else → discard.

---

## 12. The top-level `Editor` (`editor/mod.rs` orchestration; owned by `app.rs`)

### 12.1 Struct

```rust
pub struct Editor {
    pub graph: NodeGraph,
    pub cache: Option<EvalCache>,
    pub layout: DockLayout,
    pub gpu_ctx: Arc<GpuContext>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_format: wgpu::TextureFormat,

    pub palette_drag: Option<PaletteDrag>,
    pub modifiers: ModifierState,
    pub focused_panel: Option<PanelId>,   // panel that receives keyboard input
    pub solved: Vec<(PanelId, kurbo::Rect)>, // this frame's panel rects (§9.1 L1)
}
```

### 12.2 Lifecycle
- `Editor::new(device, queue, format, gpu_ctx)` builds the default layout (§9.2),
  creates one `Canvas`, one `Viewport` (unbound), one `Palette`, one `Inspector`,
  and an empty graph.
- Optional: seed the graph with a `source.load` + `sink.viewer` already wired, and
  bind the viewport to the viewer, so first run shows something after the user
  picks a file.

### 12.3 Per-event entry

```rust
impl Editor {
    pub fn on_input(&mut self, ev: InputEvent) {
        // 1. Update modifiers if it's a Key event.
        // 2. If palette_drag is active, the Editor handles move/up itself (§11.5).
        // 3. Else route by panel: find panel under the event's screen point using
        //    self.solved; translate screen->panel-local; dispatch to that Panel's
        //    handler (canvas.on_input / viewport.on_input / palette / inspector).
        // 4. Canvas/viewport handlers mutate self.graph via the mutation API.
    }
}
```

> **RULE 12A.** Scroll/keyboard go to the panel under the cursor (or
> `focused_panel` for keys after a click). A **viewport** panel's scroll =
> image zoom (drive its own `ViewportController`/camera, reuse
> `controller.rs`); a **canvas** panel's scroll = graph zoom. They never cross.

### 12.4 Per-frame render

```rust
pub fn render(&mut self, encoder, surface_view, window_rect) {
    self.solved = self.layout.solve(window_rect);

    // (a) eval if dirty
    let rev = self.graph.revision;
    if self.cache.as_ref().map_or(true, |c| c.built_revision != rev) {
        self.cache = Some(evaluate(&self.graph, &self.gpu_ctx));
    }
    let cache = self.cache.as_ref().unwrap();

    // (b) sync each viewport panel from the cache
    for (pid, rect) in &self.solved {
        if let Panel::Viewport(vp) = &mut self.layout.panels[*pid] {
            vp.renderer.bounds = rect_to_bounds(*rect);
            vp.renderer.resize(rect.width(), rect.height());
            vp.sync(cache, rev);
            vp.renderer.prepare();
        }
    }

    // (c) draw: viewports (wgpu), then canvas+palette+inspector (Vello), then ghost
    for (pid, rect) in &self.solved {
        match &mut self.layout.panels[*pid] {
            Panel::Viewport(vp) => vp.renderer.draw(encoder, surface_view,
                                       rect.x0 as u32, rect.y0 as u32,
                                       rect.width() as u32, rect.height() as u32),
            Panel::Canvas(c)    => c.draw(&self.graph, cache, *rect, /*vello*/),
            Panel::Palette(p)   => p.draw(*rect),
            Panel::Inspector(i) => i.draw(&self.graph, &self.selected(), *rect),
        }
    }
    if let Some(drag) = &self.palette_drag { draw_ghost(drag); }
}
```

> **RULE 12B.** All Vello panels (canvas/palette/inspector + ghost) can share one
> offscreen Vello texture the size of the **window**, rendered once per frame
> with one `vello::Scene` that places each panel's sub-scene at its rect (clip to
> the rect with `scene.push_layer` + a rect clip). Then a single blit composites
> it over the surface (after the viewport draws). This is one Vello submit per
> frame — cheapest and simplest. (Alternative: per-panel textures; do NOT do this
> in v1.)

### 12.5 Async file dialog
Reuse `app.rs`'s existing pattern: spawn a thread running
`rfd::FileDialog::pick_file()`, send the `PathBuf` back over an
`mpsc::channel`; poll the receiver each frame; on receipt call
`graph.set_param(node, index, ParamValue::Path(Some(path)))`.

---

## 13. Reserved interfaces (design-for, don't build in v1)

### 13.1 Incremental eval
`EvalCache` already keys by `(NodeKey, out)`. When wanted: track a per-node
`dirty` set, recompute only dirty nodes + their descendants. The lazy-handle
design (E1) makes this low priority.

### 13.2 Background materialization
Viewports already fetch tiles on a worker (`fetcher/`). No change needed; the
node editor just hands fresh `Data` handles to `ViewportRenderer`.

### 13.3 Serialization
`NodeKindId` is a stable string; `ParamValue` is serde-friendly; positions are
plain floats; edges are `(NodeKindId-keyed)` addresses. Add `serde` derives to
`EditorNode`/`Edge`/`ParamValue` and a `GraphDoc { nodes: Vec<(NodeKindId, Point,
Vec<ParamValue>)>, edges: Vec<(usize,u16,usize,u16)> }` when wanted.

### 13.4 Undo/redo
Wrap every mutation-API call in a `Command` that records its inverse. The
mutation API being the *single* write path (RULE I1) is what makes this clean
later.

---

## 14. Performance notes (so the AI doesn't over-engineer)

1. **Redraw every frame is fine.** The canvas is vector and small; Vello handles
   it. Do not port the viewport's dirty-region overlay caching to the canvas in
   v1.
2. **Eval is pointer work.** Re-running `evaluate` on every revision change is
   cheap because no pixels move (RULE E1). Pixels only move when a viewport pulls
   tiles, which is already throttled by the fetcher.
3. **Request redraw on demand.** Mirror `app.rs`'s `request_frame` model: redraw
   when (a) an input event mutated state, (b) any viewport `is_fetching()`, or
   (c) a viewport camera spring is active. Otherwise idle (don't spin).
4. **One Vello submit/frame** (RULE 12B). One surface (RULE V3).

---

## 15. Implementation milestones (build in THIS order; each compiles & runs)

1. **M1 — skeleton.** `editor/types.rs`, `graph.rs` (model + mutation API + tests
   for G1–G4), `params.rs`, `registry.rs` with 3 descriptors (load, exposure,
   viewer). No UI. Unit-test `evaluate` produces a `PortValue::Image` from
   load→exposure→viewer. (Pull it once in the test via a `GpuBufferTarget` to
   prove the DAG is real — test only; app never pulls.)
2. **M2 — single canvas, no panels.** Whole window = node canvas. Draw grid +
   nodes + wires + sockets (Vello, `theme.rs`, `text.rs`). Camera pan/zoom. No
   editing yet. Hard-code two nodes.
3. **M3 — interaction.** Implement the §11.3 state machine: drag nodes, drag
   wires (connect/disconnect with type checks), box select, delete.
4. **M4 — params & inspector.** Widgets (§8.3), inline node widgets, inspector
   panel, float/int drag (§8.5), path picker.
5. **M5 — docking + one viewport.** `DockLayout` (fixed ratios), add a
   `ViewportPanel`, bind it to the viewer node, `sync` after eval. Confirm
   load→exposure→viewer shows a live image that updates as you drag the exposure
   slider.
6. **M6 — palette + drag-instantiate.** `PalettePanel`, cross-panel
   `PaletteDrag`, ghost node, drop-to-create.
7. **M7 — multiple viewports.** Command to split a viewport / add a second Viewer
   + viewport. Confirm independent pan/zoom.
8. **M8 — fill out descriptors.** Add the rest of the engine ops (filters,
   geometry, arithmetic, composite) as descriptors. Each is ~15 lines (§6.5).

After each milestone: `cargo build -p chromors-viewer` must be 0 errors and the
binary must run.

---

## 16. Quick reference — engine APIs the descriptors call

(From `src/data/image.rs`, `src/operation/*`. All on `Image2D<GpuBackend>`,
return a new `Image2D<GpuBackend>`. All lazy.)

| Node | Engine call | Params |
|---|---|---|
| Load | `Image2D::<VipsBackend>::open(path)` → `VipsImageSource::new` → `Data::from_source(Arc::new(src), ctx)` | path |
| Exposure | `.exposure(stops: f32, preserve: f32)` | 2 floats |
| Brightness | `.brightness(gain: f32)` | 1 float (1.0 neutral) |
| Saturation | `.saturation(amount: f32)` | 1 float (1.0 neutral) |
| Gamma | `.gamma(Some(exp: f64))` | 1 float (1.0 neutral) |
| Linear | `.linear(a: Vec<f64>, b: Vec<f64>)` | per-channel a,b |
| Invert | `.invert()` | — |
| Blur | `.blur(sigma: f32)` | 1 float |
| Sharpen | `.sharpen(...)` (see `filters.rs:400`) | see signature |
| Crop | `.crop(left,top,width,height: i32)` | 4 ints |
| Extract area | `.extract_area(l,t,w,h)` | 4 ints |
| Flip | `.flip(Direction)` | choice |
| Rotate | `.rotate(...)` (see `geometry.rs:1826`) | see signature |
| Resize | `.resize(...)` (see `geometry.rs:1888`) | see signature |
| Shrink | `.shrink(h: f64, v: f64, ceil: Option<bool>)` | 2 floats |
| Extract band | `.extract_band(band: i32, count: Option<i32>)` → Mask | 1–2 ints |
| Add/Sub/Mul | `.add(&other)` / `.subtract(&other)` / `.multiply(&other)` | 2 image inputs |
| Convert | `.convert(layout: PixelLayout)` | (used internally by viewport, §9.4) |

> When a signature is unclear, OPEN the cited file and read the `pub fn` — do not
> guess argument order. The table's line numbers are from the current tree.

---

## 17. Hard rules recap (the things that break the design if violated)

- **G0** EditorNode ≠ engine node; the DAG is compiled, never stored as identity.
- **M1** `editor/` (model+engine) and `ui/` (pixels) never mix.
- **B1** `build` closures stay lazy — never pull/materialize/download.
- **I1** all graph changes go through `NodeGraph`'s mutation API; it owns `revision`.
- **V1/V2/V3** each viewport = own `ViewportRenderer`, drawn clipped to its panel,
  one shared surface.
- **S1/P1** socket and param indices are descriptor-fixed and 1:1 with instance Vecs.
- **E1** recompile the whole `EvalCache` on revision change (laziness makes it cheap).
- **C1** draw the canvas in graph space via the camera affine; store hit rects in
  screen space.
- Engine rules in `CLAUDE.md` are untouched — the editor only *consumes* `Data`.
```
