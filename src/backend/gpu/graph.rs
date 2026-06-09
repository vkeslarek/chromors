// ── NodeEval ──────────────────────────────────────────────────────────────────

/// Evaluation strategy for a graph node.
///
/// Breaking the old hard coupling "every node == one Slang kernel call" into
/// an explicit strategy enum makes it possible to extend the graph with
/// non-kernel nodes in later phases (View, Host, Reduction).
#[derive(Clone, Debug)]
pub enum NodeEval {
    /// Standard fused Slang kernel dispatch.  The most common variant —
    /// corresponds to the old `kernel: KernelSpec` field.
    Kernel(super::graph::KernelSpec),
    // Future variants (Phase C / D):
    // View(ChannelRewrite),         — read-side fusion, no dispatch, no temp
    // Reduction(KernelSpec),        — image → non-image (histogram, scalar, …)
    // Host(Arc<dyn HostOp>),        — pure CPU execution on materialised inputs
}

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use crate::backend::gpu::source::{AnyGpuSource, GpuSource};

pub use super::value::ValueKind;

/// A shader kernel specification — module + function name in Slang.
#[derive(Clone, Debug)]
pub struct KernelSpec {
    pub module: &'static str,
    pub function: &'static str,
}

/// Unique identifier for a graph node.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct NodeId(pub u32);

/// A node in the computation graph.
#[derive(Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub inputs: Vec<NodeId>,
    /// Evaluation strategy — what the node does at dispatch time.
    pub eval: NodeEval,
    /// The operation that created this node. Used for `inverse_map` during
    /// the materialize walk, and queried for `output_codec_override`.  Always set.
    pub op: Arc<dyn super::op::GpuOperation>,
    pub params: Vec<super::param::Param>,
    /// What kind of value this node outputs.
    pub output: ValueKind,
}

/// A source (leaf) node — provides input pixels to the graph.
#[derive(Clone)]
pub struct SourceNode {
    pub id: NodeId,
    pub source: Arc<super::source::GpuSource>,
}

/// Cache key for a materialised region: content hash + rect (x, y, w, h).
/// The leading `u64` is a content-addressed identity (see [`Graph::content_hash`]),
/// not a node id — so identical computations share cache entries across graph
/// forks and sessions.
pub type RegionKey = (u64, i32, i32, i32, i32);

#[derive(Clone)]
pub struct Graph {
    pub nodes: Vec<GraphNode>,
    pub sources: Vec<SourceNode>,
    next_id: u32,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            sources: Vec::new(),
            next_id: 0,
        }
    }

    /// Content hash of the value produced by `root`: a stable `u64` identity for
    /// the *computation* (source identity + each op's kernel + params + output
    /// codec + input order), independent of node-id churn across forks. This is
    /// the basis of the unified cache key (see `RegionKey`).
    pub fn content_hash(&self, root: NodeId) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        let mut h = FNV_OFFSET;
        self.fold_content(root, &mut h, 0);
        h
    }

    fn fold_content(&self, id: NodeId, h: &mut u64, depth: u32) {
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let fnv = |h: &mut u64, bytes: &[u8]| {
            for &b in bytes {
                *h ^= b as u64;
                *h = h.wrapping_mul(FNV_PRIME);
            }
        };
        if depth > 1024 {
            return; // guard against pathological graphs (DAG should never recurse this deep)
        }
        if let Some(src) = self.get_source(id) {
            fnv(h, b"src:");
            let sid = super::source::source_identity(&src.source);
            fnv(h, &sid.to_le_bytes());
            return;
        }
        let Some(node) = self.get_node(id) else {
            return;
        };
        let NodeEval::Kernel(k) = &node.eval;
        fnv(h, k.module.as_bytes());
        fnv(h, k.function.as_bytes());
        for p in &node.params {
            fnv(h, &p.to_bytes());
        }
        // Output decoder — distinguishes convert nodes that differ only by target space.
        fnv(h, format!("{:?}", node.op.output_decoder()).as_bytes());
        fnv(h, format!("{:?}", node.output).as_bytes());
        for &inp in &node.inputs {
            fnv(h, b"|");
            self.fold_content(inp, h, depth + 1);
        }
    }

    fn alloc_id(&mut self) -> NodeId {
        let id = NodeId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Advance `next_id` so that subsequent `alloc_id` calls on this fork
    /// produce node IDs that can never collide with other forks derived
    /// from the same original graph.
    ///
    /// Uses a global atomic salt to carve out a unique ID range per fork
    /// (upper 12 bits = fork salt, lower 20 bits = node index).
    /// Call this immediately after `Graph::clone()` in `Image::fork()`.
    pub fn salt_fork(&mut self) {
        use std::sync::atomic::{AtomicU32, Ordering};
        static SALT: AtomicU32 = AtomicU32::new(0);
        let salt = SALT.fetch_add(1, Ordering::Relaxed);
        self.next_id = (salt << 20) | (self.next_id & 0x000F_FFFF);
    }

    pub fn add_node(&mut self, mut node: GraphNode) -> NodeId {
        let id = self.alloc_id();
        node.id = id;
        self.nodes.push(node);
        id
    }

    pub fn add_source(&mut self, source: Arc<super::source::GpuSource>) -> NodeId {
        let id = self.alloc_id();
        self.sources.push(SourceNode { id, source });
        id
    }

    /// Import all nodes and sources from `other` into `self`.
    /// Returns a mapping old NodeId → new NodeId.
    pub fn merge_from(&mut self, other: &Graph) -> HashMap<NodeId, NodeId> {
        let mut remap = HashMap::new();

        for src in &other.sources {
            let new_id = self.alloc_id();
            remap.insert(src.id, new_id);
            self.sources.push(SourceNode {
                id: new_id,
                source: src.source.clone(),
            });
        }

        for old_id in other.topo_order() {
            let Some(node) = other.get_node(old_id) else {
                continue;
            };
            let new_id = self.alloc_id();
            remap.insert(old_id, new_id);
            let new_inputs: Vec<NodeId> = node
                .inputs
                .iter()
                .map(|i| *remap.get(i).unwrap_or(i))
                .collect();
            self.nodes.push(GraphNode {
                id: new_id,
                inputs: new_inputs,
                eval: node.eval.clone(),
                op: node.op.clone(),
                params: node.params.clone(),
                output: node.output.clone(),
            });
        }

        remap
    }

    /// Topological sort (Kahn's algorithm).
    pub fn topo_order(&self) -> Vec<NodeId> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for src in &self.sources {
            in_degree.entry(src.id).or_insert(0);
        }
        for node in &self.nodes {
            in_degree.entry(node.id).or_insert(0);
            for &input in &node.inputs {
                adjacency.entry(input).or_default().push(node.id);
                *in_degree.entry(node.id).or_insert(0) += 1;
            }
        }

        let mut initial_queue: Vec<NodeId> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&id, _)| id)
            .collect();
        initial_queue.sort_by_key(|id| std::cmp::Reverse(id.0));
        let mut stack: Vec<NodeId> = initial_queue;

        let mut order = Vec::new();
        while let Some(id) = stack.pop() {
            order.push(id);
            if let Some(outs) = adjacency.get(&id) {
                // To keep deterministic DFS order, sort outs
                let mut sorted_outs = outs.clone();
                sorted_outs.sort_by_key(|id| std::cmp::Reverse(id.0));
                for &out in &sorted_outs {
                    let d = in_degree.get_mut(&out).unwrap();
                    *d -= 1;
                    if *d == 0 {
                        stack.push(out);
                    }
                }
            }
        }
        order
    }

    pub fn get_node(&self, id: NodeId) -> Option<&GraphNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn get_source(&self, id: NodeId) -> Option<&SourceNode> {
        self.sources.iter().find(|s| s.id == id)
    }

    /// Walk backwards from `id` to find the first source node and return its
    /// full-resolution width and height. Returns `None` for orphaned nodes.
    pub fn source_dimensions(&self, id: NodeId) -> Option<(u32, u32)> {
        if let Some(src) = self.get_source(id) {
            return Some((src.source.width(), src.source.height()));
        }
        if let Some(node) = self.get_node(id) {
            for &inp in &node.inputs {
                if let Some(dims) = self.source_dimensions(inp) {
                    return Some(dims);
                }
            }
        }
        None
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Walk the graph backwards from `roots`, computing the bounding rect each
    /// node needs at `lod`.  Returns the per-node rect map.
    ///
    /// This is the single canonical implementation of the inverse-region walk —
    /// both [`Self::trace_inverse`] and the materialize STEP 1 delegate here.
    ///
    /// ### LOD dimension formula
    /// Source dims are computed as `ceil(full_dim / lod.scale_factor())`.  Using
    /// `lod.scale_factor()` (an `f64` cast of `1u64 << lod.0`) avoids the
    /// integer-overflow risk of `(1i32 << lod.0)` for large LOD values.
    pub fn walk_inverse(
        &self,
        roots: &[(NodeId, super::work_unit::WorkUnit)],
        _lod: super::Lod,
    ) -> HashMap<NodeId, crate::geometry::Rect> {
        let mut node_rects: HashMap<NodeId, crate::geometry::Rect> = HashMap::new();
        let mut pending: Vec<(NodeId, super::work_unit::WorkUnit)> = roots.to_vec();

        while let Some((node_id, unit)) = pending.pop() {
            let unit_lod = unit.lod();
            let (iw, ih) = self
                .source_dimensions(node_id)
                .map(|(w, h)| {
                    let scale = unit_lod.scale_factor();
                    ((w as f64 / scale).ceil() as u32, (h as f64 / scale).ceil() as u32)
                })
                .unwrap_or((0, 0));

            let r = unit.resolve(iw, ih);
            node_rects
                .entry(node_id)
                .and_modify(|existing| *existing = existing.bounding_box(r))
                .or_insert(r);

            if self.get_source(node_id).is_some() {
                continue;
            }

            let Some(node) = self.get_node(node_id) else {
                continue;
            };

            let requests = node.op.input_demands(&unit);
            for (input_idx, req_unit) in requests {
                if let Some(&target) = node.inputs.get(input_idx) {
                    pending.push((target, req_unit));
                }
            }
        }

        node_rects
    }

    /// Returns the bounding source rect required to materialise `root_id` at
    /// the given output `rect` and `lod`.  Delegates to [`Self::walk_inverse`].
    pub fn trace_inverse(
        &self,
        root_id: NodeId,
        rect: crate::geometry::Rect,
        lod: super::Lod,
    ) -> crate::geometry::Rect {
        let node_rects = self.walk_inverse(&[(root_id, super::work_unit::WorkUnit::Region { rect, lod })], lod);
        let mut bounding: Option<crate::geometry::Rect> = None;
        for s in &self.sources {
            if let Some(&r) = node_rects.get(&s.id) {
                bounding = Some(match bounding {
                    Some(b) => b.bounding_box(r),
                    None => r,
                });
            }
        }
        bounding.unwrap_or(rect)
    }

    /// Build an ephemeral sub-graph rooted at `root_id` where every node listed
    /// in `overrides` is replaced by a `SourceNode` backed by the provided
    /// `GpuSource` (typically an `ImageBufferSource` holding a pre-materialised tile).
    ///
    /// The returned `NodeId` is the ID of `root_id` inside the new graph.
    ///
    /// Used by the staged-compilation path to keep each shader pass within the
    /// device's source-buffer budget.
    pub fn subgraph_with_overrides(
        &self,
        root_id: NodeId,
        overrides: &HashMap<NodeId, GpuSource>,
    ) -> (Graph, NodeId) {
        // ── 1. Collect reachable nodes (DFS from root, stop at overrides / sources) ──
        let mut reachable: BTreeSet<NodeId> = BTreeSet::new();
        let mut stack = vec![root_id];

        while let Some(id) = stack.pop() {
            if !reachable.insert(id) {
                continue;
            }
            if overrides.contains_key(&id) || self.get_source(id).is_some() {
                continue; // boundary — will become a SourceNode, don't descend
            }
            if let Some(node) = self.get_node(id) {
                stack.extend(node.inputs.iter().copied());
            }
        }

        // ── 2. Replay in topological order so inputs are always mapped before consumers ──
        let mut new_graph = Graph::new();
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();

        for old_id in self.topo_order() {
            if !reachable.contains(&old_id) {
                continue;
            }

            let new_id = if let Some(override_src) = overrides.get(&old_id) {
                // Cut node → inject as Buffer source.
                new_graph.add_source(Arc::new(override_src.clone()))
            } else if let Some(src_node) = self.get_source(old_id) {
                // Real source node → copy as-is.
                new_graph.add_source(src_node.source.clone())
            } else if let Some(node) = self.get_node(old_id) {
                // Compute node → copy with remapped inputs.
                let new_inputs = node
                    .inputs
                    .iter()
                    .filter_map(|inp| id_map.get(inp).copied())
                    .collect();
                new_graph.add_node(GraphNode {
                    id: NodeId(0), // overwritten by add_node
                    inputs: new_inputs,
                    eval: node.eval.clone(),
                    op: node.op.clone(),
                    params: node.params.clone(),
                    output: node.output.clone(),
                })
            } else {
                continue;
            };

            id_map.insert(old_id, new_id);
        }

        let new_root_id = *id_map
            .get(&root_id)
            .expect("root_id must be reachable from itself");

        (new_graph, new_root_id)
    }
}
