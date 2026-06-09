//! Graph traversal cut finder (DFS) — determines which nodes to pre-materialise
//! so that a single shader pass stays within the device's storage-buffer budget.
//!
//! Walks the graph depth-first from the root node, counting storage-buffer
//! bindings. When adding a node would exceed the device budget the node is
//! marked as a staging cut — it is pre-materialised and its output re-injected
//! as an `ImageBufferSource` in the parent pass.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::Lod;
use super::graph::{Graph, NodeId};
use super::source::AnyGpuSource;
use crate::geometry::Rect;

/// The set of nodes that must be pre-materialised before the main pass.
pub struct StagingCuts {
    /// `(node_id, rect)` pairs — each node will be materialised to cover `rect`
    /// and its output re-injected as an `ImageBufferSource` in the parent pass.
    pub staging: Vec<(NodeId, Rect)>,
}

impl StagingCuts {
    pub fn empty() -> Self {
        Self { staging: vec![] }
    }
}

/// Compute which nodes need to be staged (pre-materialised) so that the shader
/// pass rooted at `root_id` fits within `device_limit` storage buffers per stage.
pub struct CutFinder<'a> {
    graph: &'a Graph,
    root_id: NodeId,
    root_rect: Rect,
    lod: Lod,
    device_limit: usize,
}

impl<'a> CutFinder<'a> {
    pub fn new(
        graph: &'a Graph,
        root_id: NodeId,
        root_rect: Rect,
        lod: Lod,
        device_limit: usize,
    ) -> Self {
        Self {
            graph,
            root_id,
            root_rect,
            lod,
            device_limit,
        }
    }

    pub fn execute(self) -> StagingCuts {
        // Trivial cases: no compute nodes or degenerate limit.
        if self.device_limit == 0 || self.graph.node_count() == 0 {
            return StagingCuts::empty();
        }

        // Group 0: [source_0 .. source_n, params]   budget = device_limit - 1
        // Group 1: [temp_0 .. temp_m, target_0 ..]  budget = device_limit - 1
        let source_budget = self.device_limit.saturating_sub(1);
        let temp_budget = self.device_limit.saturating_sub(2);

        let trans = self.compute_transitive_sources();
        let (iw, ih) = self.source_dims();

        let mut included_sources: BTreeSet<NodeId> = BTreeSet::new();
        let mut temp_count: usize = 0;
        let mut node_rects: HashMap<NodeId, Rect> = HashMap::new();
        let mut expanded: HashSet<NodeId> = HashSet::new();
        let mut staging: Vec<(NodeId, Rect)> = Vec::new();
        let mut staged: HashSet<NodeId> = HashSet::new();

        let mut stack: Vec<(NodeId, Rect)> = Vec::new();
        stack.push((self.root_id, self.root_rect));

        while let Some((nid, rect)) = stack.pop() {
            node_rects
                .entry(nid)
                .and_modify(|r| *r = r.bounding_box(rect))
                .or_insert(rect);

            if expanded.contains(&nid) {
                continue;
            }

            if staged.contains(&nid) {
                included_sources.insert(nid);
                expanded.insert(nid);
                continue;
            }

            if self.graph.get_source(nid).is_some() {
                included_sources.insert(nid);
                expanded.insert(nid);
                continue;
            }

            let Some(node) = self.graph.get_node(nid) else {
                expanded.insert(nid);
                continue;
            };

            if nid != self.root_id {
                let new_source_count = trans
                    .get(&nid)
                    .map(|s| {
                        s.iter()
                            .filter(|id| !included_sources.contains(*id) && !staged.contains(*id))
                            .count()
                    })
                    .unwrap_or(0);

                let would_sources = included_sources.len() + new_source_count;
                let would_temps = temp_count + 1;

                if would_sources > source_budget || would_temps > temp_budget {
                    let cut_rect = node_rects[&nid];
                    staging.push((nid, cut_rect));
                    staged.insert(nid);
                    included_sources.insert(nid);
                    expanded.insert(nid);
                    continue;
                }

                temp_count += 1;
            }

            expanded.insert(nid);

            let requests = node.op.inverse_map(rect, iw, ih, self.lod);
            for (idx, req_rect) in requests {
                let Some(&child_id) = node.inputs.get(idx) else {
                    continue;
                };
                if self.graph.get_source(child_id).is_some() && !expanded.contains(&child_id) {
                    included_sources.insert(child_id);
                }
                stack.push((child_id, req_rect));
            }
        }

        StagingCuts { staging }
    }

    fn compute_transitive_sources(&self) -> HashMap<NodeId, BTreeSet<NodeId>> {
        let order = self.graph.topo_order();
        let mut trans: HashMap<NodeId, BTreeSet<NodeId>> = HashMap::new();

        for &id in &order {
            if self.graph.get_source(id).is_some() {
                trans.insert(id, BTreeSet::from([id]));
            } else if let Some(node) = self.graph.get_node(id) {
                let mut sources = BTreeSet::new();
                for &inp in &node.inputs {
                    if let Some(s) = trans.get(&inp) {
                        sources.extend(s.iter().copied());
                    }
                }
                trans.insert(id, sources);
            }
        }
        trans
    }

    fn source_dims(&self) -> (u32, u32) {
        self.graph
            .sources
            .first()
            .map(|s| {
                let scale = self.lod.scale_factor();
                let w = (s.source.width() as f64 / scale).ceil() as u32;
                let h = (s.source.height() as f64 / scale).ceil() as u32;
                (w, h)
            })
            .unwrap_or((0, 0))
    }
}
