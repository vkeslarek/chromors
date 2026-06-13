//! GPU pass splitting and parallel dispatch.
//!
//! Pipeline: **DAG → BFS analysis → cuts → rebuild with wrappers → shaders → dispatches**
//!
//! When a fused pass would exceed the device's `max_storage_buffers_per_shader_stage`,
//! this module splits the work:
//!
//! 1. **Analyze** the immutable DAG via BFS, counting sources and ops per level.
//! 2. **Find cuts** — the shallowest BFS depth where splitting brings the
//!    remaining pass under the binding budget.
//! 3. **Pre-materialize** each cut subgraph independently (parallel via rayon).
//! 4. **Rebuild** the DAG with lightweight wrapper nodes (`RebuiltOp`) that
//!    swap the cut children for `StagingSource` leaves. The wrapper
//!    delegates `lower`/`demand`/`output_kind`/`dyn_hash` to the original op —
//!    only `inputs()` is overridden. `GraphWalk` traverses the rebuilt tree
//!    naturally, no modifications needed.
//! 5. **Materialize** the reduced DAG through the standard `node::materialize`.
//!
//! Everything here is **GPU-specific** — the agnostic core is untouched.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::Hasher;
use std::sync::Arc;

use crate::backend::Backend;
use crate::buffer::Buffer;
use crate::error::Error;
use crate::io::Source;
use crate::kind::AnyKind;
use crate::node::{Node, NodeId};
use crate::operation::{AnyInput, AnyOperation};
use crate::work_unit::{Region, WorkUnit, WorkUnitFor};

use super::buffer::GpuBuffer;
use super::context::GpuContext;
use super::view::RegionParams;
use super::{GpuBackend, GpuBuilder, GpuView};

// ── Binding budget helpers ───────────────────────────────────────────────────

/// Total storage buffer bindings a fused pass needs.
/// Layout: `target(1) + params(1) + work_buffers(W) + sources(S)`.
pub fn binding_count(n_steps: usize, n_sources: usize, needs_scratch: bool) -> usize {
    let work = if needs_scratch { n_steps } else { n_steps.saturating_sub(1) };
    2 + work + n_sources
}

/// True if the pass would exceed the device binding limit.
pub fn exceeds_binding_limit(
    n_steps: usize,
    n_sources: usize,
    needs_scratch: bool,
    max_storage_buffers: u32,
) -> bool {
    binding_count(n_steps, n_sources, needs_scratch) > max_storage_buffers as usize
}

// ── DAG wrapper nodes (GPU-specific) ──────────────────────────────────────────

/// A lightweight `AnyInput` for rebuilt children.
struct StubInput {
    node: Arc<Node<GpuBackend>>,
    kind: Arc<dyn AnyKind>,
}

impl AnyInput<GpuBackend> for StubInput {
    fn src(&self) -> &Arc<Node<GpuBackend>> { &self.node }
    fn spec(&self) -> &dyn AnyKind { self.kind.as_ref() }
}

/// Wraps an existing node, overrides `inputs()` with rebuilt children.
/// All other behavior delegates to the original node.
struct RebuiltOp {
    original: Arc<Node<GpuBackend>>,
    inputs: Vec<StubInput>,
}

impl AnyOperation<GpuBackend> for RebuiltOp {
    fn inputs(&self) -> Vec<&dyn AnyInput<GpuBackend>> {
        self.inputs.iter().map(|i| i as &dyn AnyInput<GpuBackend>).collect()
    }
    fn demand_erased(&self, out: &WorkUnit) -> Vec<Option<WorkUnit>> {
        self.original.demand_erased(out)
    }
    fn output_kind(&self) -> Arc<dyn AnyKind> {
        self.original.output_kind()
    }
    fn lower(&self, cx: &mut GpuBuilder) {
        self.original.lower(cx)
    }
    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write_usize(Arc::as_ptr(&self.original) as usize);
    }
}

// ── StagingSource — staging cut result as a DAG leaf ─────────────────────

/// A GPU source backed by a pre-materialized `GpuBuffer`. Created by the
/// CutFinder when a subgraph is pre-dispatched and its result injected as a
/// new source leaf. The buffer contains pixel data in the format described
/// by its `ImageKind`.
pub struct StagingSource {
    spec: Arc<crate::data::image::ImageKind>,
    buffer: Arc<GpuBuffer>,
}

impl Source<GpuBackend> for StagingSource {
    type Kind = crate::data::image::ImageKind;

    fn spec(&self) -> Arc<crate::data::image::ImageKind> {
        self.spec.clone()
    }

    fn fetch(
        &self,
        _ctx: &GpuContext,
        _wu: &Region,
    ) -> Result<Buffer<GpuBackend>, Error> {
        Ok(Buffer {
            payload: self.buffer.clone(),
            spec: self.spec.clone(),
        })
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let geom = RegionParams::tight(self.spec.width, self.spec.height);
        cx.input(
            self.spec.input(),
            geom.into_block("region_in_{slot}"),
            self.buffer.clone(),
        );
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_usize(Arc::as_ptr(&self.buffer) as usize);
    }
}

// ── BFS DAG analysis ─────────────────────────────────────────────────────────

/// BFS analysis result: nodes grouped by level, source/op counts.
struct DagAnalysis {
    /// Nodes at each BFS level (level 0 = root, deepest = sources).
    levels: Vec<Vec<Arc<Node<GpuBackend>>>>,
    n_sources: usize,
    n_ops: usize,
}

/// BFS from the root, grouping nodes by depth level. Deduplicates diamonds.
fn analyze_dag(root: &Arc<Node<GpuBackend>>) -> DagAnalysis {
    let mut visited = HashSet::new();
    let mut levels: Vec<Vec<Arc<Node<GpuBackend>>>> = Vec::new();
    let mut queue = VecDeque::new();
    let mut n_sources = 0usize;
    let mut n_ops = 0usize;

    visited.insert(NodeId::of(root));
    queue.push_back((root.clone(), 0usize));

    while let Some((node, depth)) = queue.pop_front() {
        while levels.len() <= depth {
            levels.push(Vec::new());
        }
        levels[depth].push(node.clone());

        if node.is_source() {
            n_sources += 1;
        } else {
            n_ops += 1;
        }

        for input in node.inputs() {
            let child = input.src().clone();
            if visited.insert(NodeId::of(&child)) {
                queue.push_back((child, depth + 1));
            }
        }
    }

    DagAnalysis { levels, n_sources, n_ops }
}

/// Find nodes whose subgraphs should be pre-materialized (staging cuts).
///
/// BFS from root, accumulating binding cost level by level. At each candidate
/// cut depth, compute what the remaining pass would look like if everything
/// at that depth and below were replaced by sources. Return the shallowest
/// cut that brings the remaining pass under budget. BFS maximizes the width
/// of independent sub-trees at the cut level → maximizes rayon parallelism.
fn find_cuts(
    root: &Arc<Node<GpuBackend>>,
    max_bindings: u32,
) -> Vec<Arc<Node<GpuBackend>>> {
    let analysis = analyze_dag(root);
    let full_bindings = binding_count(analysis.n_ops, analysis.n_sources, true);

    if full_bindings <= max_bindings as usize {
        return vec![];
    }

    let is_op = |n: &Arc<Node<GpuBackend>>| n.is_op();
    let is_source = |n: &Arc<Node<GpuBackend>>| n.is_source();

    // Try each BFS depth as a candidate cut level (shallow first).
    for cut_depth in 1..analysis.levels.len() {
        let ops_above: usize = analysis.levels[..cut_depth]
            .iter()
            .flatten()
            .filter(|n| is_op(n))
            .count();

        let sources_above: usize = analysis.levels[..cut_depth]
            .iter()
            .flatten()
            .filter(|n| is_source(n))
            .count();

        // Op nodes at the cut depth become new sources after pre-materialization.
        let cut_ops: Vec<_> = analysis.levels[cut_depth]
            .iter()
            .filter(|n| is_op(n))
            .cloned()
            .collect();

        let cut_sources = analysis.levels[cut_depth]
            .iter()
            .filter(|n| is_source(n))
            .count();

        let remaining_sources = sources_above + cut_sources + cut_ops.len();
        let remaining_bindings = binding_count(ops_above, remaining_sources, true);

        if remaining_bindings <= max_bindings as usize {
            return cut_ops;
        }
    }

    // Fallback: cut at the deepest op level.
    analysis
        .levels
        .iter()
        .rev()
        .flat_map(|level| level.iter())
        .filter(|n| is_op(n))
        .take(1)
        .cloned()
        .collect()
}

// ── DAG rebuild ──────────────────────────────────────────────────────────────

/// Recursively rebuild the DAG, replacing cut nodes with `StagingSource`
/// leaves. Unchanged subtrees share the original `Arc<Node>`. Only paths
/// that contain a replacement get new `RebuiltOp` wrapper nodes.
fn rebuild_dag(
    node: &Arc<Node<GpuBackend>>,
    replacements: &HashMap<NodeId, Arc<Node<GpuBackend>>>,
) -> Arc<Node<GpuBackend>> {
    let node_id = NodeId::of(node);

    if let Some(replacement) = replacements.get(&node_id) {
        return replacement.clone();
    }

    if node.is_source() {
        return node.clone();
    }

    // Node is an op — check if any child subtree has a replacement.
    let original_inputs = node.inputs();
    let mut any_changed = false;

    let rebuilt_children: Vec<Arc<Node<GpuBackend>>> = original_inputs
        .iter()
        .map(|input| {
            let child = input.src();
            let rebuilt = rebuild_dag(child, replacements);
            if !Arc::ptr_eq(child, &rebuilt) {
                any_changed = true;
            }
            rebuilt
        })
        .collect();

    if !any_changed {
        return node.clone(); // No replacements in this subtree.
    }

    // Build StubInputs pointing to the rebuilt children.
    let stubs: Vec<StubInput> = rebuilt_children
        .into_iter()
        .zip(original_inputs.iter())
        .map(|(child, orig_input)| StubInput {
            kind: orig_input.spec().as_any()
                .downcast_ref::<crate::data::image::ImageKind>()
                .map(|k| Arc::new(k.clone()) as Arc<dyn AnyKind>)
                .unwrap_or_else(|| child.output_kind()),
            node: child,
        })
        .collect();

    Arc::new(Node::from_op(Arc::new(RebuiltOp {
        original: node.clone(),
        inputs: stubs,
    })))
}

// ── GPU materialization with cuts ────────────────────────────────────────────

/// GPU-specific materialization entry point.
///
/// Pipeline: DAG → BFS analysis → find cuts → parallel pre-materialize →
/// rebuild DAG with wrappers → standard materialize on reduced DAG.
///
/// Called by `GpuBackend::materialize` (the `Backend` trait override).
pub(crate) fn gpu_materialize(
    ctx: &Arc<GpuContext>,
    root: &Arc<Node<GpuBackend>>,
    wu: &WorkUnit,
) -> Result<Buffer<GpuBackend>, Error> {
    let max_bindings = ctx.max_storage_buffers;
    let cuts = find_cuts(root, max_bindings);

    if cuts.is_empty() {
        return crate::node::materialize::<GpuBackend>(ctx, root, wu);
    }

    tracing::info!(
        "pass: {} staging cuts at max {} bindings",
        cuts.len(),
        max_bindings,
    );

    // Pre-materialize each cut subgraph in parallel via rayon.
    // Each cut is independent (BFS guarantees same-level nodes don't
    // depend on each other) → maximum parallelism.
    use rayon::prelude::*;

    let cut_results: Vec<Result<(NodeId, Arc<Node<GpuBackend>>), Error>> = cuts
        .par_iter()
        .map(|cut_node| {
            let cut_id = NodeId::of(cut_node);

            // Recursive: if the subgraph still exceeds limits, it'll cut again.
            let buf = gpu_materialize(ctx, cut_node, wu)?;

            // Wrap the result as a StagingSource leaf.
            let img_kind = buf
                .spec
                .as_any()
                .downcast_ref::<crate::data::image::ImageKind>()
                .ok_or_else(|| {
                    Error::Backend(format!(
                        "staging cut produces {:?}, only ImageKind supported",
                        buf.spec
                    ))
                })?;

            let source = Arc::new(StagingSource {
                spec: Arc::new(img_kind.clone()),
                buffer: buf.payload,
            });

            let source_node: Arc<Node<GpuBackend>> = Arc::new(Node::from_source(
                source as Arc<dyn crate::io::AnySource<GpuBackend>>,
            ));

            Ok((cut_id, source_node))
        })
        .collect();

    // Collect and build the replacement map.
    let mut replacements: HashMap<NodeId, Arc<Node<GpuBackend>>> = HashMap::new();
    for result in cut_results {
        let (cut_id, source_node) = result?;
        replacements.insert(cut_id, source_node);
    }

    // Rebuild the DAG with wrapper nodes.
    let new_root = rebuild_dag(root, &replacements);

    // Materialize the reduced DAG (should now fit in budget).
    crate::node::materialize::<GpuBackend>(ctx, &new_root, wu)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binding_count_scratch_output() {
        assert_eq!(binding_count(3, 2, true), 7);
    }

    #[test]
    fn test_binding_count_direct_output() {
        assert_eq!(binding_count(3, 2, false), 6);
    }

    #[test]
    fn test_binding_count_zero_steps() {
        assert_eq!(binding_count(0, 1, true), 3);
        assert_eq!(binding_count(0, 1, false), 3);
    }

    #[test]
    fn test_exceeds_binding_limit_check() {
        assert!(exceeds_binding_limit(10, 5, true, 16));
        assert!(!exceeds_binding_limit(10, 5, true, 17));
        assert!(!exceeds_binding_limit(10, 5, true, 18));
    }
}
