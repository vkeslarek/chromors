//! Standalone graph-construction entry point — decoupled from
//! [`super::datatype::Executable::execute`] / `Image2D::execute`.
//!
//! [`GraphBuilder::build`] constructs a node from an operation plus ALL of
//! its input handles, splicing each non-host input's subgraph into the
//! first input's graph via [`super::graph::Graph::merge_from`]. This is the
//! entry point a future node-editor targets directly; `Executable::execute`
//! is thin sugar over this for the common single-input case.

use std::sync::Arc;

use super::graph::NodeId;
use super::handle::GraphNodeHandle;
use super::op::{GpuOperation, TypedOperation};

pub struct GraphBuilder;

impl GraphBuilder {
    /// Build a node from `op` applied to `inputs`.
    ///
    /// `inputs[0]`'s graph becomes the host graph. Any other input that
    /// shares the same underlying graph (`Arc::ptr_eq`) contributes its
    /// `root_id` directly; otherwise its subgraph is merged into the host
    /// graph and the remapped id is used.
    pub fn build<O>(op: &O, inputs: &[&GraphNodeHandle]) -> GraphNodeHandle
    where
        O: TypedOperation + Clone + 'static,
    {
        let host = inputs[0];
        let self_arc: Arc<dyn GpuOperation> = Arc::new(op.clone());

        let node_id = {
            let mut graph = host.graph.lock().unwrap();
            let mut node_ids: Vec<NodeId> = Vec::with_capacity(inputs.len());
            node_ids.push(host.root_id);
            for other in &inputs[1..] {
                if Arc::ptr_eq(&other.graph, &host.graph) {
                    node_ids.push(other.root_id);
                } else {
                    let other_graph = other.graph.lock().unwrap();
                    let remap = graph.merge_from(&other_graph);
                    node_ids.push(remap[&other.root_id]);
                }
            }
            op.emit(&node_ids, &mut graph, self_arc)
        };

        GraphNodeHandle {
            graph: host.graph.clone(),
            root_id: node_id,
            ctx: host.ctx.clone(),
        }
    }
}
