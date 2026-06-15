use crate::editor::graph::{NodeGraph, NodeKey, PortAddr, Side};
use crate::editor::registry::{BuildError, registry};
use crate::editor::types::PortValue;
use poc::backend::gpu::GpuContext;
use std::collections::HashMap;
use std::sync::Arc;

pub struct EvalCache {
    pub values: HashMap<(NodeKey, u16), PortValue>,
    pub errors: HashMap<NodeKey, BuildError>,
    pub built_revision: u64,
}

impl EvalCache {
    pub fn new(built_revision: u64) -> Self {
        Self {
            values: HashMap::new(),
            errors: HashMap::new(),
            built_revision,
        }
    }
}

pub fn topo_order(g: &NodeGraph) -> Result<Vec<NodeKey>, NodeKey> {
    let mut in_degree: HashMap<NodeKey, usize> = HashMap::new();
    let mut adjacency: HashMap<NodeKey, Vec<NodeKey>> = HashMap::new();

    for key in g.nodes.keys() {
        in_degree.insert(key, 0);
        adjacency.insert(key, Vec::new());
    }

    for edge in g.edges.values() {
        let from = edge.from.node;
        let to = edge.to.node;
        adjacency.get_mut(&from).unwrap().push(to);
        *in_degree.get_mut(&to).unwrap() += 1;
    }

    let mut queue = Vec::new();
    for (key, &deg) in &in_degree {
        if deg == 0 {
            queue.push(*key);
        }
    }

    let mut order = Vec::new();
    while let Some(curr) = queue.pop() {
        order.push(curr);
        for &neighbor in &adjacency[&curr] {
            let deg = in_degree.get_mut(&neighbor).unwrap();
            *deg -= 1;
            if *deg == 0 {
                queue.push(neighbor);
            }
        }
    }

    if order.len() == g.nodes.len() {
        Ok(order)
    } else {
        // Find a node that wasn't included (part of a cycle)
        for key in g.nodes.keys() {
            if in_degree[&key] > 0 {
                return Err(key);
            }
        }
        unreachable!()
    }
}

pub fn evaluate(g: &NodeGraph, ctx: &Arc<GpuContext>) -> EvalCache {
    let mut cache = EvalCache::new(g.revision);
    let order = match topo_order(g) {
        Ok(o) => o,
        Err(_cycle) => return cache,
    };

    for key in order {
        let node = &g.nodes[key];
        let desc = registry().get(node.kind);

        let mut inputs: Vec<Option<PortValue>> = Vec::with_capacity(desc.inputs.len());
        let mut upstream_failed = false;

        for in_idx in 0..desc.inputs.len() as u16 {
            let addr = PortAddr {
                node: key,
                side: Side::In,
                index: in_idx,
            };
            match g.incoming(addr) {
                Some((_edge, from)) => match cache.values.get(&(from.node, from.index)) {
                    Some(v) => inputs.push(Some(v.clone())),
                    None => {
                        upstream_failed = true;
                        inputs.push(None);
                    }
                },
                None => inputs.push(None),
            }
        }

        if upstream_failed {
            cache
                .errors
                .insert(key, BuildError("upstream error".into()));
            continue;
        }

        match (desc.build)(&inputs, &node.params, ctx) {
            Ok(outs) => {
                for (i, v) in outs.into_iter().enumerate() {
                    cache.values.insert((key, i as u16), v);
                }
            }
            Err(e) => {
                cache.errors.insert(key, e);
            }
        }
    }

    cache
}
