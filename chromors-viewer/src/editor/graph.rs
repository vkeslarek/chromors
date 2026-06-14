use slotmap::{new_key_type, SlotMap};
use vello::kurbo::{Point, Rect, Size};
use crate::editor::params::ParamValue;
use crate::editor::registry::{registry, NodeKindId};
use crate::editor::types::DataType;

new_key_type! {
    pub struct NodeKey;
    pub struct EdgeKey;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Side { In, Out }

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PortAddr {
    pub node: NodeKey,
    pub side: Side,
    pub index: u16,
}

pub struct NodeLayout {
    pub size: Size,
    pub title_rect: Rect,
    pub input_sockets: Vec<Point>,
    pub output_sockets: Vec<Point>,
    pub widget_rows: Vec<Rect>,
}

pub struct EditorNode {
    pub kind: NodeKindId,
    pub pos: Point,
    pub params: Vec<ParamValue>,
    pub title: String,
    pub collapsed: bool,
    pub layout_cache: Option<NodeLayout>,
}

pub struct Edge {
    pub from: PortAddr, // Side::Out
    pub to: PortAddr,   // Side::In
}

pub struct NodeGraph {
    pub nodes: SlotMap<NodeKey, EditorNode>,
    pub edges: SlotMap<EdgeKey, Edge>,
    pub revision: u64,
}

#[derive(Debug)]
pub enum ConnectError {
    TypeMismatch { out: DataType, in_: DataType },
    WouldCycle,
    NotAnOutput,
    NotAnInput,
    SameNode,
}

impl NodeGraph {
    pub fn new() -> Self {
        Self {
            nodes: SlotMap::with_key(),
            edges: SlotMap::with_key(),
            revision: 0,
        }
    }

    pub fn add_node(&mut self, kind: NodeKindId, pos: Point) -> NodeKey {
        let desc = registry().get(kind);
        let params = desc.params.iter().map(|p| p.default_value()).collect();
        let node = EditorNode {
            kind,
            pos,
            params,
            title: desc.title.to_string(),
            collapsed: false,
            layout_cache: None,
        };
        let key = self.nodes.insert(node);
        self.revision += 1;
        key
    }

    pub fn remove_node(&mut self, key: NodeKey) {
        if self.nodes.remove(key).is_some() {
            let mut edges_to_remove = Vec::new();
            for (edge_key, edge) in &self.edges {
                if edge.from.node == key || edge.to.node == key {
                    edges_to_remove.push(edge_key);
                }
            }
            for e in edges_to_remove {
                self.edges.remove(e);
            }
            self.revision += 1;
        }
    }

    pub fn set_param(&mut self, node: NodeKey, index: usize, value: ParamValue) {
        if let Some(n) = self.nodes.get_mut(node) {
            if index < n.params.len() {
                n.params[index] = value;
                n.layout_cache = None;
                self.revision += 1;
            }
        }
    }

    pub fn move_node(&mut self, node: NodeKey, new_pos: Point) {
        if let Some(n) = self.nodes.get_mut(node) {
            n.pos = new_pos;
            n.layout_cache = None;
            // Does not bump revision, only moves nodes
        }
    }

    pub fn incoming(&self, addr: PortAddr) -> Option<(EdgeKey, PortAddr)> {
        for (k, e) in &self.edges {
            if e.to == addr {
                return Some((k, e.from));
            }
        }
        None
    }

    pub fn outgoing(&self, addr: PortAddr) -> impl Iterator<Item = (EdgeKey, PortAddr)> + '_ {
        self.edges.iter().filter_map(move |(k, e)| {
            if e.from == addr {
                Some((k, e.to))
            } else {
                None
            }
        })
    }

    pub fn connect(&mut self, from: PortAddr, to: PortAddr) -> Result<EdgeKey, ConnectError> {
        if from.side != Side::Out { return Err(ConnectError::NotAnOutput); }
        if to.side != Side::In { return Err(ConnectError::NotAnInput); }
        if from.node == to.node { return Err(ConnectError::SameNode); }

        let desc_from = registry().get(self.nodes[from.node].kind);
        let desc_to = registry().get(self.nodes[to.node].kind);

        let out_ty = desc_from.outputs[from.index as usize].ty;
        let in_ty = desc_to.inputs[to.index as usize].ty;

        if !in_ty.accepts(out_ty) {
            return Err(ConnectError::TypeMismatch { out: out_ty, in_: in_ty });
        }

        // Cycle check (DFS)
        let mut visited = std::collections::HashSet::new();
        let mut stack = vec![to.node];
        while let Some(curr) = stack.pop() {
            if curr == from.node {
                return Err(ConnectError::WouldCycle);
            }
            if visited.insert(curr) {
                let desc_curr = registry().get(self.nodes[curr].kind);
                for i in 0..desc_curr.outputs.len() {
                    let curr_out = PortAddr { node: curr, side: Side::Out, index: i as u16 };
                    for (_, child) in self.outgoing(curr_out) {
                        stack.push(child.node);
                    }
                }
            }
        }

        // Evict existing
        if let Some((existing_edge, _)) = self.incoming(to) {
            self.edges.remove(existing_edge);
        }

        let edge = Edge { from, to };
        let key = self.edges.insert(edge);
        self.revision += 1;
        Ok(key)
    }

    pub fn disconnect(&mut self, edge: EdgeKey) {
        if self.edges.remove(edge).is_some() {
            self.revision += 1;
        }
    }
}
