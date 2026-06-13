use std::sync::Arc;
use crate::kind::{Kind, AnyKind};
use crate::operation::{AnyOperation, Operation, Input, AnyInput};
use crate::io::{AnySource, Source, Target};
use crate::backend::{Backend, Builder};
use crate::buffer::Buffer;
use crate::error::Error;

/// Pointer-identity key for a DAG node, shared by `GraphWalk` and every
/// backend builder's node-keyed maps. An immutable `Arc` DAG never moves its
/// nodes, so the address is a stable identity for the lifetime of the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

impl NodeId {
    pub fn of<B: Backend>(node: &Arc<Node<B>>) -> Self {
        Self(Arc::as_ptr(node) as *const () as usize)
    }
}

/// An immutable, structurally shared directed acyclic graph node.
/// Pointer chasing is used to navigate the tree. There is no central Graph struct.
pub enum Node<B: Backend> {
    Op(Arc<dyn AnyOperation<B>>),
    Source(Arc<dyn AnySource<B>>),
}

impl<B: Backend> Node<B> {
    pub fn output_kind(&self) -> Arc<dyn AnyKind> {
        match self {
            Node::Source(src) => src.output_kind(),
            Node::Op(op) => op.output_kind(),
        }
    }

    pub fn lower(&self, builder: &mut B::Builder) {
        match self {
            Node::Source(src) => src.lower(builder),
            Node::Op(op) => op.lower(builder),
        }
    }

    pub fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        match self {
            Node::Source(_) => vec![],
            Node::Op(op) => op.inputs(),
        }
    }

    pub fn demand_erased(&self, wu: &crate::work_unit::WorkUnit) -> Vec<Option<crate::work_unit::WorkUnit>> {
        match self {
            Node::Source(_) => vec![],
            Node::Op(op) => op.demand_erased(wu),
        }
    }
}

/// The generic user-facing handle on a lazy pipeline tip. A user interacts with
/// type aliases over this (`Image2D = Data<ImageKind, GpuBackend>`, `FeatureSet = …`) and
/// feels they're holding their data — nothing computes until a `Target`.
pub struct Data<K: Kind, B: Backend> {
    pub root: Arc<Node<B>>,
    pub ctx: Arc<B::Ctx>,
    pub spec: Arc<K>,
}

impl<K: Kind, B: Backend> Clone for Data<K, B> {
    fn clone(&self) -> Self {
        Self {
            root: Arc::clone(&self.root),
            ctx: Arc::clone(&self.ctx),
            spec: Arc::clone(&self.spec),
        }
    }
}

impl<K: Kind, B: Backend> Data<K, B> {
    /// Evaluate the requested `WorkUnit` into a backend-resident `Buffer<B>`.
    ///
    /// **Internal on purpose.** Exposing this publicly invites callers (and
    /// AIs mid-development) to download the result and quietly break the
    /// "data stays on the backend" invariant. The only public exits run a
    /// `Target` — including the resident viewport exit, which is itself a
    /// `Target<Out = Buffer<B>>` that never downloads.
    pub(crate) fn materialize(&self, wu: K::WorkUnit) -> Result<Buffer<B>, Error> {
        use crate::work_unit::WorkUnitFor;
        materialize::<B>(&self.ctx, &self.root, &wu.erase())
    }

    /// Read-only access to this handle's backend context (e.g. a Source
    /// adapter that materializes an upstream pipeline of the *same* backend).
    pub fn ctx(&self) -> &Arc<B::Ctx> {
        &self.ctx
    }

    /// Convert this pipeline tip into a typed `Input` edge for a downstream op.
    pub fn as_input(&self) -> Input<K, B> {
        Input {
            src: Arc::clone(&self.root),
            spec: Arc::clone(&self.spec),
        }
    }

    /// Extend the DAG: wrap the op (which already holds its inputs) into a new
    /// immutable root. The old DAG is shared, never mutated.
    pub fn push<Op: Operation<B, Output = K2>, K2: Kind>(&self, op: Op) -> Data<K2, B> {
        let spec = Arc::new(op.output_spec());
        Data {
            root: Arc::new(Node::Op(Arc::new(op))),
            ctx: Arc::clone(&self.ctx),
            spec,
        }
    }

    /// Evaluates the subgraph up to this node and extracts the result via `target`.
    pub fn pull<T: Target<K, B>>(&self, target: &T, wu: K::WorkUnit) -> Result<T::Out, Error> {
        let buf = self.materialize(wu.clone())?;
        target.extract(&buf, &wu, &self.ctx)
    }

    /// Zero-cost typed cast, derived from the Kind's own declaration
    /// (`ReinterpretAs::reinterpret_spec`).
    pub fn reinterpret<T>(&self) -> Data<T, B>
    where
        K: crate::kind::ReinterpretAs<T>,
        T: Kind<WorkUnit = K::WorkUnit>,
        crate::operation::Reinterpret<K, T, B>: Operation<B, Output = T>,
    {
        let spec = self.spec.reinterpret_spec();
        self.push(crate::operation::Reinterpret { input: self.as_input(), spec })
    }

    /// Zero-cost cast with an explicit target spec — the caller asserts byte
    /// compatibility (used for the rewrap direction, where the target spec
    /// carries data the source Kind doesn't have, e.g. frame timing).
    pub fn reinterpret_with<T>(&self, spec: T) -> Data<T, B>
    where
        T: Kind<WorkUnit = K::WorkUnit>,
        crate::operation::Reinterpret<K, T, B>: Operation<B, Output = T>,
    {
        self.push(crate::operation::Reinterpret { input: self.as_input(), spec })
    }
}

impl<K: Kind, B: Backend> Data<K, B> {
    /// Build a fresh pipeline tip from a graph leaf.
    pub fn from_source<S: Source<B, Kind = K>>(source: Arc<S>, ctx: Arc<B::Ctx>) -> Self {
        let spec = source.spec();
        Self { root: Arc::new(Node::Source(source)), ctx, spec }
    }
}

use std::collections::{HashMap, HashSet};

/// Agnostic stateful walker for DAG traversals.
/// It owns the transient maps and sets needed to execute a pass over the graph.
pub struct GraphWalk<'a, B: Backend> {
    pub root: &'a Arc<Node<B>>,
    pub demands: HashMap<NodeId, crate::work_unit::WorkUnit>,
    pub lowered: HashSet<NodeId>,
}

impl<'a, B: Backend> GraphWalk<'a, B> {
    pub fn new(root: &'a Arc<Node<B>>) -> Self {
        Self {
            root,
            demands: HashMap::new(),
            lowered: HashSet::new(),
        }
    }

    /// Inverse map. From the root demand, propagate each consumer's `demand()`
    /// upstream; a node reached by several paths accumulates the `union` (bounding
    /// box) in its entry; `None` prunes a whole input subgraph. Children are only
    /// re-pushed when a revisit actually grows the node's accumulated demand —
    /// otherwise dense diamonds would re-walk their whole upstream subgraph once
    /// per path.
    pub fn demand(&mut self, root_wu: &crate::work_unit::WorkUnit) {
        use std::collections::hash_map::Entry;

        let mut stack = vec![(self.root.clone(), root_wu.clone())];

        while let Some((node, wu)) = stack.pop() {
            let k = NodeId::of(&node);
            let grown = match self.demands.entry(k) {
                Entry::Occupied(mut e) => {
                    let union = e.get().union(&wu);
                    let grown = union != *e.get();
                    e.insert(union);
                    grown
                }
                Entry::Vacant(e) => {
                    e.insert(wu.clone());
                    true
                }
            };
            if !grown {
                continue;
            }

            let demands = node.demand_erased(&wu);
            debug_assert_eq!(
                demands.len(),
                node.inputs().len(),
                "Operation::demand must return one entry per input"
            );
            for (input, child) in node.inputs().iter().zip(demands) {
                if let Some(child_wu) = child {
                    stack.push((input.src().clone(), child_wu));
                } // None => pruned this region
            }
        }
    }

    /// Post-order, **deduplicated** (diamonds lowered once). Skips inputs with no
    /// demand entry (pruned). The `enter_and_lower` closure bridges to the concrete
    /// backend builder (announcing the resolved WorkUnit and calling `.lower()`).
    pub fn lower<F>(&mut self, mut enter_and_lower: F)
    where
        F: FnMut(&Arc<Node<B>>, &crate::work_unit::WorkUnit),
    {
        self.lower_walk_impl(self.root, &mut enter_and_lower);
    }

    fn lower_walk_impl<F>(&mut self, node: &Arc<Node<B>>, enter_and_lower: &mut F)
    where
        F: FnMut(&Arc<Node<B>>, &crate::work_unit::WorkUnit),
    {
        let k = NodeId::of(node);
        let Some(wu) = self.demands.get(&k).cloned() else { return }; // pruned
        if !self.lowered.insert(k) {
            return; // diamond: lower once
        }

        for input in node.inputs() {
            self.lower_walk_impl(input.src(), enter_and_lower);
        }
        enter_and_lower(node, &wu);
    }
}

/// The whole-graph evaluation pass: demand walk, then lower walk into a
/// fresh `B::Builder`, finished into the root's `Buffer<B>`.
pub(crate) fn materialize<B: Backend>(
    ctx: &Arc<B::Ctx>,
    root: &Arc<Node<B>>,
    wu: &crate::work_unit::WorkUnit,
) -> Result<Buffer<B>, Error> {
    let mut walk = GraphWalk::new(root);
    walk.demand(wu);

    let mut builder = B::Builder::new(Arc::clone(ctx));
    walk.lower(|node, node_wu| {
        let inputs: Vec<NodeId> = node.inputs().iter().map(|i| NodeId::of(i.src())).collect();
        builder.enter(NodeId::of(node), &inputs, node_wu);
        node.lower(&mut builder);
    });
    builder.finish(NodeId::of(root), root.output_kind(), wu)
}
