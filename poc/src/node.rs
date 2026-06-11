use std::sync::Arc;
use std::marker::PhantomData;
use crate::kind::{Kind, AnyKind};
use crate::operation::{AnyOperation, Operation, Input, AnyInput};
use crate::io::{AnySource, Target};
use crate::backend::Backend;
use crate::buffer::Buffer;
use crate::error::Error;

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
    pub _m: PhantomData<(K, B)>,
}

impl<K: Kind, B: Backend> Clone for Data<K, B> {
    fn clone(&self) -> Self {
        Self {
            root: Arc::clone(&self.root),
            ctx: Arc::clone(&self.ctx),
            spec: Arc::clone(&self.spec),
            _m: PhantomData,
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
        B::materialize(&self.ctx, &self.root, &wu.erase())
    }

    /// Read-only access to this handle's backend context (e.g. a Source
    /// adapter that materializes an upstream pipeline of the *same* backend).
    pub fn ctx(&self) -> &Arc<B::Ctx> {
        &self.ctx
    }

    /// The single public terminal: evaluate `wu` and hand the result to a
    /// `Target`. A host target downloads + decodes (`Out = Vec<…>`); a disk
    /// target writes a file (`Out = ()`); the viewport target clones the GPU
    /// `Buffer` (`Out = Buffer`, no download).
    pub fn extract<T: Target<K, B>>(&self, t: &T, wu: K::WorkUnit) -> Result<T::Out, Error> {
        let buf = self.materialize(wu.clone())?;
        t.extract(&buf, &wu, &self.ctx)
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
            _m: PhantomData,
        }
    }

    /// Evaluates the subgraph up to this node and extracts the result via `target`.
    pub fn pull<T: crate::io::Target<K, B>>(&self, target: &T, wu: K::WorkUnit) -> Result<T::Out, crate::error::Error> {
        let buf = self.materialize(wu.clone())?;
        target.extract(&buf, &wu, &self.ctx)
    }
}

use std::collections::{HashMap, HashSet};

/// Agnostic stateful walker for DAG traversals.
/// It owns the transient maps and sets needed to execute a pass over the graph.
pub struct GraphWalk<'a, B: Backend> {
    pub root: &'a Arc<Node<B>>,
    pub demands: HashMap<usize, crate::work_unit::WorkUnit>,
    pub lowered: HashSet<usize>,
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
    /// box) in its entry; `None` prunes a whole input subgraph.
    pub fn demand(&mut self, root_wu: &crate::work_unit::WorkUnit) {
        let mut stack = vec![(self.root.clone(), root_wu.clone())];

        while let Some((node, wu)) = stack.pop() {
            let k = Arc::as_ptr(&node) as *const () as usize;
            self.demands
                .entry(k)
                .and_modify(|e| *e = e.union(&wu))
                .or_insert_with(|| wu.clone());

            for (input, child) in node.inputs().iter().zip(node.demand_erased(&wu)) {
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
        let k = Arc::as_ptr(node) as *const () as usize;
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
