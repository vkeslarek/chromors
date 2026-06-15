use crate::editor::params::{ParamSpec, ParamValue};
use crate::editor::types::{DataType, PortValue};
use poc::backend::gpu::GpuContext;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeKindId(pub &'static str);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Category {
    Source,
    Color,
    Filter,
    Geometry,
    Composite,
    Sink,
}

pub struct SocketSpec {
    pub name: &'static str,
    pub ty: DataType,
}

pub struct BuildError(pub String);

pub type BuildFn = fn(
    inputs: &[Option<PortValue>],
    params: &[ParamValue],
    ctx: &Arc<GpuContext>,
) -> Result<Vec<PortValue>, BuildError>;

pub struct NodeDescriptor {
    pub id: NodeKindId,
    pub title: &'static str,
    pub category: Category,
    pub inputs: Vec<SocketSpec>,
    pub outputs: Vec<SocketSpec>,
    pub params: Vec<ParamSpec>,
    pub build: BuildFn,
}

#[derive(Default)]
pub struct Registry {
    by_id: HashMap<NodeKindId, NodeDescriptor>,
    ordered: Vec<NodeKindId>,
}

impl Registry {
    pub fn add(&mut self, desc: NodeDescriptor) {
        self.ordered.push(desc.id);
        self.by_id.insert(desc.id, desc);
    }
    pub fn get(&self, id: NodeKindId) -> &NodeDescriptor {
        self.by_id
            .get(&id)
            .expect("NodeKindId not found in registry")
    }
    pub fn iter(&self) -> impl Iterator<Item = &NodeDescriptor> {
        self.ordered.iter().map(move |id| self.get(*id))
    }
    pub fn iter_category(&self, c: Category) -> impl Iterator<Item = &NodeDescriptor> {
        self.ordered
            .iter()
            .map(move |id| self.get(*id))
            .filter(move |d| d.category == c)
    }
}

pub fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(build_registry)
}

fn build_registry() -> Registry {
    let mut r = Registry::default();
    crate::editor::descriptors::sources::register(&mut r);
    crate::editor::descriptors::color::register(&mut r);
    crate::editor::descriptors::sinks::register(&mut r);
    r
}
