use crate::editor::registry::{
    BuildError, Category, NodeDescriptor, NodeKindId, Registry, SocketSpec,
};
use crate::editor::types::{DataType, PortValue};

pub fn register(r: &mut Registry) {
    r.add(NodeDescriptor {
        id: NodeKindId("sink.viewer"),
        title: "Viewer",
        category: Category::Sink,
        inputs: vec![SocketSpec {
            name: "image",
            ty: DataType::Image,
        }],
        outputs: vec![], // A sink produces nothing for the graph
        params: vec![],
        build: |inputs, _p, _ctx| {
            let img = inputs[0]
                .as_ref()
                .ok_or_else(|| BuildError("viewer: nothing connected".into()))?
                .clone();
            Ok(vec![img]) // expose the (passed-through) value as output[0]
        },
    });
}
