use crate::editor::params::ParamSpec;
use crate::editor::registry::{
    BuildError, Category, NodeDescriptor, NodeKindId, Registry, SocketSpec,
};
use crate::editor::types::{DataType, PortValue};

pub fn register(r: &mut Registry) {
    r.add(NodeDescriptor {
        id: NodeKindId("color.exposure"),
        title: "Exposure",
        category: Category::Color,
        inputs: vec![SocketSpec {
            name: "image",
            ty: DataType::Image,
        }],
        outputs: vec![SocketSpec {
            name: "out",
            ty: DataType::Image,
        }],
        params: vec![
            ParamSpec::float("stops", -5.0, 5.0, 0.0),
            ParamSpec::float("preserve", 0.0, 1.0, 0.0),
        ],
        build: |inputs, params, _ctx| {
            let img = inputs[0]
                .as_ref()
                .ok_or_else(|| BuildError("exposure: 'image' not connected".into()))?
                .image()
                .clone();
            let stops = params[0].float() as f32;
            let preserve = params[1].float() as f32;
            Ok(vec![PortValue::Image(img.exposure(stops, preserve))])
        },
    });

    r.add(NodeDescriptor {
        id: NodeKindId("color.invert"),
        title: "Invert",
        category: Category::Color,
        inputs: vec![SocketSpec {
            name: "image",
            ty: DataType::Image,
        }],
        outputs: vec![SocketSpec {
            name: "out",
            ty: DataType::Image,
        }],
        params: vec![],
        build: |inputs, _p, _ctx| {
            let img = inputs[0]
                .as_ref()
                .ok_or_else(|| BuildError("invert: 'image' not connected".into()))?
                .image()
                .clone();
            Ok(vec![PortValue::Image(img.invert())])
        },
    });
}
