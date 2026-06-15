use crate::editor::params::ParamSpec;
use crate::editor::registry::{
    BuildError, Category, NodeDescriptor, NodeKindId, Registry, SocketSpec,
};
use crate::editor::types::{DataType, PortValue};

pub fn register(r: &mut Registry) {
    r.add(NodeDescriptor {
        id: NodeKindId("source.load"),
        title: "Load Image",
        category: Category::Source,
        inputs: vec![],
        outputs: vec![SocketSpec {
            name: "image",
            ty: DataType::Image,
        }],
        params: vec![ParamSpec::path("file")],
        build: |_inputs, params, ctx| {
            use poc::backend::vips::VipsBackend;
            use poc::data::image::{Image2D, VipsImageSource};
            use poc::node::Data;
            use std::sync::Arc;

            let path = params[0]
                .path()
                .ok_or_else(|| BuildError("load: no file selected".into()))?;
            let vips = Image2D::<VipsBackend>::open(path)
                .map_err(|e| BuildError(format!("open failed: {e:?}")))?;
            let src = VipsImageSource::new(vips);
            let gpu_img = Data::from_source(Arc::new(src), ctx.clone());
            Ok(vec![PortValue::Image(gpu_img)])
        },
    });
}
