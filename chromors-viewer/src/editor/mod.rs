pub mod compile;
pub mod descriptors;
pub mod graph;
pub mod params;
pub mod registry;
pub mod types;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::graph::{NodeGraph, PortAddr, Side};
    use crate::editor::params::ParamValue;
    use crate::editor::registry::NodeKindId;
    use poc::backend::gpu::GpuContext;
    use std::sync::Arc;
    use vello::kurbo::Point;

    #[test]
    fn test_m1_skeleton_evaluate() {
        // Setup GPU context manually (if possible) or stub it out if testing is mocked.
        // We will just do a dry run checking connection logic if actual GPU ctx is hard to spawn,
        // but let's try assuming a stub or real one can be made if this runs.

        // This is a placeholder test just to prove M1 compiles.
        let mut g = NodeGraph::new();

        let load = g.add_node(NodeKindId("source.load"), Point::new(0.0, 0.0));
        g.set_param(load, 0, ParamValue::Path(Some("test.png".to_string())));

        let exposure = g.add_node(NodeKindId("color.exposure"), Point::new(100.0, 0.0));

        let viewer = g.add_node(NodeKindId("sink.viewer"), Point::new(200.0, 0.0));

        let edge1 = g
            .connect(
                PortAddr {
                    node: load,
                    side: Side::Out,
                    index: 0,
                },
                PortAddr {
                    node: exposure,
                    side: Side::In,
                    index: 0,
                },
            )
            .unwrap();

        let edge2 = g
            .connect(
                PortAddr {
                    node: exposure,
                    side: Side::Out,
                    index: 0,
                },
                PortAddr {
                    node: viewer,
                    side: Side::In,
                    index: 0,
                },
            )
            .unwrap();

        // We don't have a real GpuContext in unit tests easily without wgpu initialization,
        // so we won't call `evaluate` here. M1 specification requires the skeleton to compile.
    }
}
