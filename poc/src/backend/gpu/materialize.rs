use std::sync::Arc;
use crate::error::Error;
use crate::node::Node;
use crate::work_unit::WorkUnit;
use super::{GpuBackend, GpuBuilder, GpuContext};

pub struct Materializer<'a> {
    pub ctx: &'a Arc<GpuContext>,
    pub root: &'a Arc<Node<GpuBackend>>,
}

impl<'a> Materializer<'a> {
    pub fn execute(&self, root_wu: &WorkUnit) -> Result<crate::buffer::Buffer<GpuBackend>, Error> {
        // 1. Demand walk (inverse map): every live node's resolved WorkUnit for
        //    *this* request. Pruned inputs (demand -> None) never enter the map.
        let mut walk = crate::node::GraphWalk::new(self.root);
        walk.demand(root_wu);

        // 2. Lower walk: each live node injects its GPU config at its resolved
        //    WorkUnit. Fully type-blind — `node.lower` is the only concrete
        //    site, and a Source's `lower` fetches + uploads its own buffer via
        //    `builder.ctx()`, so there is no `Node::Source` branch here.
        let mut builder = GpuBuilder::new(self.ctx.clone());
        walk.lower(|node, n_wu| {
            let node_key = Arc::as_ptr(node) as *const () as usize;
            let input_keys: Vec<usize> = node
                .inputs()
                .iter()
                .map(|i| Arc::as_ptr(i.src()) as *const () as usize)
                .collect();
            builder.enter(node_key, &input_keys, n_wu.clone());
            node.lower(&mut builder);
        });
        if let Some(e) = builder.take_error() {
            return Err(e);
        }

        // 3. Emit Slang + hash, 4. compile (pipeline-cache by IR hash),
        //    5. encode + dispatch. Output sized by the agnostic byte_size.
        let slang = super::emit::emit_slang(&builder, self.ctx.wg_dim);
        let hash = super::emit::hash_slang(&slang);
        let pass = super::compile::compile(self.ctx.as_ref(), &builder, slang, hash)?;

        let spec = self.root.output_kind();
        let out_bytes = spec.byte_size(root_wu);
        let dims = match root_wu {
            WorkUnit::Region(r) => (r.w.max(0) as u32, r.h.max(0) as u32),
            _ => (1, 1),
        };
        let payload = super::compile::dispatch(self.ctx.as_ref(), &pass, &builder, out_bytes, dims)?;

        Ok(crate::buffer::Buffer { payload, spec })
    }
}
