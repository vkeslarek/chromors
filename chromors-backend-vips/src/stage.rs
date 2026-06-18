use std::hash::Hasher;
use std::sync::Arc;

use crate::Kind;
use crate::VipsBackend;
use crate::VipsBuilder;
use crate::VipsBand;
use crate::VipsHandle;
use crate::Buffer;
use crate::Error;
use crate::Source;
use crate::WorkUnitFor;
use chromors_core::stage::BoundarySource;

impl<K> Source<VipsBackend> for BoundarySource<K, VipsBackend>
where
    K: VipsBand,
{
    type Kind = K;

    fn spec(&self) -> Arc<K> {
        self.upstream.spec.clone()
    }

    fn fetch(&self, _ctx: &(), wu: &K::WorkUnit) -> Result<Buffer<VipsBackend>, Error> {
        self.serve(wu)
    }

    fn lower(&self, cx: &mut VipsBuilder) {
        let wu = cx.wu().clone();
        let Some(typed) = K::WorkUnit::typed(&wu) else {
            // VipsBuilder has no fail() — vips ops are not fused, so we panic
            // here the same way a missing input would. In practice the DAG
            // ensures work-unit shapes always match.
            panic!("BoundarySource<VipsBackend>: work unit shape mismatch");
        };
        match self.serve(&typed) {
            Ok(buf) => {
                let handle = (*buf.payload).clone();
                cx.emit(handle);
            }
            Err(e) => {
                // Mirror GPU convention: propagate via panic (no builder
                // error accumulator on the vips side).
                panic!("BoundarySource<VipsBackend> serve failed: {e}");
            }
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.content);
    }
}
