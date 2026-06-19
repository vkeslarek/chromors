use std::hash::Hasher;
use std::sync::Arc;

use crate::Buffer;
use crate::Error;
use crate::GpuBackend;
use crate::GpuBuilder;
use crate::GpuView;
use crate::Kind;
use crate::Source;
use crate::WorkUnitFor;
use chromors_core::stage::BoundarySource;

impl<K> Source<GpuBackend> for BoundarySource<K, GpuBackend>
where
    K: GpuView,
{
    type Kind = K;

    fn spec(&self) -> Arc<K> {
        self.upstream.spec.clone()
    }

    fn fetch(
        &self,
        _ctx: &crate::GpuContext,
        wu: &K::WorkUnit,
    ) -> Result<Buffer<GpuBackend>, Error> {
        self.serve(wu)
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let Some(typed) = K::WorkUnit::typed(&wu) else {
            cx.fail(Error::InvalidWorkUnit(
                "boundary source: work unit shape mismatch".into(),
            ));
            return;
        };
        match self.serve(&typed) {
            Ok(buf) => {
                cx.input(
                    self.spec().input(),
                    self.spec().source_params(&wu),
                    buf.payload,
                );
            }
            Err(e) => cx.fail(e),
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.content);
    }
}
