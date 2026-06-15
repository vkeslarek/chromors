//! Lazy materialization boundary — [`BoundarySource`] and [`Data::stage`].
//!
//! A `Source` is a hard cut in GPU pass fusion: the engine fetches a real
//! `Buffer<B>` there before the downstream pass fuses from it. [`BoundarySource`]
//! turns that cut into an ordinary graph leaf, generic over **any** `GpuView`
//! Kind (Region/Range/Atomic):
//!
//! - [`Data::stage`] inserts a boundary with no store: the upstream sub-DAG is
//!   materialized in full (one pass) and re-injected as a source for the
//!   downstream pass (another pass). Two dispatches, data never leaves the GPU.
//!   This is the barrier data-dependent ops need (e.g. a histogram reduction
//!   must finish before a CDF/LUT step reads it) — see [`crate::data::histogram`].
//! - [`crate::cache::Cached`] is the same boundary *with* a [`crate::cache::RegionCache`]
//!   store: a hit skips materializing the upstream entirely (memoized boundary).
//!
//! `stage()` is **not memoized** — each pull re-runs the staged sub-DAG. Wrap
//! with `.cache()` (on the staged tip, or upstream of `stage`) if a result
//! should persist across pulls.

use std::hash::Hasher;
use std::sync::Arc;

use crate::backend::Backend;
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::buffer::Buffer;
use crate::cache::{CacheKey, RegionCache};
use crate::error::Error;
use crate::io::Source;
use crate::kind::Kind;
use crate::node::{Data, NodeId};
use crate::work_unit::WorkUnitFor;

/// A lazy materialization boundary. On `fetch`/`lower` it serves the requested
/// work unit from `store` (if any and present), or materializes the upstream
/// sub-DAG for exactly that work unit and (if `store` is set) caches it.
///
/// Generic over the upstream Kind; the per-backend lowering is implemented for
/// each backend below. [`crate::cache::Cached::handle`] builds one with
/// `store: Some(..)`; [`Data::stage`] builds one with `store: None`.
pub struct BoundarySource<K: Kind, B: Backend> {
    pub(crate) upstream: Data<K, B>,
    pub(crate) store: Option<Arc<RegionCache<B>>>,
    pub(crate) content: u64,
}

impl<K: Kind, B: Backend> BoundarySource<K, B> {
    /// Hit the store (if any), else materialize the upstream for `wu` and, if a
    /// store is configured, cache the result.
    fn serve(&self, wu: &K::WorkUnit) -> Result<Buffer<B>, Error> {
        let erased = wu.erase();
        if let Some(store) = &self.store {
            let key = CacheKey::new(self.content, &erased);
            if let Some(buf) = store.get(&key) {
                return Ok(buf);
            }
            let buf = self.upstream.materialize(wu.clone())?;
            let len = buf.spec.byte_size(&erased);
            store.insert(key, &buf, len);
            return Ok(buf);
        }
        self.upstream.materialize(wu.clone())
    }
}

/// GPU lowering: materialize (or serve from the store) and re-inject the result
/// as a decoded input, using the Kind's own [`GpuView::source_params`] for the
/// slot geometry — works for Region (image), Range (LUT), and Atomic
/// (histogram/vectorscope) Kinds alike.
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
        _ctx: &crate::backend::gpu::GpuContext,
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

impl<K: Kind, B: Backend> Data<K, B> {
    /// Insert a lazy materialization barrier: the upstream sub-DAG is computed
    /// in full (its own pass) and re-injected as a source for whatever consumes
    /// the returned tip (a fresh pass). Lazy — nothing runs until pulled, and
    /// each pull re-runs the staged sub-DAG (no memoization; see
    /// [`crate::cache::Cached`] for that).
    ///
    /// This is the barrier data-dependent ops need: a reduction (e.g. a
    /// histogram) must be fully accumulated before a consumer (e.g. a CDF→LUT
    /// step) can read it, and that cannot happen within one fused pass.
    pub fn stage(&self) -> Self
    where
        BoundarySource<K, B>: Source<B, Kind = K>,
    {
        let src = BoundarySource {
            upstream: self.clone(),
            store: None,
            content: NodeId::of(&self.root).0 as u64,
        };
        Data::from_source(Arc::new(src), self.ctx.clone())
    }
}
