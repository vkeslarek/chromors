//! Generic materialization boundary.
//!
//! `BoundarySource<K, B>` is a `Source`-shaped wrapper around an upstream
//! `Data<K, B>`. The backend-specific `impl Source<B>` lives in each backend
//! crate (only the wiring differs: GPU injects a decoded buffer slot, vips
//! emits a VipsHandle). The memoization store (`RegionCache`) is optional —
//! `.stage()` omits it (pure pass-boundary), `.cache()` attaches it.

use std::sync::Arc;

use crate::stage_cache::RegionCache;
use crate::work_unit::WorkUnitFor;
use crate::{Backend, Buffer, Data, Error, Kind};

pub struct BoundarySource<K: Kind, B: Backend> {
    pub upstream: Data<K, B>,
    pub content: u64,
    pub store: Option<Arc<RegionCache<B>>>,
}

impl<K: Kind, B: Backend> BoundarySource<K, B> {
    pub fn serve(&self, wu: &K::WorkUnit) -> Result<Buffer<B>, Error> {
        if let Some(store) = &self.store {
            let erased = wu.erase();
            let key = crate::stage_cache::CacheKey::new(self.content, &erased);
            if let Some(buf) = store.get(&key) {
                return Ok(buf);
            }
            let buf = self.upstream.materialize(wu.clone())?;
            let len = buf.spec.byte_size(&erased);
            store.insert(key, &buf, len);
            Ok(buf)
        } else {
            self.upstream.materialize(wu.clone())
        }
    }
}
