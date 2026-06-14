//! Pluggable materialization cache — a memoization **boundary** expressed as a
//! `Source`.
//!
//! ## Why a Source, not a backend feature
//!
//! The GPU backend fuses consecutive ops into one shader; there is no place
//! *inside* a fused pass to "stop and reuse a previous result", and forcing one
//! would fight the whole point of fusion. So the cache is not a backend concept
//! at all — it is an ordinary graph leaf.
//!
//! A [`Cached`] wraps an upstream pipeline tip (`Data<K, B>`). Reading it builds
//! a fresh DAG whose **source** is a [`CacheSource`]. When a downstream DAG pulls
//! a region:
//!
//! 1. the `CacheSource` checks its store for that region;
//! 2. **hit** → the cached `Buffer<B>` is fed straight in (the upstream DAG never
//!    runs);
//! 3. **miss** → the upstream DAG is materialized for *exactly that region*
//!    (priority), the result is cached, then fed in.
//!
//! Because a `Source` is a hard materialization point (the engine fetches a real
//! `Buffer` there), this is a deliberate cut in the fusion chain — the upstream
//! collapses into one buffer, and the downstream fuses fresh from that buffer.
//! That cut is the cache.
//!
//! ## Layer-stack use case
//!
//! Insert a `.cache()` at each intermediate result you don't want recomputed:
//!
//! ```ignore
//! let base   = src.exposure(0.3, 0.0).blur(8.0).cache(); // boundary A
//! let layer1 = base.handle().saturation(1.2);            // pulls A (cached)
//! let layer2 = base.handle().invert();                  // ALSO pulls A — no recompute
//! ```
//!
//! Both branches share `base`'s materialized tiles instead of re-running the
//! exposure+blur chain twice.
//!
//! ## Active (eager) priming
//!
//! The consumer pulls the regions it needs first (priority). To keep the rest
//! warm, call [`Cached::prime`] with the remaining regions — typically from a
//! background worker (the viewport already owns a fetch thread). The engine
//! itself stays synchronous; "active cache" = the caller driving `prime` on idle.
//!
//! ## Tiers
//!
//! v1 is single-tier (VRAM-resident `Buffer<B>`) with CLOCK eviction by byte
//! budget. The store key/accounting is tier-ready: a future RAM/Disk spill (port
//! of `pixors-engine`'s `TieredCache`) slots in behind [`RegionCache`] without
//! touching `CacheSource`.

use std::collections::{HashMap, VecDeque};
use std::hash::Hasher;
use std::sync::{Arc, Mutex};

use crate::backend::Backend;
use crate::backend::gpu::view::RegionParams;
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::buffer::Buffer;
use crate::error::Error;
use crate::io::Source;
use crate::kind::{AnyKind, Kind};
use crate::node::{Data, NodeId};
use crate::work_unit::{Region, WorkUnit, WorkUnitFor};

const MIB: u64 = 1024 * 1024;
/// Default VRAM budget for a freshly-created store.
pub const DEFAULT_BUDGET: u64 = 256 * MIB;

// ── Key ─────────────────────────────────────────────────────────────────────

/// Content-addressed cache key: which boundary (`content`) and which slice
/// (`wu`). `wu` is the `Debug` rendering of the erased `WorkUnit`, which is a
/// total, allocation-cheap, shape-agnostic identity for any work unit
/// (Region/Range/Atomic) without adding a shape `match` here.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CacheKey {
    pub content: u64,
    pub wu: String,
}

impl CacheKey {
    fn new(content: u64, wu: &WorkUnit) -> Self {
        Self {
            content,
            wu: format!("{wu:?}"),
        }
    }
}

// ── Store ───────────────────────────────────────────────────────────────────

struct Entry<B: Backend> {
    payload: Arc<B::Payload>,
    spec: Arc<dyn AnyKind>,
    /// Logical byte length for budget accounting.
    len: u64,
    /// CLOCK recency bit.
    used: bool,
}

struct Inner<B: Backend> {
    map: HashMap<CacheKey, Entry<B>>,
    ring: VecDeque<CacheKey>,
    bytes: u64,
    budget: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

/// Diagnostics snapshot (see [`RegionCache::stats`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub entries: usize,
    pub bytes: u64,
    pub budget: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// A shared, content-addressed buffer store with CLOCK (second-chance) eviction
/// by VRAM byte budget. Cheap to `Arc`-clone and share across several
/// [`Cached`] boundaries (each boundary namespaces its keys by `content`).
pub struct RegionCache<B: Backend> {
    inner: Mutex<Inner<B>>,
}

impl<B: Backend> RegionCache<B> {
    pub fn new(budget: u64) -> Self {
        Self {
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                ring: VecDeque::new(),
                bytes: 0,
                budget,
                hits: 0,
                misses: 0,
                evictions: 0,
            }),
        }
    }

    /// Look up a slice; sets its recency bit. Returns a `Buffer<B>` view (Arc
    /// clones, no copy) on hit.
    pub fn get(&self, key: &CacheKey) -> Option<Buffer<B>> {
        let mut g = self.inner.lock().unwrap();
        match g.map.get_mut(key) {
            Some(e) => {
                e.used = true;
                let buf = Buffer {
                    payload: e.payload.clone(),
                    spec: e.spec.clone(),
                };
                g.hits += 1;
                Some(buf)
            }
            None => {
                g.misses += 1;
                None
            }
        }
    }

    /// Insert (or replace) a slice and enforce the budget. `len` is the logical
    /// byte size (`AnyKind::byte_size`).
    pub fn insert(&self, key: CacheKey, buf: &Buffer<B>, len: u64) {
        let mut g = self.inner.lock().unwrap();
        // Replace any prior entry so accounting stays exact.
        if let Some(old) = g.map.remove(&key) {
            g.bytes -= old.len.min(g.bytes);
        }
        g.map.insert(
            key.clone(),
            Entry {
                payload: buf.payload.clone(),
                spec: buf.spec.clone(),
                len,
                used: false,
            },
        );
        g.bytes += len;
        g.ring.push_back(key);
        Self::evict(&mut g);
    }

    /// Drop every entry whose `content` matches — invalidate one boundary.
    pub fn invalidate_content(&self, content: u64) {
        let mut g = self.inner.lock().unwrap();
        let doomed: Vec<CacheKey> = g
            .map
            .keys()
            .filter(|k| k.content == content)
            .cloned()
            .collect();
        for k in doomed {
            if let Some(e) = g.map.remove(&k) {
                g.bytes -= e.len.min(g.bytes);
                g.evictions += 1;
            }
        }
    }

    /// Reset the byte budget and immediately enforce it.
    pub fn set_budget(&self, budget: u64) {
        let mut g = self.inner.lock().unwrap();
        g.budget = budget;
        Self::evict(&mut g);
    }

    pub fn stats(&self) -> CacheStats {
        let g = self.inner.lock().unwrap();
        CacheStats {
            entries: g.map.len(),
            bytes: g.bytes,
            budget: g.budget,
            hits: g.hits,
            misses: g.misses,
            evictions: g.evictions,
        }
    }

    /// CLOCK sweep. A victim must be (a) past its recency second chance and
    /// (b) not referenced anywhere else (`Arc::strong_count == 1`, i.e. the
    /// store holds the only handle — nothing downstream is using it right now).
    fn evict(g: &mut Inner<B>) {
        let mut skips = 0usize;
        while g.bytes > g.budget {
            let Some(key) = g.ring.pop_front() else {
                break; // ring drained
            };
            let Some(e) = g.map.get_mut(&key) else {
                continue; // stale slot — entry already gone
            };
            // Still in use by a live holder → never reclaim; second chance.
            if Arc::strong_count(&e.payload) > 1 {
                e.used = false;
                g.ring.push_back(key);
                skips += 1;
                if skips > g.ring.len() {
                    break; // everything is pinned/in-use — accept overshoot
                }
                continue;
            }
            // Recency second chance.
            if e.used {
                e.used = false;
                g.ring.push_back(key);
                skips += 1;
                if skips > g.ring.len() {
                    break;
                }
                continue;
            }
            // Victim.
            skips = 0;
            let len = e.len;
            g.map.remove(&key);
            g.bytes -= len.min(g.bytes);
            g.evictions += 1;
        }
    }
}

// ── CacheSource ───────────────────────────────────────────────────────────────

/// The graph leaf a [`Cached`] reads through. On `fetch`/`lower` it serves the
/// requested region from the store, or materializes the upstream DAG for exactly
/// that region on a miss and caches it. Generic over the upstream Kind; the
/// per-backend lowering is implemented for each backend below.
pub struct CacheSource<K: Kind, B: Backend> {
    upstream: Data<K, B>,
    store: Arc<RegionCache<B>>,
    content: u64,
}

impl<K: Kind, B: Backend> CacheSource<K, B> {
    /// Hit the store, else materialize the upstream for `wu` (priority) and cache.
    fn serve(&self, wu: &K::WorkUnit) -> Result<Buffer<B>, Error> {
        let erased = wu.erase();
        let key = CacheKey::new(self.content, &erased);
        if let Some(buf) = self.store.get(&key) {
            return Ok(buf);
        }
        let buf = self.upstream.materialize(wu.clone())?;
        let len = buf.spec.byte_size(&erased);
        self.store.insert(key, &buf, len);
        Ok(buf)
    }
}

/// GPU lowering. Constrained to `WorkUnit = Region` (image-shaped data) so the
/// fetched buffer can be re-fed through the Kind's decode `View` with tight
/// region geometry — exactly how `VipsImageSource`/`GpuConstantSource` feed a
/// fetched buffer back into a fused pass.
impl<K> Source<GpuBackend> for CacheSource<K, GpuBackend>
where
    K: Kind<WorkUnit = Region> + GpuView,
{
    type Kind = K;

    fn spec(&self) -> Arc<K> {
        self.upstream.spec.clone()
    }

    fn fetch(&self, _ctx: &crate::backend::gpu::GpuContext, wu: &Region) -> Result<Buffer<GpuBackend>, Error> {
        self.serve(wu)
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let WorkUnit::Region(region) = &wu else {
            cx.fail(Error::InvalidWorkUnit(
                "cache source expects a Region".into(),
            ));
            return;
        };
        match self.serve(region) {
            Ok(buf) => {
                // `serve` returns the upstream root buffer for `region`, which is
                // tightly packed at the region's dimensions.
                let geom = RegionParams::tight(region.w, region.h);
                cx.input(
                    self.spec().input(),
                    geom.into_block("region_in_{slot}"),
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

// ── Cached handle ─────────────────────────────────────────────────────────────

/// A materialization boundary over an upstream pipeline tip.
///
/// [`Cached::handle`] hands out a `Data` whose source is this boundary's
/// [`CacheSource`]; pulling it serves from the store or materializes the upstream
/// on a miss. [`Cached::prime`] eagerly warms additional regions.
pub struct Cached<K: Kind, B: Backend> {
    upstream: Data<K, B>,
    store: Arc<RegionCache<B>>,
    content: u64,
}

impl<K: Kind, B: Backend> Cached<K, B> {
    /// The shared store backing this boundary (clone the `Arc` to reuse it
    /// across boundaries, query [`RegionCache::stats`], or change the budget).
    pub fn store(&self) -> &Arc<RegionCache<B>> {
        &self.store
    }

    /// This boundary's content id (its key namespace in the store).
    pub fn content(&self) -> u64 {
        self.content
    }
}

impl<K, B> Cached<K, B>
where
    K: Kind,
    B: Backend,
    CacheSource<K, B>: Source<B, Kind = K>,
{
    /// A fresh `Data` reading through this cache boundary. Cheap; call it once
    /// per downstream branch.
    pub fn handle(&self) -> Data<K, B> {
        let src = CacheSource {
            upstream: self.upstream.clone(),
            store: self.store.clone(),
            content: self.content,
        };
        Data::from_source(Arc::new(src), self.upstream.ctx.clone())
    }
}

impl<K: Kind, B: Backend> Cached<K, B> {
    /// Eagerly materialize and cache `regions` that are not already resident
    /// (the "active cache" warm-up). Already-cached regions are skipped, so this
    /// is cheap to call repeatedly. Run it from a background worker after the
    /// consumer has pulled the regions it needs *now* (priority).
    pub fn prime(&self, regions: &[K::WorkUnit]) -> Result<(), Error> {
        for wu in regions {
            let erased = wu.erase();
            let key = CacheKey::new(self.content, &erased);
            if self.store.get(&key).is_some() {
                continue; // already warm
            }
            let buf = self.upstream.materialize(wu.clone())?;
            let len = buf.spec.byte_size(&erased);
            self.store.insert(key, &buf, len);
        }
        Ok(())
    }
}

// ── Ergonomics on Data ────────────────────────────────────────────────────────

impl<K: Kind, B: Backend> Data<K, B> {
    /// Wrap this pipeline tip in a cache boundary with a fresh store
    /// ([`DEFAULT_BUDGET`]). See [`Cached`].
    pub fn cache(&self) -> Cached<K, B> {
        self.cache_with(Arc::new(RegionCache::new(DEFAULT_BUDGET)))
    }

    /// Wrap this pipeline tip in a cache boundary backed by an existing store
    /// (share one store across several boundaries / control its budget).
    pub fn cache_with(&self, store: Arc<RegionCache<B>>) -> Cached<K, B> {
        // Pointer identity of the upstream root is a stable, collision-free
        // namespace for this boundary's keys for as long as `Cached` keeps the
        // upstream DAG alive (it does — it holds the `Data`).
        let content = NodeId::of(&self.root).0 as u64;
        Cached {
            upstream: self.clone(),
            store,
            content,
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────
//
// The store's accounting + CLOCK eviction are backend-generic, so they are
// proven here against a trivial fake backend (no GPU needed). The end-to-end
// hit/miss behaviour of `CacheSource` on real GPU pipelines is exercised by the
// `tests/gpu` suite.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Builder;
    use std::any::Any;

    /// Minimal backend whose payload is a byte vec — enough to build `Buffer`s
    /// and exercise `RegionCache`.
    struct TestBackend;
    struct TestBuilder;

    impl Builder<TestBackend> for TestBuilder {
        fn new(_ctx: Arc<()>) -> Self {
            TestBuilder
        }
        fn enter(&mut self, _node: NodeId, _inputs: &[NodeId], _wu: &WorkUnit) {}
        fn finish(
            self,
            _root: NodeId,
            _spec: Arc<dyn AnyKind>,
            _root_wu: &WorkUnit,
        ) -> Result<Buffer<TestBackend>, Error> {
            Err(Error::InvalidWorkUnit("test backend never materializes".into()))
        }
    }

    impl Backend for TestBackend {
        type Ctx = ();
        type Payload = Vec<u8>;
        type Builder = TestBuilder;
    }

    /// A spec that simply reports a fixed byte size.
    #[derive(Debug)]
    struct SizeKind(u64);
    impl AnyKind for SizeKind {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn byte_size(&self, _wu: &WorkUnit) -> u64 {
            self.0
        }
        fn dyn_hash(&self, _state: &mut dyn Hasher) {}
    }

    fn key(content: u64, n: i32) -> CacheKey {
        CacheKey::new(
            content,
            &WorkUnit::Region(Region {
                x: n,
                y: 0,
                w: 1,
                h: 1,
                lod: crate::work_unit::Lod(0),
            }),
        )
    }

    fn buf(len: u64) -> Buffer<TestBackend> {
        Buffer {
            payload: Arc::new(vec![0u8; len as usize]),
            spec: Arc::new(SizeKind(len)),
        }
    }

    #[test]
    fn hit_and_miss_accounting() {
        let c = RegionCache::<TestBackend>::new(10_000);
        assert!(c.get(&key(1, 0)).is_none());
        c.insert(key(1, 0), &buf(40), 40);
        c.insert(key(1, 1), &buf(40), 40);
        assert!(c.get(&key(1, 0)).is_some());
        assert!(c.get(&key(1, 2)).is_none());
        let s = c.stats();
        assert_eq!(s.entries, 2);
        assert_eq!(s.bytes, 80);
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 2);
    }

    #[test]
    fn over_budget_evicts_coldest() {
        let c = RegionCache::<TestBackend>::new(100);
        // Each buffer is dropped locally right after insert, so the store holds
        // the only Arc (strong_count == 1) and the entry is reclaimable.
        c.insert(key(1, 0), &buf(40), 40);
        c.insert(key(1, 1), &buf(40), 40);
        c.insert(key(1, 2), &buf(40), 40); // 120 > 100 → one cold entry evicted
        let s = c.stats();
        assert!(s.bytes <= 100, "bytes {} must respect budget", s.bytes);
        assert!(s.evictions >= 1);
    }

    #[test]
    fn recently_used_survives_second_chance() {
        let c = RegionCache::<TestBackend>::new(100);
        c.insert(key(1, 0), &buf(40), 40);
        c.insert(key(1, 1), &buf(40), 40);
        let _ = c.get(&key(1, 0)); // touch 0 → recency bit set
        c.insert(key(1, 2), &buf(40), 40); // evicts the cold one (1), keeps 0
        assert!(c.get(&key(1, 0)).is_some(), "touched entry must survive");
        assert!(c.get(&key(1, 1)).is_none(), "cold entry must be evicted");
    }

    #[test]
    fn in_use_entry_is_never_evicted() {
        let c = RegionCache::<TestBackend>::new(10);
        let held = buf(40);
        c.insert(key(1, 0), &held, 40); // strong_count >= 2 while `held` lives
        c.insert(key(1, 1), &buf(40), 40); // over budget, but 0 is in use
        assert!(c.get(&key(1, 0)).is_some(), "in-use entry protected");
        drop(held);
        c.insert(key(1, 2), &buf(4), 4); // now 0 is reclaimable
        assert!(c.get(&key(1, 0)).is_none(), "freed entry now evictable");
    }

    #[test]
    fn invalidate_content_drops_only_that_namespace() {
        let c = RegionCache::<TestBackend>::new(10_000);
        c.insert(key(1, 0), &buf(40), 40);
        c.insert(key(2, 0), &buf(40), 40);
        c.invalidate_content(1);
        assert!(c.get(&key(1, 0)).is_none());
        assert!(c.get(&key(2, 0)).is_some());
        assert_eq!(c.stats().bytes, 40);
    }
}
