//! Pluggable materialization cache expressed as a Source-based boundary.
//!
//! `RegionCache<B>` is generic over any backend. The actual `Source<B>` impl
//! for `BoundarySource<K,B>` lives in each backend crate — plugging in the
//! cache for a backend is just one more `impl Source<B>`.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use crate::{AnyKind, Backend, Buffer, Data, Error, Kind, NodeId, Source, WorkUnit};
use crate::work_unit::WorkUnitFor;
use crate::stage::BoundarySource;

const MIB: u64 = 1024 * 1024;
pub const DEFAULT_BUDGET: u64 = 256 * MIB;

// ── Key ─────────────────────────────────────────────────────────────────────

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CacheKey {
    pub content: u64,
    pub wu: String,
}

impl CacheKey {
    pub fn new(content: u64, wu: &WorkUnit) -> Self {
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
    len: u64,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub entries: usize,
    pub bytes: u64,
    pub budget: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

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

    pub fn insert(&self, key: CacheKey, buf: &Buffer<B>, len: u64) {
        let mut g = self.inner.lock().unwrap();
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

    fn evict(g: &mut Inner<B>) {
        let mut skips = 0usize;
        while g.bytes > g.budget {
            let Some(key) = g.ring.pop_front() else { break; };
            let Some(e) = g.map.get_mut(&key) else { continue; };
            if Arc::strong_count(&e.payload) > 1 {
                e.used = false;
                g.ring.push_back(key);
                skips += 1;
                if skips > g.ring.len() { break; }
                continue;
            }
            if e.used {
                e.used = false;
                g.ring.push_back(key);
                skips += 1;
                if skips > g.ring.len() { break; }
                continue;
            }
            skips = 0;
            let len = e.len;
            g.map.remove(&key);
            g.bytes -= len.min(g.bytes);
            g.evictions += 1;
        }
    }
}

// ── Cached handle ─────────────────────────────────────────────────────────────

pub struct Cached<K: Kind, B: Backend> {
    upstream: Data<K, B>,
    store: Arc<RegionCache<B>>,
    content: u64,
}

impl<K: Kind, B: Backend> Cached<K, B> {
    pub fn store(&self) -> &Arc<RegionCache<B>> {
        &self.store
    }

    pub fn content(&self) -> u64 {
        self.content
    }
}

impl<K, B> Cached<K, B>
where
    K: Kind,
    B: Backend,
    BoundarySource<K, B>: Source<B, Kind = K>,
{
    pub fn handle(&self) -> Data<K, B> {
        let src = BoundarySource {
            upstream: self.upstream.clone(),
            store: Some(self.store.clone()),
            content: self.content,
        };
        Data::from_source(Arc::new(src), self.upstream.ctx.clone())
    }
}

impl<K: Kind, B: Backend> Cached<K, B> {
    pub fn prime(&self, regions: &[K::WorkUnit]) -> Result<(), Error> {
        for wu in regions {
            let erased = wu.erase();
            let key = CacheKey::new(self.content, &erased);
            if self.store.get(&key).is_some() {
                continue;
            }
            let buf = self.upstream.materialize(wu.clone())?;
            let len = buf.spec.byte_size(&erased);
            self.store.insert(key, &buf, len);
        }
        Ok(())
    }
}

// ── Ergonomics on Data ────────────────────────────────────────────────────────

pub trait CacheExt<K: Kind, B: Backend> {
    fn cache(&self) -> Cached<K, B>;
    fn cache_with(&self, store: Arc<RegionCache<B>>) -> Cached<K, B>;
}

impl<K: Kind, B: Backend> CacheExt<K, B> for Data<K, B> {
    fn cache(&self) -> Cached<K, B> {
        self.cache_with(Arc::new(RegionCache::new(DEFAULT_BUDGET)))
    }

    fn cache_with(&self, store: Arc<RegionCache<B>>) -> Cached<K, B> {
        let content = NodeId::of(&self.root).0 as u64;
        Cached {
            upstream: self.clone(),
            store,
            content,
        }
    }
}

// ── StageExt ────────────────────────────────────────────────────────────────

pub trait StageExt {
    fn stage(&self) -> Self where Self: Sized;
}

impl<K: Kind, B: Backend> StageExt for Data<K, B>
where
    BoundarySource<K, B>: Source<B, Kind = K>,
{
    fn stage(&self) -> Self {
        let src = BoundarySource {
            upstream: self.clone(),
            content: NodeId::of(&self.root).0 as u64,
            store: None,
        };
        Data::from_source(Arc::new(src), self.ctx.clone())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Builder;
    use std::any::Any;
    use std::hash::Hasher;

    struct TestBackend;
    struct TestBuilder;

    impl Builder<TestBackend> for TestBuilder {
        fn new(_ctx: Arc<()>) -> Self { TestBuilder }
        fn enter(&mut self, _node: NodeId, _inputs: &[NodeId], _wu: &WorkUnit) {}
        fn finish(
            self,
            _root: NodeId,
            _spec: Arc<dyn AnyKind>,
            _root_wu: &WorkUnit,
        ) -> Result<Buffer<TestBackend>, Error> {
            Err(Error::Backend("test backend never materializes".into()))
        }
    }

    impl Backend for TestBackend {
        type Ctx = ();
        type Payload = Vec<u8>;
        type Builder = TestBuilder;
    }

    #[derive(Debug)]
    struct SizeKind(u64);
    impl AnyKind for SizeKind {
        fn as_any(&self) -> &dyn Any { self }
        fn byte_size(&self, _wu: &WorkUnit) -> u64 { self.0 }
        fn dyn_hash(&self, _state: &mut dyn Hasher) {}
    }

    fn key(content: u64, n: i32) -> CacheKey {
        CacheKey::new(
            content,
            &WorkUnit::Region(crate::Region {
                x: n, y: 0, w: 1, h: 1,
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
        c.insert(key(1, 0), &buf(40), 40);
        c.insert(key(1, 1), &buf(40), 40);
        c.insert(key(1, 2), &buf(40), 40);
        let s = c.stats();
        assert!(s.bytes <= 100, "bytes {} must respect budget", s.bytes);
        assert!(s.evictions >= 1);
    }

    #[test]
    fn recently_used_survives_second_chance() {
        let c = RegionCache::<TestBackend>::new(100);
        c.insert(key(1, 0), &buf(40), 40);
        c.insert(key(1, 1), &buf(40), 40);
        let _ = c.get(&key(1, 0));
        c.insert(key(1, 2), &buf(40), 40);
        assert!(c.get(&key(1, 0)).is_some(), "touched entry must survive");
        assert!(c.get(&key(1, 1)).is_none(), "cold entry must be evicted");
    }

    #[test]
    fn in_use_entry_is_never_evicted() {
        let c = RegionCache::<TestBackend>::new(10);
        let held = buf(40);
        c.insert(key(1, 0), &held, 40);
        c.insert(key(1, 1), &buf(40), 40);
        assert!(c.get(&key(1, 0)).is_some(), "in-use entry protected");
        drop(held);
        c.insert(key(1, 2), &buf(4), 4);
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
