//! Unified tiered tile cache: **VRAM → RAM → Disk**, content-addressed, with
//! CLOCK eviction and a pin generation for the interactive working set.
//!
//! Replaces the per-graph `RegionCache` `HashMap`. One instance lives on
//! [`super::context::GpuContext`], shared by every graph on that device.
//!
//! ## Tiers
//! * **VRAM** — live image values (`GraphValue::Image`) resident in GPU memory.
//! * **RAM**  — live raw values (`GraphValue::Raw`, e.g. histograms) *and* images
//!   spilled to host bytes (downloaded, kept for cheap re-upload).
//! * **Disk** — image or raw bytes written to a scratch file.
//!
//! Each tier has its own byte budget and CLOCK ring. When a tier exceeds budget,
//! the clock hand sweeps it: an entry whose recency bit is set gets a second
//! chance; an entry still referenced elsewhere (`Arc::strong_count > 1`) or
//! pinned to the current generation is protected; otherwise the victim is
//! **demoted one tier down** (VRAM→RAM→Disk), and dropped entirely off Disk.
//! Eviction cascades downward only (never upward).
//!
//! ## Eviction policy — CLOCK (second-chance)
//! A cache *hit* sets a recency bit (one write, O(1) lookup). The sweep clears
//! bits and reclaims the first already-clear, unpinned, unreferenced entry.
//!
//! ## Concurrency (v1)
//! The cache is guarded by a single outer `Mutex` (see `RegionCache`), the
//! darktable model for "a handful of threads". GPU/disk IO during promote/demote
//! currently happens under that lock; sharding + IO-outside-lock are a documented
//! perf follow-up. Correctness first.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use crate::geometry::Rect;
use crate::pixel::PixelMeta;

use super::buffer::ImageBuffer;
use super::context::GpuContext;
use super::graph::RegionKey;
use super::value::{GraphValue, ValueKind};

const MIB: u64 = 1024 * 1024;
const DEFAULT_GPU_BUDGET: u64 = 256 * MIB;
const DEFAULT_RAM_BUDGET: u64 = 1024 * MIB;
const DEFAULT_DISK_BUDGET: u64 = 4096 * MIB;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tier {
    Vram,
    Ram,
    Disk,
}

/// How to rebuild a live [`GraphValue`] from spilled raw bytes.
#[derive(Clone)]
enum Payload {
    Image {
        width: u32,
        height: u32,
        meta: PixelMeta,
        source_rect: Rect,
    },
    Raw {
        kind: ValueKind,
        source_rect: Rect,
    },
}

impl Payload {
    fn is_image(&self) -> bool {
        matches!(self, Payload::Image { .. })
    }
}

enum Res {
    /// Live, in-memory value — an image in VRAM, or a raw value's CPU bytes.
    Live(Arc<GraphValue>),
    /// Image/raw bytes spilled to host RAM, plus how to rebuild the value.
    Ram {
        bytes: Arc<Vec<u8>>,
        payload: Payload,
    },
    /// Image/raw bytes spilled to a disk file, plus how to rebuild the value.
    Disk { path: PathBuf, payload: Payload },
}

struct Entry {
    res: Res,
    /// Logical byte length — constant across tiers (tightly packed), used for
    /// budget accounting so counters never drift.
    len: u64,
    /// True for image values (live → VRAM tier). Raw values live in the RAM tier.
    is_image: bool,
    /// CLOCK recency bit.
    used: bool,
    /// Pin generation; protected from eviction iff `pin == cache.generation`.
    pin: u64,
}

impl Entry {
    fn tier(&self) -> Tier {
        match &self.res {
            Res::Live(_) => {
                if self.is_image {
                    Tier::Vram
                } else {
                    Tier::Ram
                }
            }
            Res::Ram { .. } => Tier::Ram,
            Res::Disk { .. } => Tier::Disk,
        }
    }
}

/// A tiered, content-addressed tile cache. See module docs.
pub struct TieredCache {
    map: HashMap<RegionKey, Entry>,
    vram_ring: VecDeque<RegionKey>,
    ram_ring: VecDeque<RegionKey>,
    disk_ring: VecDeque<RegionKey>,
    vram_bytes: u64,
    ram_bytes: u64,
    disk_bytes: u64,
    vram_budget: u64,
    ram_budget: u64,
    disk_budget: u64,
    generation: u64,
    disk_dir: PathBuf,
    next_file: u64,
    ctx: Weak<GpuContext>,

    // ── Diagnostics ──────────────────────────────────────────────────────
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub spills_to_ram: u64,
    pub spills_to_disk: u64,
    pub promotions: u64,
}

impl TieredCache {
    pub fn new() -> Self {
        Self::with_budgets(DEFAULT_GPU_BUDGET, DEFAULT_RAM_BUDGET, DEFAULT_DISK_BUDGET)
    }

    /// Construct with a given VRAM budget; RAM/Disk use defaults.
    pub fn with_budget(gpu: u64) -> Self {
        Self::with_budgets(gpu, DEFAULT_RAM_BUDGET, DEFAULT_DISK_BUDGET)
    }

    pub fn with_budgets(vram_budget: u64, ram_budget: u64, disk_budget: u64) -> Self {
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let disk_dir =
            std::env::temp_dir().join(format!("pixors_tilecache_{}_{n}", std::process::id()));
        let _ = std::fs::create_dir_all(&disk_dir);
        Self {
            map: HashMap::new(),
            vram_ring: VecDeque::new(),
            ram_ring: VecDeque::new(),
            disk_ring: VecDeque::new(),
            vram_bytes: 0,
            ram_bytes: 0,
            disk_bytes: 0,
            vram_budget,
            ram_budget,
            disk_budget,
            generation: 1,
            disk_dir,
            next_file: 0,
            ctx: Weak::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
            spills_to_ram: 0,
            spills_to_disk: 0,
            promotions: 0,
        }
    }

    /// Bind the owning context so the cache can upload/download during
    /// promote/demote. Call once, after the `Arc<GpuContext>` is built.
    pub fn bind_ctx(&mut self, ctx: Weak<GpuContext>) {
        self.ctx = ctx;
    }

    /// Set the VRAM budget (RAM/Disk unchanged) and immediately enforce it.
    pub fn set_budget(&mut self, vram: u64) {
        self.vram_budget = vram;
        self.evict(Tier::Vram);
    }

    // ── Lookup / insert ──────────────────────────────────────────────────

    /// Look up a key, promoting it back into memory if it was spilled. Sets the
    /// recency bit. Returns `None` on a miss (or if a spilled image cannot be
    /// re-uploaded because no GPU context is bound).
    pub fn get(&mut self, key: &RegionKey) -> Option<Arc<GraphValue>> {
        match self.map.get_mut(key) {
            None => {
                self.misses += 1;
                None
            }
            Some(e) => {
                e.used = true;
                if let Res::Live(v) = &e.res {
                    self.hits += 1;
                    Some(v.clone())
                } else {
                    self.hits += 1;
                    self.promote(*key)
                }
            }
        }
    }

    /// Insert a freshly materialised value. Images go to the VRAM tier, raw
    /// values to the RAM tier. The entry starts *cold* (ring position protects
    /// it from premature eviction); re-inserting an existing key replaces it.
    pub fn insert(&mut self, key: RegionKey, value: Arc<GraphValue>) {
        let (len, is_image) = match &*value {
            GraphValue::Image { buffer, .. } => (buffer.total_bytes(), true),
            GraphValue::Raw { bytes, .. } => (bytes.len() as u64, false),
        };
        // Clean any prior entry so byte accounting stays exact.
        self.remove(&key);
        let tier = if is_image { Tier::Vram } else { Tier::Ram };
        self.map.insert(
            key,
            Entry {
                res: Res::Live(value),
                len,
                is_image,
                used: false,
                pin: 0,
            },
        );
        self.add_bytes(tier, len);
        self.ring(tier).push_back(key);
        self.evict(tier);
    }

    /// Remove an entry (any tier). Returns its live value if it was resident.
    /// Deletes the backing disk file if spilled. Ring slots are skipped lazily.
    pub fn remove(&mut self, key: &RegionKey) -> Option<Arc<GraphValue>> {
        let e = self.map.remove(key)?;
        let tier = e.tier();
        self.sub_bytes(tier, e.len);
        match e.res {
            Res::Live(v) => Some(v),
            Res::Ram { .. } => None,
            Res::Disk { path, .. } => {
                let _ = std::fs::remove_file(path);
                None
            }
        }
    }

    /// Drop every entry whose key does not satisfy `keep`.
    pub fn retain(&mut self, keep: impl Fn(&RegionKey) -> bool) {
        let doomed: Vec<RegionKey> = self.map.keys().filter(|k| !keep(k)).copied().collect();
        for k in doomed {
            self.remove(&k);
        }
    }

    /// Drop every entry whose content hash (key's leading field) matches
    /// `content`, regardless of lod/rect. Used to invalidate a source/asset.
    pub fn invalidate_content(&mut self, content: u64) {
        let doomed: Vec<RegionKey> = self
            .map
            .keys()
            .filter(|k| k.0 == content)
            .copied()
            .collect();
        for k in doomed {
            self.remove(&k);
        }
    }

    // ── Pin generation ───────────────────────────────────────────────────

    /// Advance the pin generation. Entries pinned to the old generation are no
    /// longer protected. Call on commit / tool switch (ends a preview cycle).
    pub fn bump_generation(&mut self) {
        self.generation += 1;
    }

    /// Pin a key to the current generation so it is protected from eviction
    /// until the next [`Self::bump_generation`].
    pub fn touch_pin(&mut self, key: &RegionKey) {
        let cur_gen = self.generation;
        if let Some(e) = self.map.get_mut(key) {
            e.pin = cur_gen;
        }
    }

    // ── Introspection ────────────────────────────────────────────────────

    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
    /// VRAM bytes currently resident (kept name for back-compat).
    pub fn resident_bytes(&self) -> u64 {
        self.vram_bytes
    }
    pub fn vram_bytes(&self) -> u64 {
        self.vram_bytes
    }
    pub fn ram_bytes(&self) -> u64 {
        self.ram_bytes
    }
    pub fn disk_bytes(&self) -> u64 {
        self.disk_bytes
    }
    pub fn budget(&self) -> u64 {
        self.vram_budget
    }

    // ── Tier accounting helpers ──────────────────────────────────────────

    fn tier_bytes(&self, t: Tier) -> u64 {
        match t {
            Tier::Vram => self.vram_bytes,
            Tier::Ram => self.ram_bytes,
            Tier::Disk => self.disk_bytes,
        }
    }
    fn tier_budget(&self, t: Tier) -> u64 {
        match t {
            Tier::Vram => self.vram_budget,
            Tier::Ram => self.ram_budget,
            Tier::Disk => self.disk_budget,
        }
    }
    fn add_bytes(&mut self, t: Tier, n: u64) {
        match t {
            Tier::Vram => self.vram_bytes += n,
            Tier::Ram => self.ram_bytes += n,
            Tier::Disk => self.disk_bytes += n,
        }
    }
    fn sub_bytes(&mut self, t: Tier, n: u64) {
        match t {
            Tier::Vram => self.vram_bytes -= n.min(self.vram_bytes),
            Tier::Ram => self.ram_bytes -= n.min(self.ram_bytes),
            Tier::Disk => self.disk_bytes -= n.min(self.disk_bytes),
        }
    }
    fn ring(&mut self, t: Tier) -> &mut VecDeque<RegionKey> {
        match t {
            Tier::Vram => &mut self.vram_ring,
            Tier::Ram => &mut self.ram_ring,
            Tier::Disk => &mut self.disk_ring,
        }
    }

    fn alloc_disk_path(&mut self) -> PathBuf {
        let id = self.next_file;
        self.next_file += 1;
        self.disk_dir.join(format!("tile_{id:016x}.blob"))
    }

    fn rebuild(&self, bytes: &[u8], payload: &Payload) -> Option<Arc<GraphValue>> {
        match payload {
            Payload::Raw { kind, source_rect } => Some(Arc::new(GraphValue::raw(
                bytes.to_vec(),
                kind.clone(),
                *source_rect,
            ))),
            Payload::Image {
                width,
                height,
                meta,
                source_rect,
            } => {
                let ctx = self.ctx.upgrade()?;
                let img = ImageBuffer::upload(bytes, *width, *height, *meta, &ctx).ok()?;
                Some(Arc::new(GraphValue::image(img, *source_rect)))
            }
        }
    }

    // ── Promotion (spilled → live) ───────────────────────────────────────

    fn promote(&mut self, key: RegionKey) -> Option<Arc<GraphValue>> {
        // Snapshot the spill, then release the borrow before any IO.
        let (bytes, payload, from_tier, disk_path) = {
            let e = self.map.get(&key)?;
            match &e.res {
                Res::Live(v) => return Some(v.clone()), // raced — already promoted
                Res::Ram { bytes, payload } => {
                    (Arc::clone(bytes), payload.clone(), Tier::Ram, None)
                }
                Res::Disk { path, payload } => {
                    let data = std::fs::read(path).ok()?;
                    (
                        Arc::new(data),
                        payload.clone(),
                        Tier::Disk,
                        Some(path.clone()),
                    )
                }
            }
        };

        let value = match self.rebuild(&bytes, &payload) {
            Some(v) => v,
            None => {
                // Cannot rebuild (image with no GPU context) — drop the entry.
                self.drop_entry(key);
                return None;
            }
        };

        let to_tier = if payload.is_image() {
            Tier::Vram
        } else {
            Tier::Ram
        };
        let len = self
            .map
            .get(&key)
            .map(|e| e.len)
            .unwrap_or(bytes.len() as u64);

        if let Some(e) = self.map.get_mut(&key) {
            e.res = Res::Live(value.clone());
            e.used = true;
        } else {
            // Entry vanished mid-flight; still hand the rebuilt value back.
            return Some(value);
        }
        if let Some(p) = disk_path {
            let _ = std::fs::remove_file(p);
        }
        self.sub_bytes(from_tier, len);
        self.add_bytes(to_tier, len);
        self.ring(to_tier).push_back(key);
        self.promotions += 1;
        self.evict(to_tier);
        Some(value)
    }

    // ── Eviction (CLOCK per tier, demote downward) ───────────────────────

    fn evict(&mut self, tier: Tier) {
        let mut skips = 0usize;
        while self.tier_bytes(tier) > self.tier_budget(tier) {
            let Some(key) = self.ring(tier).pop_front() else {
                break; // ring empty
            };
            let cur_gen = self.generation;

            let Some(e) = self.map.get_mut(&key) else {
                continue; // stale ring slot — entry already gone
            };
            if e.tier() != tier {
                continue; // entry moved tiers since queued — stale slot
            }
            // Protected: pinned to the current generation.
            if e.pin == cur_gen {
                e.used = false;
                self.ring(tier).push_back(key);
                skips += 1;
                if skips > self.ring(tier).len() {
                    break;
                }
                continue;
            }
            // In use by a live holder → never reclaim; second chance.
            if let Res::Live(v) = &e.res
                && Arc::strong_count(v) > 1
            {
                e.used = false;
                self.ring(tier).push_back(key);
                skips += 1;
                if skips > self.ring(tier).len() {
                    break;
                }
                continue;
            }
            // Recency second chance.
            if e.used {
                e.used = false;
                self.ring(tier).push_back(key);
                skips += 1;
                if skips > self.ring(tier).len() {
                    break;
                }
                continue;
            }

            // Victim: demote one tier down (or drop off disk).
            skips = 0;
            match tier {
                Tier::Vram => self.demote_vram_to_ram(key),
                Tier::Ram => self.demote_ram_to_disk(key),
                Tier::Disk => self.drop_entry(key),
            }
        }
    }

    fn demote_vram_to_ram(&mut self, key: RegionKey) {
        // Clone the live value out, drop the map borrow, then do the download.
        let (v, len) = match self.map.get(&key) {
            Some(e) => match &e.res {
                Res::Live(v) => (v.clone(), e.len),
                _ => return,
            },
            None => return,
        };
        let GraphValue::Image {
            buffer,
            source_rect,
            ..
        } = &*v
        else {
            return; // VRAM tier only holds images
        };
        let ctx = match self.ctx.upgrade() {
            Some(c) => c,
            None => {
                self.drop_entry(key);
                return;
            }
        };
        let bytes = match buffer.read_to_cpu(&ctx) {
            Ok(b) => b,
            Err(_) => {
                self.drop_entry(key);
                return;
            }
        };
        let payload = Payload::Image {
            width: buffer.width,
            height: buffer.height,
            meta: buffer.meta,
            source_rect: *source_rect,
        };
        if let Some(e) = self.map.get_mut(&key) {
            e.res = Res::Ram {
                bytes: Arc::new(bytes),
                payload,
            };
            e.used = false;
        } else {
            return;
        }
        self.sub_bytes(Tier::Vram, len);
        self.add_bytes(Tier::Ram, len);
        self.ram_ring.push_back(key);
        self.spills_to_ram += 1;
        self.evict(Tier::Ram);
    }

    fn demote_ram_to_disk(&mut self, key: RegionKey) {
        let (bytes, payload, len) = match self.map.get(&key) {
            Some(e) => match &e.res {
                Res::Ram { bytes, payload } => (Arc::clone(bytes), payload.clone(), e.len),
                Res::Live(v) => match &**v {
                    GraphValue::Raw {
                        bytes,
                        kind,
                        source_rect,
                    } => (
                        Arc::new(bytes.clone()),
                        Payload::Raw {
                            kind: kind.clone(),
                            source_rect: *source_rect,
                        },
                        e.len,
                    ),
                    GraphValue::Image { .. } => return, // image in RAM tier shouldn't be Live
                },
                Res::Disk { .. } => return,
            },
            None => return,
        };
        let path = self.alloc_disk_path();
        if std::fs::write(&path, &*bytes).is_err() {
            return; // keep it in RAM; accept transient overshoot
        }
        if let Some(e) = self.map.get_mut(&key) {
            e.res = Res::Disk { path, payload };
            e.used = false;
        } else {
            let _ = std::fs::remove_file(&path);
            return;
        }
        self.sub_bytes(Tier::Ram, len);
        self.add_bytes(Tier::Disk, len);
        self.disk_ring.push_back(key);
        self.spills_to_disk += 1;
        self.evict(Tier::Disk);
    }

    fn drop_entry(&mut self, key: RegionKey) {
        if let Some(e) = self.map.remove(&key) {
            let tier = e.tier();
            self.sub_bytes(tier, e.len);
            if let Res::Disk { path, .. } = &e.res {
                let _ = std::fs::remove_file(path);
            }
            self.evictions += 1;
        }
    }
}

impl Drop for TieredCache {
    fn drop(&mut self) {
        // Best-effort cleanup of the scratch directory.
        let _ = std::fs::remove_dir_all(&self.disk_dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::gpu::value::ValueKind;
    use crate::geometry::Rect;

    // Raw values exercise the full tier state machine without a GPU:
    // insert → RAM(live) → demote to Disk → promote back, with byte-exact
    // round-trips. The image path uses identical transitions with GPU
    // upload/download swapped in (validated on hardware).

    fn raw(n: usize, fill: u8) -> Arc<GraphValue> {
        Arc::new(GraphValue::raw(
            vec![fill; n],
            ValueKind::Scalar,
            Rect::new(0, 0, 1, 1),
        ))
    }
    fn key(n: u64) -> RegionKey {
        (n, 0, 0, 0, 0)
    }

    #[test]
    fn raw_inserts_into_ram_tier() {
        let mut c = TieredCache::with_budgets(0, 1000, 1000);
        c.insert(key(1), raw(40, 1));
        assert_eq!(c.ram_bytes(), 40);
        assert_eq!(c.vram_bytes(), 0);
        assert!(c.get(&key(1)).is_some());
    }

    #[test]
    fn ram_over_budget_demotes_to_disk_then_promotes_back() {
        let mut c = TieredCache::with_budgets(0, 100, 10_000);
        c.insert(key(1), raw(40, 0xAA));
        c.insert(key(2), raw(40, 0xBB));
        c.insert(key(3), raw(40, 0xCC)); // 120 > 100 RAM budget → one spills to disk

        assert!(c.disk_bytes() > 0, "a cold entry must have spilled to disk");
        assert!(c.ram_bytes() <= 100);

        // Whatever spilled, every entry is still retrievable, byte-exact.
        for (k, fill) in [(1u64, 0xAA), (2, 0xBB), (3, 0xCC)] {
            let v = c
                .get(&key(k))
                .expect("entry must survive (in RAM or on disk)");
            match &*v {
                GraphValue::Raw { bytes, .. } => {
                    assert_eq!(bytes.len(), 40);
                    assert!(bytes.iter().all(|&b| b == fill), "round-trip must be exact");
                }
                _ => panic!("expected raw"),
            }
        }
    }

    #[test]
    fn recently_used_survives_second_chance() {
        let mut c = TieredCache::with_budgets(0, 100, 10_000);
        c.insert(key(1), raw(40, 1));
        c.insert(key(2), raw(40, 2));
        let _ = c.get(&key(1)); // touch 1 → hot
        c.insert(key(3), raw(40, 3)); // evicts the coldest

        // 1 was touched → stays live in RAM; 2 is the victim → spilled to disk.
        assert!(matches!(&c.map.get(&key(1)).unwrap().res, Res::Live(_)));
        assert!(matches!(&c.map.get(&key(2)).unwrap().res, Res::Disk { .. }));
    }

    #[test]
    fn pinned_entry_is_protected_until_generation_bump() {
        let mut c = TieredCache::with_budgets(0, 100, 10_000);
        c.insert(key(1), raw(80, 1));
        c.touch_pin(&key(1));
        c.insert(key(2), raw(80, 2)); // 160 > 100 but key(1) pinned

        assert!(
            matches!(&c.map.get(&key(1)).unwrap().res, Res::Live(_)),
            "pinned entry must not be demoted"
        );

        c.bump_generation();
        c.insert(key(3), raw(80, 3)); // now key(1) is unprotected
        assert!(
            matches!(&c.map.get(&key(1)).unwrap().res, Res::Disk { .. }),
            "after bump, the formerly-pinned entry can spill"
        );
    }

    #[test]
    fn in_use_value_is_never_demoted() {
        let mut c = TieredCache::with_budgets(0, 10, 10_000);
        let held = raw(40, 7);
        c.insert(key(1), held.clone()); // strong_count >= 2 while `held` lives
        assert!(matches!(&c.map.get(&key(1)).unwrap().res, Res::Live(_)));
        drop(held);
        c.insert(key(2), raw(4, 0)); // sweep now frees key(1)
        assert!(matches!(&c.map.get(&key(1)).unwrap().res, Res::Disk { .. }));
    }

    #[test]
    fn disk_over_budget_drops_coldest() {
        // Tiny RAM and disk budgets: entries spill to disk then get dropped.
        let mut c = TieredCache::with_budgets(0, 40, 40);
        c.insert(key(1), raw(40, 1));
        c.insert(key(2), raw(40, 2));
        c.insert(key(3), raw(40, 3));
        // Total demand 120 ≫ 40 RAM + 40 disk → at least one fully dropped.
        let present = [1u64, 2, 3]
            .iter()
            .filter(|&&k| c.get(&key(k)).is_some())
            .count();
        assert!(present < 3, "disk budget must force a full drop");
        assert!(c.evictions > 0);
    }

    #[test]
    fn remove_and_retain_keep_accounting_consistent() {
        let mut c = TieredCache::with_budgets(0, 10_000, 10_000);
        c.insert(key(1), raw(40, 1));
        c.insert(key(2), raw(40, 2));
        assert_eq!(c.ram_bytes(), 80);
        c.remove(&key(1));
        assert_eq!(c.ram_bytes(), 40);
        c.retain(|k| *k == key(2));
        assert_eq!(c.ram_bytes(), 40);
        c.retain(|_| false);
        assert_eq!(c.ram_bytes(), 0);
        assert!(c.is_empty());
    }
}
