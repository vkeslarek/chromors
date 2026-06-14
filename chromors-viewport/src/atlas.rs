use std::collections::{HashMap, HashSet};

pub const TILE: u32 = 256;
pub const PADDING_TILES: u32 = 2;

/// Pool of reusable GPU textures to avoid create/destroy churn during MIP switches.
///
/// Textures are keyed by (width, height) and kept in a LIFO stack per size class.
/// The pool tracks total bytes and evicts the largest idle entries when the budget
/// is exceeded.
pub struct TexturePool {
    entries: Vec<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    pub total_bytes: u64,
    pub budget_bytes: u64,
}

impl TexturePool {
    /// Create a pool with the given byte budget (e.g. 256 MiB).
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            entries: Vec::new(),
            total_bytes: 0,
            budget_bytes,
        }
    }

    /// Try to acquire a texture that is at least `(w, h)`.
    /// Returns the first entry whose dimensions are >= requested.
    pub fn acquire(
        &mut self,
        w: u32,
        h: u32,
    ) -> Option<(wgpu::Texture, wgpu::TextureView, u32, u32)> {
        let idx = self
            .entries
            .iter()
            .position(|(_, _, ew, eh)| *ew >= w && *eh >= h)?;
        let entry = self.entries.swap_remove(idx);
        self.total_bytes -= (entry.2 as u64) * (entry.3 as u64) * 4;
        Some(entry)
    }

    /// Return a texture to the pool for future reuse.
    pub fn release(&mut self, tex: wgpu::Texture, view: wgpu::TextureView, w: u32, h: u32) {
        let tex_bytes = w as u64 * h as u64 * 4;
        // Evict old entries if over budget
        while self.total_bytes + tex_bytes > self.budget_bytes && !self.entries.is_empty() {
            let removed = self.entries.remove(0); // evict oldest first
            self.total_bytes -= (removed.2 as u64) * (removed.3 as u64) * 4;
            // texture is dropped here, freeing VRAM
        }
        if tex_bytes <= self.budget_bytes {
            self.total_bytes += tex_bytes;
            self.entries.push((tex, view, w, h));
        }
    }
}

/// Per-tile metadata for LRU eviction.
#[derive(Clone)]
struct TileSlotEntry {
    atlas_x: u32,
    atlas_y: u32,
    last_used_frame: u64,
    version: u64,
    /// The fetcher version at the time this slot was allocated. Tiles arriving
    /// with this version are accepted even if the global fetcher version has
    /// since advanced.
    fetch_version: u64,
    has_valid_data: bool,
}

#[derive(Default)]
pub struct TileAtlas {
    pub atlas: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,
    /// Maps tile grid coord → atlas slot + LRU metadata.
    slots: HashMap<(u32, u32), TileSlotEntry>,
    /// Tiles currently being fetched (not yet ready for drawing).
    pub fetching: HashSet<(u32, u32)>,
    /// Reusable atlas slots freed by LRU eviction (LIFO stack).
    free_slots: Vec<(u32, u32)>,
    /// How many slots have ever been allocated (high watermark for new slot assignment).
    slot_hwm: u32,
    /// Cache key — when this changes, all tiles are stale and must be re-fetched.
    pub cache_key: u64,
}

impl TileAtlas {
    pub fn ensure(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        mip: u32,
        mip_w: u32,
        mip_h: u32,
    ) {
        self.ensure_with_pool(device, format, mip, mip_w, mip_h, None);
    }

    pub fn ensure_with_pool(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        mip: u32,
        mip_w: u32,
        mip_h: u32,
        pool: Option<&mut TexturePool>,
    ) {
        let ntx = mip_w.div_ceil(TILE);
        let nty = mip_h.div_ceil(TILE);
        let total = ntx * nty;
        let side = (total as f64).sqrt().ceil() as u32;
        let aw = side * TILE;
        let ah = side.div_ceil(1) * TILE;
        let max_dim = device.limits().max_texture_dimension_2d;
        let aw = aw.min(max_dim);
        let ah = ah.min(max_dim);

        if let Some((_, _, w, h)) = self.atlas
            && w >= aw
            && h >= ah
        {
            return;
        }

        // Try to acquire from pool first
        if let Some(pool) = pool {
            if let Some(entry) = pool.acquire(aw, ah) {
                tracing::debug!(target: "atlas", "reused pool atlas {}x{} for mip={mip}", entry.2, entry.3);
                // Return old texture to pool if we have one
                if let Some((old_tex, old_view, old_w, old_h)) = self.atlas.take() {
                    pool.release(old_tex, old_view, old_w, old_h);
                }
                self.atlas = Some(entry);
                return;
            }
            // Return old texture to pool before creating new one
            if let Some((old_tex, old_view, old_w, old_h)) = self.atlas.take() {
                pool.release(old_tex, old_view, old_w, old_h);
            }
        }

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("vp_atlas_mip{}", mip)),
            size: wgpu::Extent3d {
                width: aw,
                height: ah,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        tracing::debug!(target: "atlas", "created atlas mip={mip} {}x{} ({} tiles)", aw, ah, total);
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        self.atlas = Some((tex, view, aw, ah));
    }

    /// Return the atlas texture to the pool when this atlas is being reset/discarded.
    pub fn return_to_pool(&mut self, pool: &mut TexturePool) {
        if let Some((tex, view, w, h)) = self.atlas.take() {
            pool.release(tex, view, w, h);
        }
        self.slots.clear();
        self.free_slots.clear();
        self.slot_hwm = 0;
    }

    // ── Public tile_slots API (wraps internal LRU-tracked HashMap) ──

    /// Check if a tile is present in the atlas (any version — used for drawing fallback).
    pub fn has_tile(&self, key: &(u32, u32)) -> bool {
        self.slots.contains_key(key)
    }

    /// Check if a tile is present AND matches the expected version (used for dispatch).
    pub fn has_tile_version(&self, key: &(u32, u32), version: u64) -> bool {
        self.slots
            .get(key)
            .map(|e| e.version == version)
            .unwrap_or(false)
    }

    /// Get the atlas pixel offset for a tile, if present.
    pub fn get_slot(&self, key: &(u32, u32)) -> Option<(u32, u32)> {
        self.slots.get(key).map(|e| (e.atlas_x, e.atlas_y))
    }

    /// Mark a tile as used in the current frame (for LRU tracking).
    pub fn touch(&mut self, key: &(u32, u32), frame: u64) {
        if let Some(entry) = self.slots.get_mut(key) {
            entry.last_used_frame = frame;
        }
    }

    pub fn mark_valid(&mut self, key: &(u32, u32)) {
        if let Some(entry) = self.slots.get_mut(key) {
            entry.has_valid_data = true;
        }
    }

    pub fn has_valid_data(&self, key: &(u32, u32)) -> bool {
        self.slots
            .get(key)
            .map(|e| e.has_valid_data)
            .unwrap_or(false)
    }

    /// Return the fetch_version for a tile slot, if present.
    pub fn tile_fetch_version(&self, key: &(u32, u32)) -> Option<u64> {
        self.slots.get(key).map(|e| e.fetch_version)
    }

    /// Update the fetch_version for an existing slot. Called before spawning a
    /// patch fetch so that arriving patch tiles are accepted by process_fetch_responses.
    pub fn update_fetch_version(&mut self, key: &(u32, u32), fetch_version: u64) {
        if let Some(entry) = self.slots.get_mut(key) {
            entry.fetch_version = fetch_version;
        }
    }

    /// Allocate a slot for a new tile. Returns `Some((atlas_x, atlas_y))` on success,
    /// `None` if the atlas is full and no slots can be reclaimed.
    pub fn alloc_slot(
        &mut self,
        key: (u32, u32),
        version: u64,
        fetch_version: u64,
    ) -> Option<(u32, u32)> {
        let (_, _, atlas_w, atlas_h) = self.atlas.as_ref()?;
        let cols = *atlas_w / TILE;

        // Prefer a recycled slot
        if let Some(pos) = self.free_slots.pop() {
            self.slots.insert(
                key,
                TileSlotEntry {
                    atlas_x: pos.0,
                    atlas_y: pos.1,
                    last_used_frame: 0,
                    version,
                    fetch_version,
                    has_valid_data: false,
                },
            );
            self.fetching.insert(key);
            return Some(pos);
        }

        // Allocate a fresh slot
        let idx = self.slot_hwm;
        let sx = (idx % cols) * TILE;
        let sy = (idx / cols) * TILE;
        if sy + TILE <= *atlas_h {
            self.slot_hwm += 1;
            self.slots.insert(
                key,
                TileSlotEntry {
                    atlas_x: sx,
                    atlas_y: sy,
                    last_used_frame: 0,
                    version,
                    fetch_version,
                    has_valid_data: false,
                },
            );
            self.fetching.insert(key);
            return Some((sx, sy));
        }

        None
    }

    /// Allocate or recycle a slot for a tile. If a slot already exists for this
    /// key (stale version), reuse its atlas position instead of allocating a new one.
    pub fn alloc_or_recycle_slot(
        &mut self,
        key: (u32, u32),
        version: u64,
        fetch_version: u64,
    ) -> Option<(u32, u32)> {
        if let Some(entry) = self.slots.get_mut(&key) {
            entry.version = version;
            entry.fetch_version = fetch_version;
            self.fetching.insert(key);
            return Some((entry.atlas_x, entry.atlas_y));
        }
        self.alloc_slot(key, version, fetch_version)
    }

    /// Evict tiles not used for `max_age` frames, freeing their atlas slots for reuse.
    /// Returns the number of evicted tiles.
    pub fn evict_stale(&mut self, current_frame: u64, max_age: u64) -> usize {
        let threshold = current_frame.saturating_sub(max_age);
        let evicted: Vec<(u32, u32)> = self
            .slots
            .iter()
            .filter(|(k, e)| e.last_used_frame < threshold && !self.fetching.contains(k))
            .map(|(k, _)| *k)
            .collect();
        let count = evicted.len();
        for key in evicted {
            if let Some(entry) = self.slots.remove(&key) {
                self.free_slots.push((entry.atlas_x, entry.atlas_y));
            }
        }
        count
    }

    /// Total number of occupied slots (including tiles being fetched).
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    // ── Backward compatibility shim (used by renderer draw loop) ──

    /// Legacy accessor — returns reference to the tile_slots for iteration.
    /// Prefer `get_slot` / `has_tile` for new code.
    pub fn tile_slots_iter(&self) -> impl Iterator<Item = (&(u32, u32), (u32, u32))> {
        self.slots.iter().map(|(k, e)| (k, (e.atlas_x, e.atlas_y)))
    }
}
