use std::collections::HashMap;

use poc::backend::gpu::GpuBackend;
use poc::data::image::Image2D as GenImage;

use crate::atlas::{PADDING_TILES, TILE, TileAtlas};
use crate::camera::Camera;
use crate::fetcher::TileFetcher;
use crate::rect::Rect;
use crate::source::{ImageViewportSource, MippedViewportSource, ViewportLayerSource};

#[derive(Clone, Copy, Debug)]
pub struct LayerTransform {
    pub x: f32,
    pub y: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub opacity: f32,
}

impl Default for LayerTransform {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            opacity: 1.0,
        }
    }
}

pub struct ImageLayer {
    pub id: u64,
    /// Pluggable source for this layer's mip slots (see [`ViewportLayerSource`]).
    pub source: Box<dyn ViewportLayerSource>,
    pub base_w: u32,
    pub base_h: u32,
    pub transform: LayerTransform,
    pub mip_states: HashMap<u32, TileAtlas>,
    pub front_states: Option<HashMap<u32, TileAtlas>>,
    pub target_mip: u32,
    pub layer_bufs: Option<Vec<wgpu::Buffer>>,
    /// Optional overlay source whose plugged mip slots take precedence over
    /// [`source`](Self::source) (used for live preview). Expressed as a
    /// [`MippedViewportSource`] adapter rather than an ad-hoc override map.
    pub override_source: Option<MippedViewportSource>,
    /// Monotonic version counter — incremented on every image data change.
    /// Used as part of the cache key to invalidate stale tiles.
    pub image_version: u64,
}

impl ImageLayer {
    pub fn new(id: u64, image: GenImage<GpuBackend>) -> Self {
        Self::with_source(id, Box::new(ImageViewportSource::new(image)))
    }

    /// Build a layer from any [`ViewportLayerSource`].
    pub fn with_source(id: u64, source: Box<dyn ViewportLayerSource>) -> Self {
        let (base_w, base_h) = source.base_size();
        Self {
            id,
            source,
            base_w,
            base_h,
            transform: LayerTransform::default(),
            mip_states: HashMap::new(),
            front_states: None,
            target_mip: 0,
            layer_bufs: None,
            override_source: None,
            image_version: 0,
        }
    }

    /// Swap the layer's source, refreshing the cached base dimensions.
    pub fn set_source(&mut self, source: Box<dyn ViewportLayerSource>) {
        let (base_w, base_h) = source.base_size();
        self.source = source;
        self.base_w = base_w;
        self.base_h = base_h;
    }

    /// Image for a mip slot: the override source's plugged slot if present,
    /// otherwise the base source.
    pub fn mip_image(&mut self, mip: u32) -> Option<GenImage<GpuBackend>> {
        if let Some(img) = self.override_source.as_ref().and_then(|o| o.plugged(mip)) {
            return Some(img);
        }
        self.source.slot_image(mip)
    }

    pub fn fallback_mip(&self) -> u32 {
        let mut m = 0;
        while (self.base_w >> m).max(1) > TILE || (self.base_h >> m).max(1) > TILE {
            m += 1;
        }
        m
    }

    /// Warm the source up to `mip` (the source builds + caches the chain).
    pub fn ensure_mip(&mut self, mip: u32) {
        let _ = self.source.slot_image(mip);
    }

    pub fn visible_tiles(
        &self,
        camera: &Camera,
        mip: u32,
        mip_w: u32,
        mip_h: u32,
    ) -> Vec<(u32, u32)> {
        let scale = (1u32 << mip) as f32;
        let ts = TILE as f32;

        let time_horizon = 0.5; // Half a second lookahead
        let pred_pan_x = camera.pan_x - (camera.velocity_x / camera.zoom) * time_horizon;
        let pred_pan_y = camera.pan_y - (camera.velocity_y / camera.zoom) * time_horizon;

        let local_pan_x = (camera.pan_x - self.transform.x) / self.transform.scale_x;
        let local_pan_y = (camera.pan_y - self.transform.y) / self.transform.scale_y;
        let pred_local_pan_x = (pred_pan_x - self.transform.x) / self.transform.scale_x;
        let pred_local_pan_y = (pred_pan_y - self.transform.y) / self.transform.scale_y;

        let local_vp_w = (camera.vp_w / camera.zoom) / self.transform.scale_x;
        let local_vp_h = (camera.vp_h / camera.zoom) / self.transform.scale_y;

        let min_x = local_pan_x.min(pred_local_pan_x);
        let max_x = (local_pan_x + local_vp_w).max(pred_local_pan_x + local_vp_w);
        let min_y = local_pan_y.min(pred_local_pan_y);
        let max_y = (local_pan_y + local_vp_h).max(pred_local_pan_y + local_vp_h);

        let x0 = ((min_x.max(0.0) / scale / ts).floor() as u32).saturating_sub(PADDING_TILES);
        let y0 = ((min_y.max(0.0) / scale / ts).floor() as u32).saturating_sub(PADDING_TILES);
        let x1 = ((max_x.min(self.base_w as f32) / scale / ts).ceil() as u32 + PADDING_TILES)
            .min(mip_w.div_ceil(TILE));
        let y1 = ((max_y.min(self.base_h as f32) / scale / ts).ceil() as u32 + PADDING_TILES)
            .min(mip_h.div_ceil(TILE));
        let mut tiles = Vec::new();
        for ty in y0..y1 {
            for tx in x0..x1 {
                tiles.push((tx, ty));
            }
        }
        tiles
    }

    pub fn swapchain_push(&mut self) {
        if self.front_states.is_none() {
            // Move current mip_states to front buffer to keep them on screen
            self.front_states = Some(std::mem::take(&mut self.mip_states));
        } else {
            // If already swapping, just clear the backbuffer so it fetches fresh
            for state in self.mip_states.values_mut() {
                state.fetching.clear();
            }
            self.mip_states.clear();
        }
    }

    pub fn swapchain_commit(&mut self, pool: &mut crate::atlas::TexturePool) {
        if let Some(mut front) = self.front_states.take() {
            for state in front.values_mut() {
                state.return_to_pool(pool);
            }
        }
    }

    pub fn invalidate_region(&mut self, rect: Rect, fetcher: &TileFetcher) {
        let mip = self.target_mip;
        let scale = 1u32 << mip;

        let mip_x = rect.x / scale as i32;
        let mip_y = rect.y / scale as i32;
        let mip_w = rect.width / scale as i32;
        let mip_h = rect.height / scale as i32;
        if mip_w <= 0 || mip_h <= 0 {
            return;
        }

        let tx0 = (mip_x / TILE as i32).max(0) as u32;
        let ty0 = (mip_y / TILE as i32).max(0) as u32;
        let tx1 = ((mip_x + mip_w + TILE as i32 - 1) / (TILE as i32)).max(0) as u32;
        let ty1 = ((mip_y + mip_h + TILE as i32 - 1) / (TILE as i32)).max(0) as u32;

        let mut patches = Vec::new();
        if let Some(state) = self.mip_states.get(&mip) {
            for ty in ty0..ty1 {
                for tx in tx0..tx1 {
                    if state.has_tile(&(tx, ty)) && !state.fetching.contains(&(tx, ty)) {
                        let tile_x = (tx * TILE) as i32;
                        let tile_y = (ty * TILE) as i32;
                        let ix = mip_x.max(tile_x);
                        let iy = mip_y.max(tile_y);
                        let ix2 = (mip_x + mip_w).min(tile_x + TILE as i32);
                        let iy2 = (mip_y + mip_h).min(tile_y + TILE as i32);

                        if ix < ix2 && iy < iy2 {
                            let pw = (ix2 - ix) as u32;
                            let ph = (iy2 - iy) as u32;
                            let px = (ix - tile_x) as u32;
                            let py = (iy - tile_y) as u32;
                            patches.push((tx, ty, px, py, pw, ph));
                        }
                    }
                }
            }
        }

        if !patches.is_empty() {
            // Stamp each slot with the current fetcher version so process_fetch_responses
            // accepts arriving patch tiles. Without this, slots keep fetch_version from
            // initial allocation and reject all patch data.
            let fv = fetcher.version();
            if let Some(state) = self.mip_states.get_mut(&mip) {
                for &(tx, ty, ..) in &patches {
                    state.update_fetch_version(&(tx, ty), fv);
                }
            }

            // Use mip-level rect so target.pull returns a mip-sized buffer. Passing base
            // coords produces a full-res buffer and emit_tile reads wrong pixel offsets.
            let mip_rect = Rect::new(mip_x, mip_y, mip_w, mip_h);
            if let Some(mip_img) = self.mip_image(mip) {
                fetcher.spawn_patch_fetch(self.id, mip, mip_img, mip_rect, patches, 0);
            }
        }
    }
}
