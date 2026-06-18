//! <sources>
//! This module provides pluggable sources for the viewport renderer's mip
//! slots. The renderer asks the source for the image backing each slot; the
//! source decides how those images are produced.
//!
//! - [`ImageViewportSource`] — one base image, same graph for every slot. The
//!   mip level is a *pull demand* dimension (`Region.lod`): the fetcher pulls
//!   each tile at `Lod(slot)` and the source shrink-on-loads to that level. No
//!   GPU `Shrink` op, no full-res processing for coarse mips.
//! - [`MippedViewportSource`] — the caller plugs a specific image into each
//!   slot from outside (full external control, no in-viewport shrinking).
//! - [`VectorGraphicsViewportSource`] — a Vello-drawn vector layer (stub).

use chromors::backend::gpu::GpuBackend;
use chromors::data::image::Image2D as GenImage;

/// A source of GPU images for a viewport layer's mip slots.
///
/// The renderer asks the source for the image backing each mip slot instead of
/// assuming a single base image it must shrink itself. Slot 0 is full
/// resolution; higher slots are progressively half-sized.
pub trait ViewportLayerSource: Send {
    /// Full-resolution (slot 0) dimensions in pixels.
    fn base_size(&self) -> (u32, u32);

    /// How many mip slots this source exposes (>= 1). The caller can query this
    /// to know how many slots are available to plug / fetch.
    fn slot_count(&self) -> u32;

    /// GPU image backing mip `slot`, built/materialized on demand. Returns
    /// `None` if the slot can't be produced.
    fn slot_image(&mut self, slot: u32) -> Option<GenImage<GpuBackend>>;

    /// Already-built slot images (no building), indexed by mip. Used for cache
    /// GC / introspection without forcing lazy slots to materialize.
    fn built_slots(&self) -> Vec<GenImage<GpuBackend>>;
}

/// Number of mip slots from full-res down to ~1px for a `w x h` image.
fn mip_slots_for(w: u32, h: u32) -> u32 {
    let m = w.max(h).max(1);
    (u32::BITS - m.leading_zeros()).max(1)
}

// ── ImageViewportSource ──────────────────────────────────────────────────────

/// One base image, same graph for every mip slot. Downsampling is a pull-demand
/// dimension (`Region.lod`) honored by the source via shrink-on-load — coarse
/// mips never decode or process full resolution.
pub struct ImageViewportSource {
    base: GenImage<GpuBackend>,
}

impl ImageViewportSource {
    pub fn new(image: GenImage<GpuBackend>) -> Self {
        Self { base: image }
    }

    pub fn set_base(&mut self, image: GenImage<GpuBackend>) {
        self.base = image;
    }
}

impl ViewportLayerSource for ImageViewportSource {
    fn base_size(&self) -> (u32, u32) {
        (self.base.width() as u32, self.base.height() as u32)
    }

    fn slot_count(&self) -> u32 {
        let (w, h) = self.base_size();
        mip_slots_for(w, h)
    }

    fn slot_image(&mut self, _slot: u32) -> Option<GenImage<GpuBackend>> {
        // Same graph for every slot — the mip level is a *pull demand*
        // dimension (`Region.lod`), honored by the source via shrink-on-load,
        // not a GPU `Shrink` op. The fetcher pulls each tile at `Lod(slot)`.
        Some(self.base.clone())
    }

    fn built_slots(&self) -> Vec<GenImage<GpuBackend>> {
        vec![self.base.clone()]
    }
}

// ── MippedViewportSource ─────────────────────────────────────────────────────

/// The caller plugs a specific image into each mip slot from outside. The
/// viewport does no shrinking — full external control over every mip level.
pub struct MippedViewportSource {
    base_w: u32,
    base_h: u32,
    slots: Vec<Option<GenImage<GpuBackend>>>,
}

impl MippedViewportSource {
    /// Create a source with `slot_count` empty slots for a `base_w x base_h` layer.
    pub fn new(base_w: u32, base_h: u32, slot_count: u32) -> Self {
        Self {
            base_w,
            base_h,
            slots: (0..slot_count).map(|_| None).collect(),
        }
    }

    /// Plug the image for a specific mip slot.
    pub fn plug(&mut self, slot: u32, image: GenImage<GpuBackend>) {
        if let Some(s) = self.slots.get_mut(slot as usize) {
            *s = Some(image);
        }
    }

    /// The image at `slot` only if it was explicitly plugged (no fallback). Used
    /// when overlaying onto a base source — unplugged slots defer to the base.
    pub fn plugged(&self, slot: u32) -> Option<GenImage<GpuBackend>> {
        self.slots.get(slot as usize).cloned().flatten()
    }
}

impl ViewportLayerSource for MippedViewportSource {
    fn base_size(&self) -> (u32, u32) {
        (self.base_w, self.base_h)
    }

    fn slot_count(&self) -> u32 {
        self.slots.len() as u32
    }

    fn slot_image(&mut self, slot: u32) -> Option<GenImage<GpuBackend>> {
        // Exact slot if plugged; otherwise fall back to the nearest plugged
        // slot (coarser preferred) so a partially-filled source still renders.
        if let Some(Some(img)) = self.slots.get(slot as usize) {
            return Some(img.clone());
        }
        let lower = (0..slot as usize).rev().find_map(|i| self.slots[i].clone());
        lower.or_else(|| self.slots.iter().flatten().next().cloned())
    }

    fn built_slots(&self) -> Vec<GenImage<GpuBackend>> {
        self.slots.iter().flatten().cloned().collect()
    }
}

// ── VectorGraphicsViewportSource ─────────────────────────────────────────────

/// A Vello-drawn vector layer source: it *accepts a `vello::Scene`* (the vector
/// graphics) rather than a raster image, unifying "overlay" and "layer" — a
/// vector overlay is just another layer source.
///
/// The scene is the input; `slot_image` returns the rasterized result. Wiring
/// the rasterization (renderer drives [`crate::vello_overlay::VelloOverlay`] to
/// render the scene to a texture, then imports it as a GPU image) is the
/// remaining integration step — until then `set_rendered` supplies the raster.
pub struct VectorGraphicsViewportSource {
    base_w: u32,
    base_h: u32,
    /// The vector graphics to draw (the source's actual input).
    scene: Option<vello::Scene>,
    /// Cached rasterization of `scene` at full resolution.
    rendered: Option<GenImage<GpuBackend>>,
}

impl VectorGraphicsViewportSource {
    pub fn new(base_w: u32, base_h: u32) -> Self {
        Self {
            base_w,
            base_h,
            scene: None,
            rendered: None,
        }
    }

    /// Set the vector graphics (Vello scene) this layer draws. Invalidates the
    /// cached raster — the renderer must re-rasterize.
    pub fn set_scene(&mut self, scene: vello::Scene) {
        self.scene = Some(scene);
        self.rendered = None;
    }

    /// The scene awaiting rasterization, if any.
    pub fn scene(&self) -> Option<&vello::Scene> {
        self.scene.as_ref()
    }

    /// Supply the rasterized output for the current scene (set by the renderer's
    /// Vello pass).
    pub fn set_rendered(&mut self, image: GenImage<GpuBackend>) {
        self.rendered = Some(image);
    }
}

impl ViewportLayerSource for VectorGraphicsViewportSource {
    fn base_size(&self) -> (u32, u32) {
        (self.base_w, self.base_h)
    }

    fn slot_count(&self) -> u32 {
        1
    }

    fn slot_image(&mut self, _slot: u32) -> Option<GenImage<GpuBackend>> {
        self.rendered.clone()
    }

    fn built_slots(&self) -> Vec<GenImage<GpuBackend>> {
        self.rendered.iter().cloned().collect()
    }
}
