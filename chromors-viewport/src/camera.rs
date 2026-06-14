//! Virtual camera for the viewport — handles pan, zoom, and MIP-level
//! selection.
//!
//! The [`Camera`] maps image-space coordinates to screen-space coordinates
//! and determines which MIP level to render at based on the current zoom.
//! [`CameraUniform`] is the GPU-side uniform buffer layout consumed by the
//! viewport shader.

/// Range of tile grid coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileRange {
    pub tx_start: u32,
    pub tx_end: u32,
    pub ty_start: u32,
    pub ty_end: u32,
}

/// GPU uniform buffer layout for the viewport shader.
///
/// Fields are `repr(C)` and derive `bytemuck::Pod` / `Zeroable` so the struct
/// can be written directly to a GPU buffer with `write_buffer`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub vp_w: f32,
    pub vp_h: f32,
    pub pan_x: f32,
    pub pan_y: f32,
    pub zoom: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerUniform {
    pub img_w: f32,
    pub img_h: f32,
    pub img_w0: f32,
    pub img_h0: f32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub atlas_w: f32,
    pub atlas_h: f32,
    pub mip_level: f32,
    pub opacity: f32,
}

/// Computes the maximum MIP level for an image of the given size.
///
/// The maximum MIP is the number of halvings needed to reach ≤ 256 pixels
/// in the largest dimension, capped at 8.
pub fn compute_max_mip(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim <= 256 {
        return 0;
    }
    let levels = (max_dim as f64 / 256.0).log2().floor() as u32;
    levels.min(8)
}

/// Minimum MIP level required so the MIP-level image fits within the GPU's
/// maximum texture dimension.
///
/// Clamps to a conservative 8192 px bound because some GPUs report a higher
/// `max_texture_dimension` than wgpu actually enforces for 2D textures.
pub fn compute_floor_mip(width: u32, height: u32, max_texture_dim: u32) -> u32 {
    let max_tex_dim = max_texture_dim.min(8192);
    let max_img_dim = width.max(height);
    if max_img_dim <= max_tex_dim {
        0
    } else {
        ((max_img_dim as f64 / max_tex_dim as f64).log2().ceil() as u32)
            .min(compute_max_mip(width, height))
    }
}

/// Snapshot of camera state for granular change detection.
///
/// Replaces the coarse `bool stale` flag. The renderer compares the previous
/// snapshot to the current one to determine what needs to be re-prepared.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraState {
    pub pan_x: f32,
    pub pan_y: f32,
    pub zoom: f32,
    pub vp_w: f32,
    pub vp_h: f32,
    pub dpr: f32,
}

impl CameraState {
    /// Whether the visible region changed (pan or zoom moved).
    pub fn region_changed(&self, other: &CameraState) -> bool {
        self.pan_x != other.pan_x || self.pan_y != other.pan_y || self.zoom != other.zoom
    }

    /// Whether the viewport surface size changed (window resize or DPR change).
    pub fn surface_changed(&self, other: &CameraState) -> bool {
        self.vp_w != other.vp_w || self.vp_h != other.vp_h || self.dpr != other.dpr
    }
}

/// Virtual camera controlling what portion of the image is visible on screen.
///
/// The camera stores:
/// - `pan_x`, `pan_y` — image-space offset of the top-left visible corner
///   (before dividing by zoom).
/// - `zoom` — scale factor. 1.0 = 1 image pixel → 1 screen pixel.
/// - `vp_w`, `vp_h` — viewport (window) size in **logical** pixels.
/// - `dpr` — device pixel ratio (1.0 on standard displays, 2.0 on Retina).
///
/// **Physical pixels** = logical pixels × `dpr`. The atlas and render target
/// should be sized in physical pixels; camera uniforms use logical pixels for
/// coordinate math.
#[derive(Clone)]
pub struct Camera {
    pub pan_x: f32,
    pub pan_y: f32,
    pub zoom: f32,
    pub vp_w: f32,
    pub vp_h: f32,
    /// Device pixel ratio — 1.0 for standard displays, 2.0 for HiDPI/Retina.
    pub dpr: f32,
    /// Velocity in logical pixels per second, used for predictive tile fetching.
    pub velocity_x: f32,
    pub velocity_y: f32,
    /// Target zoom level, used for predictive MIP fetching.
    pub target_zoom: Option<f32>,
}

impl Camera {
    /// Creates a new camera for an image of the given size. Initial zoom is
    /// 1.0 and viewport is unset (1×1) — the controller will call `resize`
    /// and `fit` before the first frame.
    pub fn new() -> Self {
        Self {
            pan_x: 0.0,
            pan_y: 0.0,
            zoom: 1.0,
            vp_w: 1.0,
            vp_h: 1.0,
            dpr: 1.0,
            velocity_x: 0.0,
            velocity_y: 0.0,
            target_zoom: None,
        }
    }

    /// Captures a snapshot of the current camera state for change detection.
    pub fn snapshot(&self) -> CameraState {
        CameraState {
            pan_x: self.pan_x,
            pan_y: self.pan_y,
            zoom: self.zoom,
            vp_w: self.vp_w,
            vp_h: self.vp_h,
            dpr: self.dpr,
        }
    }

    /// Updates the viewport dimensions. Called on window resize.
    /// `vp_w` and `vp_h` are in **logical** pixels.
    pub fn resize(&mut self, vp_w: f32, vp_h: f32) {
        self.vp_w = vp_w;
        self.vp_h = vp_h;
    }

    /// Set the device pixel ratio (HiDPI scaling factor).
    pub fn set_dpr(&mut self, dpr: f32) {
        self.dpr = dpr.max(0.5);
    }

    /// Physical viewport width (logical × DPR).
    pub fn physical_w(&self) -> f32 {
        self.vp_w * self.dpr
    }
    /// Physical viewport height (logical × DPR).
    pub fn physical_h(&self) -> f32 {
        self.vp_h * self.dpr
    }

    /// Fits the entire image within the viewport with 5% margin.
    /// Sets zoom and pan so the image is fully visible and centred.
    pub fn fit(&mut self, bounds_w: f32, bounds_h: f32) {
        self.zoom = f32::min(self.vp_w / bounds_w, self.vp_h / bounds_h) * 0.95;
        self.pan_x = -(self.vp_w / self.zoom - bounds_w) / 2.0;
        self.pan_y = -(self.vp_h / self.zoom - bounds_h) / 2.0;
    }

    pub fn min_zoom(&self, bounds_w: f32, bounds_h: f32) -> f32 {
        if bounds_w <= 0.0 || bounds_h <= 0.0 {
            return 1.0 / 512.0;
        }
        let fit = (self.vp_w / bounds_w).min(self.vp_h / bounds_h);
        (fit * 0.2).clamp(1.0 / 512.0, 64.0)
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.pan_x -= dx / self.zoom;
        self.pan_y -= dy / self.zoom;
    }

    pub fn screen_to_world_x(&self, x: f32) -> f32 {
        x / self.zoom + self.pan_x
    }
    pub fn screen_to_world_y(&self, y: f32) -> f32 {
        y / self.zoom + self.pan_y
    }

    pub fn world_to_screen_x(&self, x: f32) -> f32 {
        (x - self.pan_x) * self.zoom
    }
    pub fn world_to_screen_y(&self, y: f32) -> f32 {
        (y - self.pan_y) * self.zoom
    }

    pub fn floor_mip(&self, img_w: u32, img_h: u32, max_texture_dim: u32) -> u32 {
        compute_floor_mip(img_w, img_h, max_texture_dim)
    }

    /// Computes the ideal MIP level for the current zoom and DPR.
    ///
    /// HiDPI adjustment: the effective zoom is multiplied by `dpr` so that
    /// on Retina displays we select a sharper (lower) MIP level — the extra
    /// physical pixels can resolve the additional detail.
    pub fn visible_mip_level(
        &self,
        img_w: u32,
        img_h: u32,
        scale_factor: f32,
        max_texture_dim: u32,
    ) -> u32 {
        let total_zoom = self.zoom * scale_factor * self.dpr;
        let zoom_mip = if total_zoom >= 1.0 {
            0
        } else if total_zoom > 0.0 {
            (-(total_zoom as f64).log2()).max(0.0) as u32
        } else {
            0
        };
        zoom_mip.max(self.floor_mip(img_w, img_h, max_texture_dim))
    }

    pub fn padded_tile_range(
        &self,
        img_w: f32,
        img_h: f32,
        transform_x: f32,
        transform_y: f32,
        transform_scale: f32,
        mip: u32,
        tile_size: u32,
        padding: u32,
    ) -> TileRange {
        let mip_scale = (1u32 << mip) as f32;
        let ts = tile_size as f32;

        let local_pan_x = (self.pan_x - transform_x) / transform_scale;
        let local_pan_y = (self.pan_y - transform_y) / transform_scale;
        let local_vp_w = (self.vp_w / self.zoom) / transform_scale;
        let local_vp_h = (self.vp_h / self.zoom) / transform_scale;

        let x0 = local_pan_x.max(0.0);
        let y0 = local_pan_y.max(0.0);
        let x1 = (local_pan_x + local_vp_w).min(img_w);
        let y1 = (local_pan_y + local_vp_h).min(img_h);

        let mip_w = (img_w as u32 >> mip).max(1);
        let mip_h = (img_h as u32 >> mip).max(1);
        let ntx = mip_w.div_ceil(tile_size);
        let nty = mip_h.div_ceil(tile_size);

        let tx_start = ((x0 / mip_scale / ts).floor() as u32).saturating_sub(padding);
        let ty_start = ((y0 / mip_scale / ts).floor() as u32).saturating_sub(padding);
        let tx_end = (((x1 / mip_scale / ts).ceil() as u32) + padding).min(ntx);
        let ty_end = (((y1 / mip_scale / ts).ceil() as u32) + padding).min(nty);

        TileRange {
            tx_start,
            tx_end,
            ty_start,
            ty_end,
        }
    }

    /// Converts camera state into a [`CameraUniform`] for upload to the GPU.
    /// `img_w` and `img_h` in the uniform are the MIP-level dimensions.
    pub fn to_uniform(&self) -> CameraUniform {
        CameraUniform {
            vp_w: self.vp_w,
            vp_h: self.vp_h,
            pan_x: self.pan_x,
            pan_y: self.pan_y,
            zoom: self.zoom,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}
