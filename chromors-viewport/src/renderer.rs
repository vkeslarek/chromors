use poc::backend::gpu::GpuBackend;
use poc::data::image::Image2D as GenImage;

use crate::atlas::TexturePool;
use crate::camera::{Camera, CameraState, LayerUniform};
use crate::fetcher::{FetchPayload, FetchTask, TileFetcher, compute_cache_key, stage_aligned};
use crate::layer::ImageLayer;
use crate::pipeline::ViewportPipeline;
use crate::rect::Rect;

/// 256 MiB texture pool budget
const TEXTURE_POOL_BUDGET: u64 = 256 * 1024 * 1024;

/// Tiles not drawn for this many frames are eligible for LRU eviction.
const LRU_MAX_AGE_FRAMES: u64 = 120;

/// How often (in frames) to run LRU eviction.
const LRU_EVICT_INTERVAL: u64 = 30;

/// How often (in frames) to run GPU cache GC. Must be a multiple of LRU_EVICT_INTERVAL
/// since it's gated by the same periodic check.
const CACHE_GC_INTERVAL: u64 = 30;
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ViewportBounds {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

pub struct ViewportRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: ViewportPipeline,
    camera_bg: wgpu::BindGroup,

    pub camera: Camera,
    pub bounds: ViewportBounds,
    pub stale: bool,
    prev_camera: CameraState,

    pub layers: Vec<ImageLayer>,
    next_layer_id: u64,

    fetcher: TileFetcher,
    fetch_rx: std::sync::mpsc::Receiver<FetchTask>,
    texture_pool: TexturePool,

    frame_counter: u64,

    /// Pending cross-boundary span IDs (start_span / finish_span).
    pub pending_spans: Vec<u64>,

    /// Interactive vector-graphics overlay, composited over the image each
    /// frame (lazily created so it shares this renderer's device + format).
    overlay: crate::vello_overlay::VelloOverlay,
}

impl ViewportRenderer {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        render_target_format: wgpu::TextureFormat,
        bounds: ViewportBounds,
    ) -> Self {
        let pipeline = ViewportPipeline::new(&device, render_target_format);
        let overlay = crate::vello_overlay::VelloOverlay::new(&device, render_target_format);
        let (fetch_tx, fetch_rx) = std::sync::mpsc::channel();
        let fetcher = TileFetcher::new(device.clone(), queue.clone(), fetch_tx);
        let camera = Camera::new();
        let prev_camera = camera.snapshot();
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cam_bg"),
            layout: &pipeline.camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: pipeline.camera_buf.as_entire_binding(),
            }],
        });

        Self {
            device,
            queue,
            pipeline,
            camera_bg,
            camera,
            bounds,
            stale: true,
            prev_camera,
            layers: Vec::new(),
            next_layer_id: 0,
            fetcher,
            fetch_rx,
            texture_pool: TexturePool::new(TEXTURE_POOL_BUDGET),
            frame_counter: 0,
            pending_spans: Vec::new(),
            overlay,
        }
    }

    /// Attach a vector graphic to the interactive overlay (drawn over the image
    /// in image-space, tracking camera pan/zoom). Returns its id for
    /// `replace_graphic`/`detach_graphic`.
    pub fn attach_graphic(
        &mut self,
        graphic: Box<dyn crate::vector::VectorGraphics>,
    ) -> u64 {
        self.stale = true;
        self.overlay.attach(graphic)
    }

    /// Swap the graphic behind `id` (e.g. an edited bezier) and request a redraw.
    pub fn replace_graphic(
        &mut self,
        id: u64,
        graphic: Box<dyn crate::vector::VectorGraphics>,
    ) -> bool {
        self.stale = true;
        self.overlay.replace(id, graphic)
    }

    pub fn detach_graphic(&mut self, id: u64) -> bool {
        self.stale = true;
        self.overlay.detach(id)
    }

    pub fn attach_image(&mut self, image: GenImage<GpuBackend>) -> u64 {
        let id = self.next_layer_id;
        self.next_layer_id += 1;

        let mut layer = ImageLayer::new(id, image);

        let mut bufs = Vec::new();
        for mip in 0..9 {
            bufs.push(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("layer_{}_mip_{}_buf", id, mip)),
                size: std::mem::size_of::<LayerUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        layer.layer_bufs = Some(bufs);

        self.layers.push(layer);
        self.fetcher.bump_version();
        self.stale = true;
        id
    }

    pub fn detach_image(&mut self, id: u64) -> bool {
        if let Some(idx) = self.layers.iter().position(|l| l.id == id) {
            let mut layer = self.layers.remove(idx);
            for (_, state) in layer.mip_states.iter_mut() {
                state.return_to_pool(&mut self.texture_pool);
            }
            self.fetcher.bump_version();
            self.stale = true;
            true
        } else {
            false
        }
    }

    pub fn clear_images(&mut self) {
        for layer in &mut self.layers {
            for (_, state) in layer.mip_states.iter_mut() {
                state.return_to_pool(&mut self.texture_pool);
            }
        }
        self.layers.clear();
        self.fetcher.bump_version();
        self.stale = true;
    }

    /// Soft swap: replace the layer's image without clearing the atlas. The old
    /// tiles stay on screen until each tile's new-version fetch overwrites it in
    /// place — no blank/blink. Hard swaps (full invalidate) go through
    /// [`attach_image`] / [`detach_image`] / [`clear`].
    pub fn replace_image(
        &mut self,
        id: u64,
        image: GenImage<GpuBackend>,
        dirty_regions: Vec<Rect>,
    ) -> bool {
        if let Some(idx) = self.layers.iter().position(|l| l.id == id) {
            let layer = &mut self.layers[idx];
            // Fresh image source — drops the cached mip chain so it rebuilds
            // from the new base. The atlas (rendered tiles) is intentionally
            // kept so the previous frame stays visible during the swap.
            layer.set_source(Box::new(crate::source::ImageViewportSource::new(image)));

            layer.image_version += 1;

            // Clear fetching markers so dispatch_fetches re-allocates with the new image
            for state in layer.mip_states.values_mut() {
                state.fetching.clear();
            }

            let target = layer.target_mip;
            layer.ensure_mip(target);

            let fallback = layer.fallback_mip();
            layer.ensure_mip(fallback.max(target));

            self.fetcher.bump_version();
            self.stale = true;

            for rect in dirty_regions {
                self.invalidate_region(idx, rect);
            }
            true
        } else {
            false
        }
    }
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            for (_, state) in layer.mip_states.iter_mut() {
                state.return_to_pool(&mut self.texture_pool);
            }
        }
        self.layers.clear();
        self.fetcher.bump_version();
        self.stale = true;
    }

    /// Plug a single mip slot of the layer's override source (soft preview
    /// overlay on top of the base source).
    pub fn set_mip_override(&mut self, layer_id: u64, mip: u32, img: GenImage<GpuBackend>) {
        if let Some(idx) = self.layers.iter().position(|l| l.id == layer_id) {
            let layer = &mut self.layers[idx];
            let (bw, bh) = (layer.base_w, layer.base_h);
            let slots = layer.source.slot_count().max(mip + 1);
            let ov = layer
                .override_source
                .get_or_insert_with(|| crate::source::MippedViewportSource::new(bw, bh, slots));
            ov.plug(mip, img);

            if let Some(state) = layer.mip_states.get_mut(&mip) {
                state.fetching.clear();
            }

            self.fetcher.bump_version();
            self.stale = true;
            self.invalidate_region(idx, Rect::new(0, 0, bw as i32, bh as i32));
        }
    }

    pub fn clear_mip_overrides(&mut self, layer_id: u64) {
        if let Some(idx) = self.layers.iter().position(|l| l.id == layer_id) {
            let layer = &mut self.layers[idx];
            if layer.override_source.is_some() {
                layer.override_source = None;
                let (w, h) = (layer.base_w as i32, layer.base_h as i32);

                for state in layer.mip_states.values_mut() {
                    state.fetching.clear();
                }

                self.fetcher.bump_version();
                self.stale = true;
                self.invalidate_region(idx, Rect::new(0, 0, w, h));
            }
        }
    }

    pub fn add_bench_span(&mut self, id: u64) {
        self.pending_spans.push(id);
    }

    /// Single-composite convenience: hard [`attach_image`] on first image, then
    /// soft [`replace_image`] for every subsequent update.
    pub fn update_composite(&mut self, image: GenImage<GpuBackend>) {
        if self.layers.is_empty() {
            self.attach_image(image);
        } else {
            let id = self.layers[0].id;
            let w = image.width() as i32;
            let h = image.height() as i32;
            self.replace_image(id, image, vec![Rect::new(0, 0, w, h)]);
        }
    }
    pub fn clear_tile_cache(&mut self, layer_idx: usize) {
        if let Some(layer) = self.layers.get_mut(layer_idx) {
            layer.mip_states.clear();
            layer.image_version += 1;
            self.fetcher.bump_version();
            self.stale = true;
        }
    }

    pub fn local_vram_bytes(&self) -> u64 {
        let mut total = self.texture_pool.total_bytes;
        for layer in &self.layers {
            for state in layer.mip_states.values() {
                if let Some((_, _, w, h)) = state.atlas {
                    // Assuming Rgba8 format (4 bytes per pixel) or Rgba16f (8 bytes).
                    // We'll approximate with 4 bytes for now, or we can use the pipeline format.
                    let bpp = 4;
                    total += (w as u64) * (h as u64) * bpp;
                }
            }
        }
        total
    }

    pub fn device_allocated_bytes(&self) -> u64 {
        if let Some(ctx) = self.device_ctx() {
            ctx.allocated_bytes
                .load(std::sync::atomic::Ordering::Relaxed)
        } else {
            0
        }
    }

    fn device_ctx(&self) -> Option<std::sync::Arc<poc::backend::gpu::GpuContext>> {
        for layer in &self.layers {
            if let Some(img) = layer.source.built_slots().first() {
                return Some(img.ctx.clone());
            }
        }
        None
    }

    pub fn clamp_camera(&mut self) {
        let (bw, bh) = self
            .layers
            .first()
            .map(|l| (l.base_w as f32, l.base_h as f32))
            .unwrap_or((1.0, 1.0));

        let vp_w_img = self.camera.vp_w / self.camera.zoom;
        let vp_h_img = self.camera.vp_h / self.camera.zoom;

        let min_pan_x = -vp_w_img / 2.0;
        let max_pan_x = bw - vp_w_img / 2.0;
        self.camera.pan_x = self.camera.pan_x.clamp(min_pan_x, max_pan_x);

        let min_pan_y = -vp_h_img / 2.0;
        let max_pan_y = bh - vp_h_img / 2.0;
        self.camera.pan_y = self.camera.pan_y.clamp(min_pan_y, max_pan_y);
    }


    pub fn resize(&mut self, w: f32, h: f32) {
        self.camera.resize(w, h);
        self.clamp_camera();
        self.stale = true;
    }
    pub fn set_dpr(&mut self, dpr: f32) {
        self.camera.set_dpr(dpr);
        self.clamp_camera();
        self.stale = true;
    }
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.camera.pan(dx, dy);
        self.clamp_camera();
        self.stale = true;
    }
    pub fn zoom(&mut self, delta: f32) {
        let f = 1.0 + delta.abs() * 0.1;
        let z = if delta > 0.0 {
            self.camera.zoom * f
        } else {
            self.camera.zoom / f
        };
        let (bw, bh) = self
            .layers
            .first()
            .map(|l| (l.base_w as f32, l.base_h as f32))
            .unwrap_or((1.0, 1.0));
        self.camera.zoom = z.clamp(self.camera.min_zoom(bw, bh), 64.0);
        self.clamp_camera();
        self.stale = true;
    }
    pub fn fit(&mut self) {
        let (bw, bh) = self
            .layers
            .first()
            .map(|l| (l.base_w as f32, l.base_h as f32))
            .unwrap_or((1.0, 1.0));
        self.camera.fit(bw, bh);
        self.clamp_camera();
        self.stale = true;
    }
    pub fn reset(&mut self) {
        self.camera.zoom = 1.0;
        self.camera.pan_x = 0.0;
        self.camera.pan_y = 0.0;
        self.clamp_camera();
        self.stale = true;
    }

    pub fn prepare(&mut self) {
        self.frame_counter += 1;

        if self.process_fetch_responses() {
            self.stale = true;
        }

        let tw = self.camera.vp_w;
        let th = self.camera.vp_h;
        if tw <= 0.0 || th <= 0.0 {
            return;
        }

        // Granular change detection via CameraState snapshot
        let current_snap = self.camera.snapshot();
        if current_snap != self.prev_camera {
            self.stale = true;
            self.prev_camera = current_snap;
        }

        if self.update_target_mips() {
            self.stale = true;
        }

        // Periodic LRU eviction + GC
        if self.frame_counter.is_multiple_of(LRU_EVICT_INTERVAL) {
            self.run_lru_eviction();
            if self.frame_counter.is_multiple_of(CACHE_GC_INTERVAL) {
                self.run_cache_gc();
            }
        }

        if !self.stale {
            return;
        }
        self.stale = false;

        self.dispatch_fetches();
    }

    pub fn process_fetch_responses(&mut self) -> bool {
        let _span = tracing::trace_span!("vp.fetch_work").entered();
        let mut fetch_count = 0;
        let mut encoder: Option<wgpu::CommandEncoder> = None;

        while let Ok(task) = self.fetch_rx.try_recv() {
            if let Some(layer) = self.layers.iter_mut().find(|l| l.id == task.layer_id)
                && let Some(state) = layer.mip_states.get_mut(&task.mip)
            {
                let slot_ok = state
                    .tile_fetch_version(&(task.tx, task.ty))
                    .is_some_and(|fv| fv == task.version);
                if !slot_ok {
                    continue;
                }
                if let Some((sx, sy)) = state.get_slot(&(task.tx, task.ty))
                    && let Some((ref tex, _, _, _)) = state.atlas
                {
                    match task.kind {
                        FetchPayload::Staged {
                            ref buffer,
                            offset,
                            bytes_per_row,
                        } => {
                            let enc = encoder.get_or_insert_with(|| {
                                self.device.create_command_encoder(
                                    &wgpu::CommandEncoderDescriptor {
                                        label: Some("fetch_gpu_copy"),
                                    },
                                )
                            });
                            enc.copy_buffer_to_texture(
                                wgpu::TexelCopyBufferInfo {
                                    buffer,
                                    layout: wgpu::TexelCopyBufferLayout {
                                        offset,
                                        bytes_per_row: Some(bytes_per_row),
                                        rows_per_image: None,
                                    },
                                },
                                wgpu::TexelCopyTextureInfo {
                                    texture: tex,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d {
                                        x: sx + task.slot_offset_x,
                                        y: sy + task.slot_offset_y,
                                        z: 0,
                                    },
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::Extent3d {
                                    width: task.width,
                                    height: task.height,
                                    depth_or_array_layers: 1,
                                },
                            );
                        }
                        FetchPayload::Raw {
                            ref buffer,
                            offset,
                            src_row_bytes,
                            bpp,
                        } => {
                            let src_x = (offset % src_row_bytes as u64) / bpp as u64;
                            let src_y = offset / src_row_bytes as u64;
                            let (staged, aligned_bpr) = stage_aligned(
                                &self.device,
                                &self.queue,
                                buffer,
                                src_row_bytes,
                                src_x as u32,
                                src_y as u32,
                                task.width,
                                task.height,
                                bpp,
                            );
                            let enc = encoder.get_or_insert_with(|| {
                                self.device.create_command_encoder(
                                    &wgpu::CommandEncoderDescriptor {
                                        label: Some("fetch_gpu_copy"),
                                    },
                                )
                            });
                            enc.copy_buffer_to_texture(
                                wgpu::TexelCopyBufferInfo {
                                    buffer: &staged,
                                    layout: wgpu::TexelCopyBufferLayout {
                                        offset: 0,
                                        bytes_per_row: Some(aligned_bpr),
                                        rows_per_image: None,
                                    },
                                },
                                wgpu::TexelCopyTextureInfo {
                                    texture: tex,
                                    mip_level: 0,
                                    origin: wgpu::Origin3d {
                                        x: sx + task.slot_offset_x,
                                        y: sy + task.slot_offset_y,
                                        z: 0,
                                    },
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::Extent3d {
                                    width: task.width,
                                    height: task.height,
                                    depth_or_array_layers: 1,
                                },
                            );
                        }
                    }
                }
                if task.slot_offset_x == 0 && task.slot_offset_y == 0 {
                    state.fetching.remove(&(task.tx, task.ty));
                    state.mark_valid(&(task.tx, task.ty));
                }
                fetch_count += 1;
            }
        }

        // Submit all GPU→GPU copies in a single batch
        if let Some(enc) = encoder {
            self.queue.submit(std::iter::once(enc.finish()));
        }

        if fetch_count > 0 {
            for id in self.pending_spans.drain(..) {
                crate::bench::finish_span(id);
            }
        }

        fetch_count > 0
    }

    fn update_target_mips(&mut self) -> bool {
        let max_dim = self.device.limits().max_texture_dimension_2d;
        let mut changed = false;
        for layer in &mut self.layers {
            let mip = self.camera.visible_mip_level(
                layer.base_w,
                layer.base_h,
                layer.transform.scale_x,
                max_dim,
            );
            if mip != layer.target_mip {
                layer.target_mip = mip;
                changed = true;
            }
        }
        changed
    }

    pub fn is_fetching(&self) -> bool {
        for layer in &self.layers {
            for state in layer.mip_states.values() {
                if !state.fetching.is_empty() {
                    return true;
                }
            }
        }
        false
    }

    /// LRU eviction: free tile atlas slots not used for `LRU_MAX_AGE_FRAMES`.
    fn run_lru_eviction(&mut self) {
        for layer in &mut self.layers {
            for (_, state) in layer.mip_states.iter_mut() {
                state.evict_stale(self.frame_counter, LRU_MAX_AGE_FRAMES);
            }
        }
    }

    /// Cache eviction is now automatic via `TieredCache` LRU.
    fn run_cache_gc(&self) {}

    fn dispatch_fetches(&mut self) {
        let _span = tracing::trace_span!("vp.fetch_disp").entered();
        let mut fetch_requests = Vec::new();
        let frame = self.frame_counter;

        for i in 0..self.layers.len() {
            let fallback_mip = self.layers[i].fallback_mip();
            let mut target_mips = vec![self.layers[i].target_mip, fallback_mip];

            if let Some(target_zoom) = self.camera.target_zoom {
                let max_dim = self.device.limits().max_texture_dimension_2d;
                let mut dummy_cam = self.camera.clone();
                dummy_cam.zoom = target_zoom;

                let predicted_mip = dummy_cam.visible_mip_level(
                    self.layers[i].base_w,
                    self.layers[i].base_h,
                    self.layers[i].transform.scale_x,
                    max_dim,
                );

                if !target_mips.contains(&predicted_mip) {
                    target_mips.push(predicted_mip);
                }
            }

            for m in target_mips {
                let mip_w = (self.layers[i].base_w >> m).max(1);
                let mip_h = (self.layers[i].base_h >> m).max(1);
                let visible = self.layers[i].visible_tiles(&self.camera, m, mip_w, mip_h);

                // Cache key validation: invalidate atlas if image or transform changed
                let layer = &mut self.layers[i];
                let new_key = compute_cache_key(m, layer.image_version);
                let state = layer.mip_states.entry(m).or_default();
                if state.cache_key != new_key && state.cache_key != 0 {
                    state.fetching.clear();
                    state.cache_key = new_key;
                } else if state.cache_key == 0 {
                    state.cache_key = new_key;
                }

                let current_version = state.cache_key;
                let missing: Vec<(u32, u32)> = visible
                    .iter()
                    .copied()
                    .filter(|k| {
                        !state.has_tile_version(k, current_version) && !state.fetching.contains(k)
                    })
                    .collect();

                // Touch visible tiles for LRU tracking
                for key in &visible {
                    state.touch(key, frame);
                }

                if missing.is_empty() {
                    continue;
                }

                let state = layer.mip_states.get_mut(&m).unwrap();

                state.ensure_with_pool(
                    &self.device,
                    self.pipeline.tex_format,
                    m,
                    mip_w,
                    mip_h,
                    Some(&mut self.texture_pool),
                );

                let state = layer.mip_states.get_mut(&m).unwrap();
                let current_fetch_version = self.fetcher.version();
                let mut newly_fetching = Vec::new();
                for &(tx, ty) in &missing {
                    if let Some(_pos) = state.alloc_or_recycle_slot(
                        (tx, ty),
                        state.cache_key,
                        current_fetch_version,
                    ) {
                        newly_fetching.push((tx, ty));
                    }
                }

                if !newly_fetching.is_empty() {
                    fetch_requests.push((i, m, mip_w, mip_h, newly_fetching));
                }
            }
        }

        for (i, m, mip_w, mip_h, missing) in fetch_requests {
            let layer_id = self.layers[i].id;
            // On-screen mip first; fallback/predictive mips are lower priority.
            let priority = if m == self.layers[i].target_mip { 0 } else { 1 };
            let Some(mip_img) = self.layers[i].mip_image(m) else {
                continue;
            };

            self.fetcher
                .spawn_fetch(layer_id, m, mip_img, missing, mip_w, mip_h, priority);
        }
    }

    pub fn invalidate_region(&mut self, layer_idx: usize, rect: Rect) {
        self.layers[layer_idx].invalidate_region(rect, &self.fetcher);
    }

    pub fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_x: u32,
        clip_y: u32,
        clip_w: u32,
        clip_h: u32,
    ) {
        let _span = tracing::trace_span!("vp.draw").entered();
        let frame = self.frame_counter;

        let cam_uniform = self.camera.to_uniform();
        self.queue.write_buffer(
            &self.pipeline.camera_buf,
            0,
            bytemuck::bytes_of(&cam_uniform),
        );

        // Collect (layer_idx, mip, tiles_to_touch) for deferred LRU updates.
        let mut touch_list: Vec<(usize, u32, Vec<(u32, u32)>)> = Vec::new();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let fallback_mip = layer.fallback_mip();
            let mut active_mips: Vec<u32> = if layer.override_source.is_some() {
                // Preview override only covers target_mip. It can be
                // semi-transparent (layer opacity baked into alpha), so stacking
                // the un-overridden fallback mip under it would bleed the
                // committed composite through — draw ONLY the override mip.
                vec![layer.target_mip]
            } else {
                layer
                    .mip_states
                    .keys()
                    .copied()
                    .filter(|&m| m >= layer.target_mip || m == fallback_mip)
                    .collect()
            };
            active_mips.sort_by(|a, b| b.cmp(a));
            active_mips.dedup();

            for mip in active_mips {
                let state = match layer.mip_states.get(&mip) {
                    Some(s) => s,
                    None => continue,
                };
                if state.atlas.is_none() {
                    continue;
                }
                let (_, ref atlas_view, aw, ah) = *state.atlas.as_ref().unwrap();

                let mip_w = (layer.base_w >> mip).max(1);
                let mip_h = (layer.base_h >> mip).max(1);

                let layer_uniform = LayerUniform {
                    img_w: mip_w as f32,
                    img_h: mip_h as f32,
                    img_w0: layer.base_w as f32,
                    img_h0: layer.base_h as f32,
                    pos_x: layer.transform.x,
                    pos_y: layer.transform.y,
                    scale_x: layer.transform.scale_x,
                    scale_y: layer.transform.scale_y,
                    atlas_w: aw as f32,
                    atlas_h: ah as f32,
                    mip_level: mip as f32,
                    opacity: layer.transform.opacity,
                };
                self.queue.write_buffer(
                    &layer.layer_bufs.as_ref().unwrap()[mip as usize],
                    0,
                    bytemuck::bytes_of(&layer_uniform),
                );

                let visible = layer.visible_tiles(&self.camera, mip, mip_w, mip_h);

                let mut instances = Vec::new();
                let mut touched = Vec::new();
                for &(tx, ty) in &visible {
                    if let Some((sx, sy)) = state.get_slot(&(tx, ty))
                        && state.has_valid_data(&(tx, ty))
                    {
                        instances.push([tx, ty, sx, sy]);
                        touched.push((tx, ty));
                    }
                }

                if instances.is_empty() {
                    continue;
                }

                if !touched.is_empty() {
                    touch_list.push((layer_idx, mip, touched));
                }

                let is_small = layer.base_w < 512 && layer.base_h < 512;
                let sampler = if (is_small && self.camera.zoom >= 1.0)
                    || (self.camera.zoom * layer.transform.scale_x >= 3.0)
                {
                    &self.pipeline.atlas_sampler_nearest
                } else {
                    &self.pipeline.atlas_sampler_linear
                };
                let atlas_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("atlas_bg"),
                    layout: &self.pipeline.atlas_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(atlas_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: layer.layer_bufs.as_ref().unwrap()[mip as usize]
                                .as_entire_binding(),
                        },
                    ],
                });

                let instance_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("instances"),
                    size: (instances.len() * std::mem::size_of::<[u32; 4]>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue
                    .write_buffer(&instance_buf, 0, bytemuck::cast_slice(&instances));

                let load_op = wgpu::LoadOp::Load;

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mip_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: target,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: load_op,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                let dpr = self.camera.dpr;
                pass.set_viewport(
                    self.bounds.x * dpr,
                    self.bounds.y * dpr,
                    self.bounds.width * dpr,
                    self.bounds.height * dpr,
                    0.0,
                    1.0,
                );
                pass.set_scissor_rect(clip_x, clip_y, clip_w, clip_h);
                pass.set_pipeline(&self.pipeline.render_pipeline);
                pass.set_bind_group(0, &self.camera_bg, &[]);
                pass.set_bind_group(1, &atlas_bg, &[]);
                pass.set_vertex_buffer(0, instance_buf.slice(..));
                pass.draw(0..4, 0..instances.len() as u32);
            }
        }

        // Deferred LRU touch — all atlas_view borrows are released now
        for (layer_idx, mip, tiles) in touch_list {
            if let Some(state) = self.layers[layer_idx].mip_states.get_mut(&mip) {
                for key in tiles {
                    state.touch(&key, frame);
                }
            }
        }

        if !self.overlay.is_empty() {
            let w = (self.bounds.width * self.camera.dpr) as u32;
            let h = (self.bounds.height * self.camera.dpr) as u32;
            let snap = self.camera.snapshot();
            self.overlay.render_scene(&self.device, &self.queue, w, h, &snap);
            self.overlay.blit_to(encoder, target);
        }
    }

    pub fn camera_bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.pipeline.camera_bgl
    }

    pub fn create_camera_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cam_bg"),
            layout: &self.pipeline.camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.pipeline.camera_buf.as_entire_binding(),
            }],
        })
    }
}
