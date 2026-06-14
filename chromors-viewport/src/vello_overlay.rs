//! Vello vector overlay: rasterizes attached [`VectorGraphics`] over the
//! viewport surface each frame.
//!
//! Two-phase per frame (mirrors the host's render flow):
//!  1. [`VelloOverlay::render_scene`] — builds one `vello::Scene` (every attached
//!     graphic, drawn in image space then transformed by the camera) and renders
//!     it into an offscreen RGBA8 staging texture (Vello does its own submit).
//!  2. [`VelloOverlay::blit_to`] — alpha-blends the staging texture onto the
//!     surface, into the caller's encoder, on top of the image.
//!
//! Graphics are owned by id (`attach`/`detach`/`replace`) so an interactive
//! editor can mutate one element and re-render without rebuilding the rest.

use std::collections::HashMap;

use crate::camera::CameraState;
use crate::vector::VectorGraphics;

pub struct VelloOverlay {
    renderer: vello::Renderer,
    staging: Option<StagingTexture>,
    blitter: wgpu::util::TextureBlitter,
    device: wgpu::Device,
    graphics: HashMap<u64, Box<dyn VectorGraphics>>,
    next_id: u64,
}

struct StagingTexture {
    #[allow(dead_code)] // kept alive for the view borrow
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

impl StagingTexture {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("vello_overlay_staging"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            width,
            height,
        }
    }
}

impl VelloOverlay {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let renderer = vello::Renderer::new(
            device,
            vello::RendererOptions {
                use_cpu: false,
                antialiasing_support: vello::AaSupport::area_only(),
                num_init_threads: None,
                pipeline_cache: None,
            },
        )
        .expect("failed to create Vello renderer");

        let blitter = wgpu::util::TextureBlitterBuilder::new(device, surface_format)
            .blend_state(wgpu::BlendState::ALPHA_BLENDING)
            .build();

        Self {
            renderer,
            staging: None,
            blitter,
            device: device.clone(),
            graphics: HashMap::new(),
            next_id: 0,
        }
    }

    /// Attach a graphic, returning its id (use it to `replace`/`detach`).
    pub fn attach(&mut self, graphic: Box<dyn VectorGraphics>) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.graphics.insert(id, graphic);
        id
    }

    pub fn detach(&mut self, id: u64) -> bool {
        self.graphics.remove(&id).is_some()
    }

    /// Swap the graphic behind an existing id (e.g. an edited bezier). Returns
    /// false if the id is unknown.
    pub fn replace(&mut self, id: u64, graphic: Box<dyn VectorGraphics>) -> bool {
        if let std::collections::hash_map::Entry::Occupied(mut e) = self.graphics.entry(id) {
            e.insert(graphic);
            true
        } else {
            false
        }
    }

    pub fn is_empty(&self) -> bool {
        self.graphics.is_empty()
    }

    /// Render every attached graphic (in image space, transformed by `camera`)
    /// into the staging texture. Vello submits its own commands.
    pub fn render_scene(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        camera: &CameraState,
    ) {
        let needs_resize = self
            .staging
            .as_ref()
            .is_none_or(|s| s.width != width || s.height != height);
        if needs_resize {
            self.staging = Some(StagingTexture::new(device, width, height));
        }
        let staging = self.staging.as_ref().unwrap();

        // Image space → screen: pan then zoom, matching the image pipeline's
        // camera so overlays track the pixels they annotate.
        let transform = vello::kurbo::Affine::translate(vello::kurbo::Vec2::new(
            camera.pan_x as f64,
            camera.pan_y as f64,
        )) * vello::kurbo::Affine::scale(camera.zoom as f64);

        let mut scene = vello::Scene::new();
        for graphic in self.graphics.values() {
            let mut local = vello::Scene::new();
            graphic.draw(&mut local, camera, width, height);
            if graphic.is_screen_space() {
                scene.append(&local, None);
            } else {
                scene.append(&local, Some(transform));
            }
        }

        self.renderer
            .render_to_texture(
                device,
                queue,
                &scene,
                &staging.view,
                &vello::RenderParams {
                    base_color: vello::peniko::Color::TRANSPARENT,
                    width: width.max(1),
                    height: height.max(1),
                    antialiasing_method: vello::AaConfig::Area,
                },
            )
            .expect("Vello render_to_texture failed");
    }

    /// Alpha-blend the staging texture onto `surface_view` via the caller's
    /// encoder. Call after [`Self::render_scene`].
    pub fn blit_to(&self, encoder: &mut wgpu::CommandEncoder, surface_view: &wgpu::TextureView) {
        let Some(staging) = self.staging.as_ref() else {
            return;
        };
        self.blitter
            .copy(&self.device, encoder, &staging.view, surface_view);
    }
}
