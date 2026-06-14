use std::path::PathBuf;
use std::sync::Arc;

use chromors_viewport::ViewportController;
use chromors_viewport::ViewportRenderer;

use tao::{
    dpi::PhysicalSize,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, StartCause, WindowEvent},
    event_loop::ControlFlow,
    keyboard::KeyCode,
    window::WindowBuilder,
};

use crate::gpu::GpuState;

use poc::backend::gpu::GpuContext;
use poc::backend::vips::VipsBackend;
use poc::color::space::ColorSpace;
use poc::data::image::{Image2D as GenImage, VipsImageSource};
use poc::node::Data;
use poc::pixel::{AlphaState, PixelLayout, Storage};

use poc::color::model::ColorModel;

const INITIAL_W: u32 = 1024;
const INITIAL_H: u32 = 768;

/// Display layout for the swapchain: straight-alpha sRGB RGBA8.
fn display_layout() -> PixelLayout {
    PixelLayout {
        storage: Storage::U8,
        model: ColorModel::Rgb,
        alpha: AlphaState::Straight,
        color_space: ColorSpace::SRGB,
    }
}

pub struct ImageViewerApp {
    pub gpu: Option<GpuState>,
    pub window: Option<Arc<tao::window::Window>>,
    pub viewport: Option<ViewportRenderer>,
    pub gpu_ctx: Option<Arc<GpuContext>>,
    pub controller: ViewportController,
    pub last_frame_time: std::time::Instant,
    pub pending_open: Option<std::sync::mpsc::Receiver<PathBuf>>,
    pub force_frames: u32,
    pub ctrl_held: bool,
    pub initialized: bool,
    pub has_image: bool,
    pub initial_path: Option<PathBuf>,
}

impl ImageViewerApp {
    pub fn new() -> Self {
        Self {
            gpu: None,
            window: None,
            viewport: None,
            gpu_ctx: None,
            controller: ViewportController::new(),
            last_frame_time: std::time::Instant::now(),
            pending_open: None,
            force_frames: 0,
            ctrl_held: false,
            initialized: false,
            has_image: false,
            initial_path: None,
        }
    }

    pub fn run(mut self, initial_path: Option<PathBuf>) {
        self.initial_path = initial_path;
        let event_loop = tao::event_loop::EventLoop::new();
        event_loop.run(move |event, elwt, control_flow| {
            *control_flow = ControlFlow::Poll;
            self.handle_event(event, elwt, control_flow);
        });
    }

    pub fn init(&mut self, elwt: &tao::event_loop::EventLoopWindowTarget<()>) {
        if self.initialized {
            return;
        }
        self.initialized = true;

        let window = Arc::new(
            WindowBuilder::new()
                .with_title("chromors-viewport — Ctrl+O to open an image")
                .with_inner_size(PhysicalSize::new(INITIAL_W, INITIAL_H))
                .build(elwt)
                .unwrap(),
        );

        self.window = Some(window.clone());

        let gpu = GpuState::init(window);
        let mut viewport = ViewportRenderer::new(
            gpu.device.clone(),
            gpu.queue.clone(),
            gpu.surface_config.format,
            chromors_viewport::ViewportBounds {
                x: 0.0,
                y: 0.0,
                width: INITIAL_W as f32,
                height: INITIAL_H as f32,
            },
        );
        viewport.resize(INITIAL_W as f32, INITIAL_H as f32);

        let limits = gpu.device.limits();
        self.gpu_ctx = Some(GpuContext::from_device(
            Arc::new(gpu.device.clone()),
            Arc::new(gpu.queue.clone()),
            &limits,
        ));
        self.gpu = Some(gpu);

        self.viewport = Some(viewport);

        if let Some(path) = self.initial_path.take() {
            self.load_image(&path);
        }

        self.request_frame();
    }

    pub fn request_open_file(&mut self) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.pending_open = Some(rx);
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new().pick_file() {
                let _ = tx.send(path);
            }
        });
    }

    pub fn load_image(&mut self, path: &std::path::Path) {
        if let (Some(ctx), Some(vp)) = (self.gpu_ctx.as_ref(), self.viewport.as_mut()) {
            vp.clear_images();

            let vips_img = match GenImage::<VipsBackend>::open(path.to_str().expect("invalid path"))
            {
                Ok(img) => img,
                Err(e) => {
                    tracing::error!(target: "app", "failed to open {path:?}: {e:?}");
                    return;
                }
            };
            let w = vips_img.width();
            let h = vips_img.height();
            tracing::info!(target: "app", "loaded {w}x{h} from {path:?}");

            let src = VipsImageSource::new(vips_img);
            let gpu_img: GenImage<poc::backend::gpu::GpuBackend> =
                Data::from_source(Arc::new(src), ctx.clone());

            // A multi-op fused color chain (exercises step-namespaced
            // param_block fusion). Net effect is mild so the photo stays
            // recognizable; paired exposures roughly cancel.
            let processed = gpu_img
                .exposure(0.3, 0.0)
                .brightness(1.0) // gain (multiplicative) — 1.0 = neutral
                .saturation(1.2)
                .gamma(Some(1.0)) // 1.0 = neutral
                .linear(vec![1.05, 1.05, 1.05], vec![0.0, 0.0, 0.0])
                .invert()
                .invert()
                .exposure(-0.3, 0.0)
                .blur(20.0);

            let display_img = processed.convert(display_layout());

            vp.attach_image(display_img);
            vp.fit();
            self.has_image = true;
            self.force_frames = 120;
            self.request_frame();
        }
    }

    pub fn render_frame(&mut self) {
        if let Some(rx) = &self.pending_open
            && let Ok(path) = rx.try_recv()
        {
            self.pending_open = None;
            self.load_image(&path);
        }

        let Some(gpu) = self.gpu.as_ref() else {
            return;
        };
        let Some(vp) = self.viewport.as_mut() else {
            return;
        };

        let now = std::time::Instant::now();
        let dt = now
            .duration_since(self.last_frame_time)
            .as_secs_f32()
            .clamp(0.001, 0.1);
        self.last_frame_time = now;

        if self.controller.update_physics(vp, dt)
            && let Some(w) = &self.window
        {
            w.request_redraw();
        }

        vp.prepare();

        let output = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(tex) => tex,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Suboptimal(_) => {
                gpu.surface.configure(&gpu.device, &gpu.surface_config);
                return;
            }
            _ => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut enc = gpu.device.create_command_encoder(&Default::default());

        // Clear the swapchain texture. ViewportRenderer::draw() uses LoadOp::Load
        // (it assumes the texture was already cleared by the host framework).
        {
            let _clear_pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.08,
                            b: 0.08,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }

        vp.draw(
            &mut enc,
            &view,
            0,
            0,
            gpu.surface_config.width,
            gpu.surface_config.height,
        );

        gpu.queue.submit(std::iter::once(enc.finish()));
        output.present();
    }

    pub fn request_frame(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    pub fn handle_event(
        &mut self,
        event: Event<()>,
        elwt: &tao::event_loop::EventLoopWindowTarget<()>,
        control_flow: &mut ControlFlow,
    ) {
        match event {
            Event::NewEvents(StartCause::Init) => {
                self.init(elwt);
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(size) => {
                    if let Some(gpu) = self.gpu.as_mut() {
                        gpu.reconfigure(size.width, size.height);
                    }
                    if let Some(vp) = self.viewport.as_mut() {
                        vp.bounds.width = size.width as f32;
                        vp.bounds.height = size.height as f32;
                        vp.resize(size.width as f32, size.height as f32);
                    }
                    self.request_frame();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if let Some(vp) = self.viewport.as_mut() {
                        self.controller.on_mouse_move(
                            position.x as f32,
                            position.y as f32,
                            &mut Default::default(),
                            vp,
                        );
                    }
                    self.request_frame();
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        if let Some(vp) = self.viewport.as_mut() {
                            if state == ElementState::Pressed {
                                let (x, y) = self.controller.last_cursor.unwrap_or((0.0, 0.0));
                                self.controller
                                    .on_mouse_down(x, y, &mut Default::default(), vp);
                            } else {
                                self.controller.on_mouse_up(&mut Default::default());
                            }
                        }
                        self.request_frame();
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    if let Some(vp) = self.viewport.as_mut() {
                        let dy = match delta {
                            MouseScrollDelta::LineDelta(_, y) => y * 20.0,
                            MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                            _ => 0.0,
                        };
                        let (x, y) = self.controller.last_cursor.unwrap_or((0.0, 0.0));
                        self.controller.on_scroll(dy, x, y, vp);
                    }
                    self.request_frame();
                }
                WindowEvent::KeyboardInput { event: ev, .. }
                    if ev.state == ElementState::Pressed =>
                {
                    match ev.physical_key {
                        KeyCode::KeyF => {
                            if let Some(vp) = self.viewport.as_mut() {
                                vp.fit();
                                self.controller.target_zoom = Some(vp.camera.zoom);
                            }
                            self.request_frame();
                        }
                        KeyCode::Digit0 => {
                            if let Some(vp) = self.viewport.as_mut() {
                                vp.reset();
                                self.controller.target_zoom = Some(vp.camera.zoom);
                            }
                            self.request_frame();
                        }
                        KeyCode::KeyO if self.ctrl_held => {
                            self.request_open_file();
                        }
                        KeyCode::KeyQ if self.ctrl_held => {
                            *control_flow = ControlFlow::Exit;
                        }
                        _ => {}
                    }
                }
                WindowEvent::ModifiersChanged(mods) => {
                    self.ctrl_held = mods.control_key();
                }
                _ => {}
            },
            Event::RedrawRequested(_) => self.render_frame(),
            Event::MainEventsCleared => {
                if self.force_frames > 0 {
                    self.force_frames -= 1;
                    self.request_frame();
                }
                let fetching = self.viewport.as_ref().is_some_and(|v| v.is_fetching());
                let physics_active = self.controller.target_zoom.is_some()
                    || self.controller.pan_velocity.0.abs() > 1.0
                    || self.controller.pan_velocity.1.abs() > 1.0
                    || fetching;
                if physics_active {
                    self.request_frame();
                }
            }
            _ => {}
        }
    }
}
