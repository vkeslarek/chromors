use std::sync::Arc;

use tao::{
    dpi::PhysicalSize,
    event::{ElementState, Event, MouseButton, MouseScrollDelta, StartCause, WindowEvent},
    event_loop::ControlFlow,
    keyboard::KeyCode,
    window::WindowBuilder,
};

use vello::kurbo::Point;
use vello::{RenderParams, RendererOptions, Scene};

use crate::gpu::GpuState;
use poc::backend::gpu::GpuContext;

use crate::editor::compile::{EvalCache, evaluate};
use crate::editor::graph::{NodeGraph, PortAddr, Side};
use crate::editor::params::ParamValue;
use crate::editor::registry::NodeKindId;
use crate::ui::canvas::NodeCanvas;

const INITIAL_W: u32 = 1024;
const INITIAL_H: u32 = 768;

pub enum InteractionState {
    None,
    DraggingNode {
        node: crate::editor::graph::NodeKey,
        offset: vello::kurbo::Vec2,
    },
    DraggingWire {
        from: PortAddr,
    },
}

pub struct ImageViewerApp {
    pub gpu: Option<GpuState>,
    pub window: Option<Arc<tao::window::Window>>,
    pub gpu_ctx: Option<Arc<GpuContext>>,
    pub vello_renderer: Option<vello::Renderer>,
    pub blitter: Option<wgpu::util::TextureBlitter>,
    pub staging: Option<(wgpu::Texture, wgpu::TextureView, u32, u32)>,

    pub canvas: NodeCanvas,
    pub graph: NodeGraph,
    pub cache: Option<EvalCache>,

    pub last_cursor: Option<Point>,
    pub panning: bool,

    pub initialized: bool,
    pub interaction: InteractionState,
}

impl ImageViewerApp {
    pub fn new() -> Self {
        Self {
            gpu: None,
            window: None,
            gpu_ctx: None,
            vello_renderer: None,
            blitter: None,
            staging: None,
            canvas: NodeCanvas::new(),
            graph: NodeGraph::new(),
            cache: None,
            last_cursor: None,
            panning: false,
            initialized: false,
            interaction: InteractionState::None,
        }
    }

    pub fn run(mut self, _initial_path: Option<std::path::PathBuf>) {
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
                .with_title("Chromors Node Editor (M2)")
                .with_inner_size(PhysicalSize::new(INITIAL_W, INITIAL_H))
                .build(elwt)
                .unwrap(),
        );

        self.window = Some(window.clone());
        let gpu = GpuState::init(window);

        let vello_renderer = vello::Renderer::new(
            &gpu.device,
            RendererOptions {
                use_cpu: false,
                antialiasing_support: vello::AaSupport::area_only(),
                num_init_threads: None,
                pipeline_cache: None,
            },
        )
        .expect("Failed to create Vello renderer");

        let limits = gpu.device.limits();
        let ctx = GpuContext::from_device(
            Arc::new(gpu.device.clone()),
            Arc::new(gpu.queue.clone()),
            &limits,
        );
        self.gpu_ctx = Some(ctx.clone());

        self.blitter = Some(
            wgpu::util::TextureBlitterBuilder::new(&gpu.device, gpu.surface_config.format)
                .blend_state(wgpu::BlendState::REPLACE)
                .build(),
        );

        self.vello_renderer = Some(vello_renderer);
        self.gpu = Some(gpu);

        // M2: Hard-code two nodes
        let load = self
            .graph
            .add_node(NodeKindId("source.load"), Point::new(100.0, 100.0));
        self.graph
            .set_param(load, 0, ParamValue::Path(Some("test.png".into())));

        let exposure = self
            .graph
            .add_node(NodeKindId("color.exposure"), Point::new(400.0, 100.0));

        let _ = self.graph.connect(
            PortAddr {
                node: load,
                side: Side::Out,
                index: 0,
            },
            PortAddr {
                node: exposure,
                side: Side::In,
                index: 0,
            },
        );

        self.cache = Some(evaluate(&self.graph, &ctx));

        self.request_frame();
    }

    pub fn render_frame(&mut self) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };
        let Some(vello_renderer) = self.vello_renderer.as_mut() else {
            return;
        };

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

        let mut scene = Scene::new();
        let rect = vello::kurbo::Rect::new(
            0.0,
            0.0,
            gpu.surface_config.width as f64,
            gpu.surface_config.height as f64,
        );

        // Solid background
        scene.fill(
            vello::peniko::Fill::NonZero,
            vello::kurbo::Affine::IDENTITY,
            crate::ui::theme::COL_BG,
            None,
            &rect,
        );

        let mut temp_wire = None;
        if let InteractionState::DraggingWire { from } = self.interaction {
            if let Some(cursor) = self.last_cursor {
                temp_wire = Some((from, self.canvas.camera.p2g(cursor)));
            }
        }

        self.canvas.draw(
            &mut self.graph,
            self.cache.as_ref().unwrap(),
            rect,
            &mut scene,
            temp_wire,
        );

        let width = gpu.surface_config.width.max(1);
        let height = gpu.surface_config.height.max(1);

        let needs_resize = self
            .staging
            .as_ref()
            .is_none_or(|s| s.2 != width || s.3 != height);
        if needs_resize {
            let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("vello_staging"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let staging_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.staging = Some((texture, staging_view, width, height));
        }

        let staging_view = &self.staging.as_ref().unwrap().1;

        vello_renderer
            .render_to_texture(
                &gpu.device,
                &gpu.queue,
                &scene,
                staging_view,
                &vello::RenderParams {
                    base_color: vello::peniko::Color::BLACK,
                    width,
                    height,
                    antialiasing_method: vello::AaConfig::Area,
                },
            )
            .expect("vello render failed");

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("blit"),
            });
        if let Some(blitter) = self.blitter.as_ref() {
            blitter.copy(&gpu.device, &mut encoder, staging_view, &view);
        }
        gpu.queue.submit(std::iter::once(encoder.finish()));

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
            Event::NewEvents(StartCause::Init) => self.init(elwt),
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(size) => {
                    if let Some(gpu) = self.gpu.as_mut() {
                        gpu.reconfigure(size.width, size.height);
                    }
                    self.request_frame();
                }
                WindowEvent::CursorMoved { position, .. } => {
                    let cursor = Point::new(position.x, position.y);
                    if self.panning {
                        if let Some(last) = self.last_cursor {
                            let delta = cursor - last;
                            self.canvas.camera.pan -= delta / self.canvas.camera.zoom;
                            self.request_frame();
                        }
                    } else if let InteractionState::DraggingNode { node, offset } = self.interaction
                    {
                        let g_pos = self.canvas.camera.p2g(cursor);
                        if let Some(n) = self.graph.nodes.get_mut(node) {
                            n.pos = g_pos + offset;
                            n.layout_cache = None;
                        }
                        self.request_frame();
                    } else if let InteractionState::DraggingWire { .. } = self.interaction {
                        self.request_frame();
                    }
                    self.last_cursor = Some(cursor);
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if button == MouseButton::Left {
                        if state == ElementState::Pressed {
                            let cursor = self.last_cursor.unwrap_or(Point::new(0.0, 0.0));
                            let g_pos = self.canvas.camera.p2g(cursor);
                            use crate::ui::canvas::HitResult;
                            match self.canvas.hit_test(&self.graph, g_pos) {
                                HitResult::NodeTitle(key) => {
                                    if let Some(node) = self.graph.nodes.get(key) {
                                        let offset = node.pos - g_pos;
                                        self.interaction = InteractionState::DraggingNode {
                                            node: key,
                                            offset: vello::kurbo::Vec2::new(offset.x, offset.y),
                                        };
                                    }
                                }
                                HitResult::Socket(addr) => {
                                    self.interaction =
                                        InteractionState::DraggingWire { from: addr };
                                }
                                HitResult::None => {}
                            }
                        } else {
                            if let InteractionState::DraggingWire { from } = self.interaction {
                                let cursor = self.last_cursor.unwrap_or(Point::new(0.0, 0.0));
                                let g_pos = self.canvas.camera.p2g(cursor);
                                use crate::ui::canvas::HitResult;
                                if let HitResult::Socket(to) =
                                    self.canvas.hit_test(&self.graph, g_pos)
                                {
                                    if from.node != to.node && from.side != to.side {
                                        // Connect Out to In
                                        let (src, dst) = if from.side == Side::Out {
                                            (from, to)
                                        } else {
                                            (to, from)
                                        };
                                        let _ = self.graph.connect(src, dst);
                                        if let Some(ctx) = &self.gpu_ctx {
                                            self.cache = Some(evaluate(&self.graph, ctx));
                                        }
                                    }
                                }
                            }
                            self.interaction = InteractionState::None;
                        }
                        self.request_frame();
                    }
                    if button == MouseButton::Middle {
                        self.panning = state == ElementState::Pressed;
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let dy = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y as f64 * 20.0,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f64,
                        _ => 0.0,
                    };
                    if let Some(cursor) = self.last_cursor {
                        let g_before = self.canvas.camera.p2g(cursor);
                        let zoom_factor = 1.05_f64.powf(dy / 20.0);
                        self.canvas.camera.zoom *= zoom_factor;
                        let g_after = self.canvas.camera.p2g(cursor);
                        self.canvas.camera.pan -= g_after - g_before;
                        self.request_frame();
                    }
                }
                WindowEvent::KeyboardInput { event: ev, .. } => {
                    if ev.state == ElementState::Pressed && ev.physical_key == KeyCode::KeyQ {
                        *control_flow = ControlFlow::Exit;
                    }
                }
                _ => {}
            },
            Event::RedrawRequested(_) => self.render_frame(),
            Event::MainEventsCleared => {
                // Not continuously rendering in M2, only on input
            }
            _ => {}
        }
    }
}
