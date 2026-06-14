const SHADER: &str = r#"
struct Camera {
    vp_w: f32, vp_h: f32,
    pan_x: f32, pan_y: f32, zoom: f32,
    _pad0: f32, _pad1: f32, _pad2: f32,
};
@group(0) @binding(0) var<uniform> cam: Camera;

struct VertexInput {
    @location(0) pos: vec2f,
    @location(1) color: vec4f,
};

struct VSOut {
    @builtin(position) pos: vec4f,
    @location(0) color: vec4f,
};

@vertex fn vs_main(in: VertexInput) -> VSOut {
    let screen = (in.pos - vec2f(cam.pan_x, cam.pan_y)) * cam.zoom;
    let ndc_x = (screen.x / cam.vp_w) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen.y / cam.vp_h) * 2.0;
    
    var out: VSOut;
    out.pos = vec4f(ndc_x, ndc_y, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment fn fs_main(in: VSOut) -> @location(0) vec4f {
    return in.color;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct OverlayVertex {
    pub pos: [f32; 2],
    pub color: [f32; 4],
}

pub struct OverlayPipeline {
    pub pipeline: wgpu::RenderPipeline,
}

impl OverlayPipeline {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        camera_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay_layout"),
            bind_group_layouts: &[Some(camera_bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<OverlayVertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 8,
                            shader_location: 1,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self { pipeline }
    }
}
