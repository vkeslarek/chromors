use crate::camera::CameraUniform;

const SHADER: &str = r#"
struct Camera {
    vp_w: f32, vp_h: f32,
    pan_x: f32, pan_y: f32, zoom: f32,
    _pad0: f32, _pad1: f32, _pad2: f32,
};

struct Layer {
    img_w: f32, img_h: f32,
    img_w0: f32, img_h0: f32,
    pos_x: f32, pos_y: f32,
    scale_x: f32, scale_y: f32,
    atlas_w: f32, atlas_h: f32,
    mip_level: f32, opacity: f32,
};

@group(0) @binding(0) var<uniform> cam: Camera;
@group(1) @binding(0) var t: texture_2d<f32>;
@group(1) @binding(1) var s: sampler;
@group(1) @binding(2) var<uniform> layer: Layer;

struct VSOut { 
    @builtin(position) pos: vec4f, 
    @location(0) uv: vec2f,
    @location(1) uv_min: vec2f,
    @location(2) uv_max: vec2f,
    @location(3) img_xy_mip0: vec2f,
};

@vertex fn vs_main(
    @builtin(vertex_index) vi: u32,
    @location(0) tx: u32,
    @location(1) ty: u32,
    @location(2) sx: u32,
    @location(3) sy: u32,
) -> VSOut {
    let vx = f32(vi & 1u);
    let vy = f32((vi >> 1u) & 1u);
    
    let TILE = 256.0;
    
    let tile_w = min(TILE, layer.img_w - f32(tx) * TILE);
    let tile_h = min(TILE, layer.img_h - f32(ty) * TILE);
    
    let img_x = f32(tx) * TILE + vx * tile_w;
    let img_y = f32(ty) * TILE + vy * tile_h;
    
    let scale = layer.img_w0 / layer.img_w;
    let img_xy_mip0 = vec2f(img_x, img_y) * scale;
    
    let transformed = vec2f(
        layer.pos_x + img_xy_mip0.x * layer.scale_x,
        layer.pos_y + img_xy_mip0.y * layer.scale_y
    );
    
    let screen = (transformed - vec2f(cam.pan_x, cam.pan_y)) * cam.zoom;
    
    let ndc_x = (screen.x / cam.vp_w) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen.y / cam.vp_h) * 2.0;
    
    var out: VSOut;
    out.pos = vec4f(ndc_x, ndc_y, 0.0, 1.0);
    out.img_xy_mip0 = img_xy_mip0;
    
    let pad = 0.5;
    out.uv_min = vec2f(f32(sx) + pad, f32(sy) + pad) / vec2f(layer.atlas_w, layer.atlas_h);
    out.uv_max = vec2f(f32(sx) + tile_w - pad, f32(sy) + tile_h - pad) / vec2f(layer.atlas_w, layer.atlas_h);
    
    let atlas_u = (f32(sx) + vx * tile_w) / layer.atlas_w;
    let atlas_v = (f32(sy) + vy * tile_h) / layer.atlas_h;
    out.uv = vec2f(atlas_u, atlas_v);
    
    return out;
}

@fragment fn fs_main(in: VSOut) -> @location(0) vec4f {
    let uv = clamp(in.uv, in.uv_min, in.uv_max);
    var color = textureSample(t, s, uv);

    let grid_alpha = smoothstep(4.0, 10.0, cam.zoom);
    if grid_alpha > 0.001 {
        let f = fract(in.img_xy_mip0);
        let pixel_size = 1.0 / cam.zoom;
        let dist_x = min(f.x, 1.0 - f.x);
        let dist_y = min(f.y, 1.0 - f.y);
        let dist = min(dist_x, dist_y);
        
        let line_w = 0.75 * pixel_size; 
        let edge = smoothstep(line_w + pixel_size, line_w - pixel_size, dist);
        
        if edge > 0.0 {
            color = mix(color, vec4f(0.15, 0.15, 0.16, 1.0), grid_alpha * 0.55 * edge);
        }
    }

    // `color` is the straight RGB texture color. `color.a` is the straight alpha.
    let final_alpha = color.a * layer.opacity;

    let checker_size = 16.0;
    let px = floor(in.pos.x / checker_size);
    let py = floor(in.pos.y / checker_size);
    let is_white = (u32(px) + u32(py)) % 2u == 0u;
    let checker_gray = select(0.75, 0.85, is_white);
    let checker = vec3f(checker_gray, checker_gray, checker_gray);

    // Alpha blend the straight texture color over the opaque checkerboard
    let final_rgb = mix(checker, color.rgb, final_alpha);
    
    // Output with alpha 1.0 because the checkerboard is opaque
    color = vec4f(final_rgb, 1.0);
    
    return color;
}
"#;

pub struct ViewportPipeline {
    pub render_pipeline: wgpu::RenderPipeline,
    pub camera_buf: wgpu::Buffer,
    pub camera_bgl: wgpu::BindGroupLayout,
    pub atlas_bgl: wgpu::BindGroupLayout,
    pub atlas_sampler_linear: wgpu::Sampler,
    pub atlas_sampler_nearest: wgpu::Sampler,
    pub tex_format: wgpu::TextureFormat,
}

impl ViewportPipeline {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let atlas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atlas_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("layout"),
            bind_group_layouts: &[Some(&camera_bgl), Some(&atlas_bgl)],
            immediate_size: 0,
        });

        let tex_format = if format.is_srgb() {
            wgpu::TextureFormat::Rgba8UnormSrgb
        } else {
            wgpu::TextureFormat::Rgba8Unorm
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("viewport_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[u32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 4,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 8,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 12,
                            shader_location: 3,
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
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera_buf"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let atlas_sampler_linear = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let atlas_sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            render_pipeline,
            camera_buf,
            camera_bgl,
            atlas_bgl,
            atlas_sampler_linear,
            atlas_sampler_nearest,
            tex_format,
        }
    }
}
