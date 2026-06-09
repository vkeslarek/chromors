use std::sync::Arc;

use crate::backend::gpu::buffer::ImageBuffer;
use crate::backend::gpu::source::GpuSource;
use crate::backend::gpu::{GpuBackend, GpuContext};
use crate::color::space::ColorSpace;
use crate::data::image::Image;
use crate::pixel::{AlphaPolicy, PixelFormat, PixelMeta};

use vello::Scene;
use vello::peniko::Color;

use super::graphics::VectorGraphics;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorAntialiasing {
    Area,
    Msaa8,
    Msaa16,
}

impl From<VectorAntialiasing> for vello::AaConfig {
    fn from(aa: VectorAntialiasing) -> Self {
        match aa {
            VectorAntialiasing::Area => vello::AaConfig::Area,
            VectorAntialiasing::Msaa8 => vello::AaConfig::Msaa8,
            VectorAntialiasing::Msaa16 => vello::AaConfig::Msaa16,
        }
    }
}

pub struct RasterConfig {
    pub width: u32,
    pub height: u32,
    pub color_space: ColorSpace,
    pub antialiasing: VectorAntialiasing,
}

pub struct GpuVectorGraphicsSource {
    pub config: RasterConfig,
}

impl GpuVectorGraphicsSource {
    pub fn new(config: RasterConfig) -> Self {
        Self { config }
    }

    pub fn rasterize(
        &self,
        graphics: &dyn VectorGraphics,
        ctx: &Arc<GpuContext>,
    ) -> Result<Image<GpuBackend>, crate::error::Error> {
        let width = self.config.width;
        let height = self.config.height;
        // Force width to be a multiple of 64 so width * 4 is a multiple of 256 (wgpu COPY_BYTES_PER_ROW_ALIGNMENT)
        let padded_w = (width + 63) & !63;

        let gpu = &ctx.device;
        let queue = &ctx.queue;

        let mut renderer = vello::Renderer::new(
            gpu,
            vello::RendererOptions {
                use_cpu: false,
                antialiasing_support: vello::AaSupport::all(),
                num_init_threads: None,
                pipeline_cache: None,
            },
        )
        .expect("Failed to create Vello renderer");

        let mut scene = Scene::new();
        graphics.draw(&mut scene);

        let staging = gpu.create_texture(&wgpu::TextureDescriptor {
            label: Some("VelloRasterStaging"),
            size: wgpu::Extent3d {
                width: padded_w,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let view = staging.create_view(&Default::default());

        renderer
            .render_to_texture(
                gpu,
                queue,
                &scene,
                &view,
                &vello::RenderParams {
                    base_color: Color::TRANSPARENT,
                    width: padded_w,
                    height,
                    antialiasing_method: self.config.antialiasing.into(),
                },
            )
            .expect("Failed to render Vello scene to texture");

        // Convert the Texture into a Buffer for Pixors Engine
        let meta = PixelMeta {
            format: PixelFormat::Rgba8,
            color_space: self.config.color_space,
            alpha_policy: AlphaPolicy::Straight,
        };

        let gpu_buffer = ImageBuffer::alloc(padded_w, height, meta, ctx);

        let mut encoder = gpu.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &staging,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: gpu_buffer.buffer.buffer.as_ref(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_w * 4),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width: padded_w,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(std::iter::once(encoder.finish()));

        let source = GpuSource::new_buffer(gpu_buffer, ctx.clone());
        Image::<GpuBackend>::new_from_source(&source)
    }
}
