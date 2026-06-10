//! GPU source — provides input pixels to the fused graph.
//!
//! Variants:
//! * `Image2D`     — already in VRAM (e.g. output of a previous pipeline).
//! * `VipsImage` — decoded on CPU via libvips; uploaded on demand.

use std::sync::Arc;

use crate::backend::vips::VipsBackend;
use crate::color::space::ColorSpace;
use crate::data::image::Image2D;
use crate::pixel::PixelFormat;
use enum_dispatch::enum_dispatch;

use super::buffer::ImageBuffer;
use super::context::GpuContext;
use crate::error::Error;
use crate::geometry::Rect;

#[enum_dispatch]
pub trait AnyGpuSource {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn format(&self) -> PixelFormat;
    fn color_space(&self) -> ColorSpace;
    fn fetch_region(
        &self,
        rect: Rect,
        lod: crate::backend::gpu::Lod,
        ctx: &Arc<GpuContext>,
    ) -> Result<Arc<ImageBuffer>, Error>;
}

// ── ImageBufferSource ────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ImageBufferSource {
    pub buffer: Arc<ImageBuffer>,
    pub ctx: Arc<GpuContext>,
    /// Bounding rect of this buffer in **full-resolution** (LOD=0) image space.
    /// For ordinary buffers: `(0, 0, buffer.width, buffer.height)`.
    /// For staging-cut buffers captured at LOD k: `cut_rect * (1 << k)`.
    pub image_rect: Rect,
}

impl AnyGpuSource for ImageBufferSource {
    /// Full-resolution width (from `image_rect`).
    fn width(&self) -> u32 {
        self.image_rect.width as u32
    }
    /// Full-resolution height (from `image_rect`).
    fn height(&self) -> u32 {
        self.image_rect.height as u32
    }
    fn format(&self) -> PixelFormat {
        self.buffer.format()
    }
    fn color_space(&self) -> ColorSpace {
        self.buffer.color_space()
    }

    fn fetch_region(
        &self,
        rect: Rect,
        lod: crate::backend::gpu::Lod,
        ctx: &Arc<GpuContext>,
    ) -> Result<Arc<ImageBuffer>, Error> {
        let scale = 1u32 << lod.0;

        let lod_origin_x = self.image_rect.x / scale as i32;
        let lod_origin_y = self.image_rect.y / scale as i32;
        let lod_w = (self.image_rect.width as u32).div_ceil(scale).max(1);
        let lod_h = (self.image_rect.height as u32).div_ceil(scale).max(1);

        let local_x = rect.x - lod_origin_x;
        let local_y = rect.y - lod_origin_y;

        if self.buffer.width == lod_w && self.buffer.height == lod_h {
            let local_rect = Rect::new(local_x, local_y, rect.width, rect.height);
            let full = self.buffer.full_rect();
            if local_rect == full {
                return Ok(self.buffer.clone());
            }
            return self.buffer.copy_region(local_rect, ctx);
        }

        let src_rect = Rect::new(
            local_x * scale as i32,
            local_y * scale as i32,
            rect.width * scale as i32,
            rect.height * scale as i32,
        );
        let src_sub = self.buffer.copy_region(src_rect, ctx)?;

        let src_image =
            crate::data::image::Image2D::<crate::backend::gpu::GpuBackend>::new_from_source(
                &GpuSource::new_buffer(src_sub, ctx.clone()),
            )?;
        let shrink = crate::operation::geometry::ShrinkOperation {
            horizontal: scale as f64,
            vertical: scale as f64,
            ceil: None,
        };
        let shrunk = src_image.execute(&shrink)?;

        let target = crate::target::ImageTarget::new(shrunk);
        let mat = target.pull(Rect::new(0, 0, rect.width, rect.height), 0)?;

        let meta = crate::pixel::PixelMeta::new(
            self.buffer.format(),
            self.buffer.color_space(),
            crate::pixel::AlphaPolicy::Straight,
        );
        Ok(ImageBuffer::from_raw(
            mat.buffer.buffer.clone(),
            rect.width as u32,
            rect.height as u32,
            meta,
        ))
    }
}

// ── VipsImageSource ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct VipsImageSource {
    pub image: Image2D<VipsBackend>,
    pub ctx: Arc<GpuContext>,
}

impl AnyGpuSource for VipsImageSource {
    fn width(&self) -> u32 {
        self.image.width() as u32
    }
    fn height(&self) -> u32 {
        self.image.height() as u32
    }
    fn format(&self) -> PixelFormat {
        gpu_safe_format(self.image.pixel_format())
    }
    fn color_space(&self) -> ColorSpace {
        self.image.pixel_meta().color_space
    }

    fn fetch_region(
        &self,
        rect: Rect,
        lod: crate::backend::gpu::Lod,
        ctx: &Arc<GpuContext>,
    ) -> Result<Arc<ImageBuffer>, Error> {
        let _sw = crate::utils::Stopwatch::new("gpu.fetch_vips");
        let mut img = gpu_safe(self.image.clone())?;

        if lod.0 > 0 {
            use crate::operation::geometry::ShrinkOperation;
            let shrink_op = ShrinkOperation {
                horizontal: (1 << lod.0) as f64,
                vertical: (1 << lod.0) as f64,
                ceil: None,
            };
            img = img.execute(&shrink_op)?;
        }

        let target = crate::target::ImageTarget::new(img);
        let mat = target
            .pull(rect, 0)
            .map_err(|e| Error::Render(format!("VIPS pull error: {e}")))?;

        let bytes = mat.buffer;
        let meta = crate::pixel::PixelMeta::new(
            mat.meta.format,
            mat.meta.color_space,
            crate::pixel::AlphaPolicy::Straight,
        );
        ImageBuffer::upload(
            &bytes,
            mat.buffer_rect.width as u32,
            mat.buffer_rect.height as u32,
            meta,
            ctx,
        )
    }
}

// ── gpu_safe promotion ───────────────────────────────────────────────────────

fn gpu_safe_format(fmt: PixelFormat) -> PixelFormat {
    let mut safe = fmt;

    // Always inject an alpha channel if it lacks one, because the GPU
    // pipeline (e.g. OpacityOperation) assumes an alpha channel exists.
    if !safe.has_alpha() {
        safe = safe.with_alpha();
    }

    let bpp = safe.bytes_per_pixel();
    let ch = safe.channel_count();
    if bpp.is_multiple_of(4) || bpp / ch >= 2 {
        return safe;
    }

    PixelFormat::Rgba8
}

fn gpu_safe(img: Image2D<VipsBackend>) -> Result<Image2D<VipsBackend>, Error> {
    let fmt = img.pixel_format();
    let safe = gpu_safe_format(fmt);
    if safe == fmt {
        return Ok(img);
    }
    let meta = crate::pixel::PixelMeta::new(
        safe,
        img.pixel_meta().color_space,
        crate::pixel::AlphaPolicy::Straight,
    );
    img.convert(meta)
}

// ── GpuSource ─────────────────────────────────────────────────────────────────

#[enum_dispatch(AnyGpuSource)]
#[derive(Clone)]
pub enum GpuSource {
    Image2D(ImageBufferSource),
    VipsImage(VipsImageSource),
}

impl GpuSource {
    /// Create a buffer source whose full-resolution rect is `(0, 0, w, h)`.
    pub fn new_buffer(buffer: Arc<ImageBuffer>, ctx: Arc<GpuContext>) -> Self {
        let image_rect = buffer.full_rect();
        GpuSource::Image2D(ImageBufferSource {
            buffer,
            ctx,
            image_rect,
        })
    }

    /// Create a buffer source whose full-resolution image-space rect is `image_rect`.
    /// Use this for staging-cut buffers captured at LOD > 0.
    pub fn new_buffer_positioned(
        buffer: Arc<ImageBuffer>,
        ctx: Arc<GpuContext>,
        image_rect: Rect,
    ) -> Self {
        GpuSource::Image2D(ImageBufferSource {
            buffer,
            ctx,
            image_rect,
        })
    }

    /// Create a vips image source. Kept as `new_vips` for backward compatibility;
    /// will be renamed to `new_vips_image` in a future release.
    pub fn new_vips(image: Image2D<VipsBackend>, ctx: Arc<GpuContext>) -> Self {
        GpuSource::VipsImage(VipsImageSource { image, ctx })
    }
}

/// Stable identity of a graph source, for content-addressed caching.
///
/// Uses in-memory object identity (the source buffer / vips image pointer),
/// which is *shared* across graph forks (a clone shares the same `Arc` / refs
/// the same libvips object) and *differs* across distinct sources. Contents are
/// never hashed — only identity.
pub fn source_identity(source: &GpuSource) -> u64 {
    match source {
        GpuSource::Image2D(ibs) => Arc::as_ptr(&ibs.buffer) as *const () as usize as u64,
        GpuSource::VipsImage(vis) => vis.image.vips_ptr() as usize as u64,
    }
}

use super::graph::Graph;
use super::handle::GraphNodeHandle;
use crate::backend::SourceInput;
use std::sync::Mutex;
// ── SourceInput ───────────────────────────────────────────────────────────────

use super::GpuBackend;
impl SourceInput for GpuBackend {
    type Source = GpuSource;

    fn open_source(source: &GpuSource) -> Result<GraphNodeHandle, crate::Error> {
        let ctx = match source {
            GpuSource::Image2D(b) => b.ctx.clone(),
            GpuSource::VipsImage(v) => v.ctx.clone(),
        };

        let mut graph = Graph::new();
        let source_id = graph.add_source(Arc::new(source.clone()));

        Ok(GraphNodeHandle {
            graph: Arc::new(Mutex::new(graph)),
            root_id: source_id,
            ctx,
        })
    }
}
