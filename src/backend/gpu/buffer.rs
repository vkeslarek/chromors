//! GPU buffer — a tracked, ref-counted, evictable VRAM allocation.
//!
//! [`GpuBuffer`] is payload-agnostic: it carries no pixel format, dimensions,
//! or color space. Image-specific concerns live on [`ImageBuffer`].

use std::sync::Arc;

use crate::color::space::ColorSpace;
use crate::pixel::{PixelFormat, PixelMeta};

use super::context::GpuContext;
use crate::error::Error;
use crate::geometry::Rect;

// ── GpuBuffer (payload-agnostic VRAM allocation) ─────────────────────────────

/// A tracked, ref-counted VRAM buffer. Carries no pixel metadata — use
/// [`ImageBuffer`] for image data, or a typed view for histograms / point
/// lists / feature maps.
pub struct GpuBuffer {
    pub buffer: Arc<wgpu::Buffer>,
    pub byte_len: u64,
    ctx: Option<Arc<GpuContext>>,
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("byte_len", &self.byte_len)
            .field("aligned", &self.buffer.size())
            .finish_non_exhaustive()
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        if let Some(ctx) = &self.ctx {
            ctx.allocated_bytes
                .fetch_sub(self.buffer.size(), std::sync::atomic::Ordering::Relaxed);
        }
    }
}

impl GpuBuffer {
    pub fn buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.buffer
    }

    /// Wrap a pre-existing wgpu buffer with known byte length.
    /// Does not increment `allocated_bytes` — the caller is responsible for
    /// tracking.
    pub fn from_raw(buffer: Arc<wgpu::Buffer>, byte_len: u64) -> Arc<Self> {
        Arc::new(GpuBuffer {
            buffer,
            byte_len,
            ctx: None,
        })
    }

    /// Wrap a pre-existing wgpu buffer and tracks it against `ctx`.
    pub fn from_raw_tracked(
        buffer: Arc<wgpu::Buffer>,
        byte_len: u64,
        ctx: &Arc<GpuContext>,
    ) -> Arc<Self> {
        ctx.allocated_bytes
            .fetch_add(buffer.size(), std::sync::atomic::Ordering::Relaxed);
        Arc::new(GpuBuffer {
            buffer,
            byte_len,
            ctx: Some(ctx.clone()),
        })
    }

    /// Read back the entire buffer to CPU bytes. Blocks until GPU work is done.
    pub fn read_to_cpu(&self, ctx: &GpuContext) -> Result<Vec<u8>, Error> {
        let size = self.buffer.size();
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuBuffer::readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, size);
        ctx.queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Render(format!("GPU poll error: {e:?}")))?;
        rx.recv()
            .map_err(|_| Error::Render("GPU readback channel closed".into()))?
            .map_err(|e| Error::Render(format!("GPU map error: {e:?}")))?;

        let bytes = slice.get_mapped_range()[..self.byte_len as usize].to_vec();
        Ok(bytes)
    }
}

// ── ImageBuffer (image view over a GpuBuffer) ────────────────────────────────

/// A 2-D image buffer in VRAM — tightly packed, row-major.
///
/// Wraps a [`GpuBuffer`] with image-specific metadata: dimensions, pixel format,
/// and color space.
pub struct ImageBuffer {
    pub buffer: Arc<GpuBuffer>,
    pub width: u32,
    pub height: u32,
    pub meta: PixelMeta,
}

impl std::fmt::Debug for ImageBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageBuffer")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("meta", &self.meta)
            .finish_non_exhaustive()
    }
}

impl ImageBuffer {
    pub fn format(&self) -> PixelFormat {
        self.meta.format
    }

    pub fn color_space(&self) -> ColorSpace {
        self.meta.color_space
    }

    pub fn bytes_per_pixel(&self) -> u32 {
        self.meta.format.bytes_per_pixel() as u32
    }

    pub fn row_bytes(&self) -> u64 {
        self.width as u64 * self.bytes_per_pixel() as u64
    }

    pub fn total_bytes(&self) -> u64 {
        self.height as u64 * self.row_bytes()
    }

    pub fn full_rect(&self) -> Rect {
        Rect::new(0, 0, self.width as i32, self.height as i32)
    }

    // ── Constructors ────────────────────────────────────────────────────

    /// Upload `data` (row-major CPU bytes) into a new GPU image buffer.
    pub fn upload(
        data: &[u8],
        width: u32,
        height: u32,
        meta: PixelMeta,
        ctx: &Arc<GpuContext>,
    ) -> Result<Arc<Self>, Error> {
        let bpp = meta.format.bytes_per_pixel() as u32;
        let expected = (width * height * bpp) as usize;
        if data.len() < expected {
            return Err(Error::Render(format!(
                "ImageBuffer::upload: need {expected} bytes, got {}",
                data.len()
            )));
        }
        let aligned_size = ((expected + 3) & !3) as u64;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ImageBuffer::upload"),
            size: aligned_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer
            .slice(..expected as u64)
            .get_mapped_range_mut()
            .copy_from_slice(&data[..expected]);
        buffer.unmap();
        ctx.allocated_bytes
            .fetch_add(aligned_size, std::sync::atomic::Ordering::Relaxed);
        let gpu_buf = GpuBuffer::from_raw(Arc::new(buffer), expected as u64);
        Ok(Arc::new(ImageBuffer {
            buffer: gpu_buf,
            width,
            height,
            meta,
        }))
    }

    /// Allocate an uninitialised GPU image buffer of `width × height`.
    pub fn alloc(width: u32, height: u32, meta: PixelMeta, ctx: &Arc<GpuContext>) -> Arc<Self> {
        let bpp = meta.format.bytes_per_pixel() as u32;
        let size = (width as u64 * height as u64 * bpp as u64 + 3) & !3;
        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ImageBuffer::alloc"),
            size: size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.allocated_bytes
            .fetch_add(size.max(4), std::sync::atomic::Ordering::Relaxed);
        let byte_len = width as u64 * height as u64 * bpp as u64;
        Arc::new(ImageBuffer {
            buffer: Arc::new(GpuBuffer {
                buffer: Arc::new(buffer),
                byte_len,
                ctx: Some(ctx.clone()),
            }),
            width,
            height,
            meta,
        })
    }

    /// Wrap a pre-existing wgpu buffer as an image buffer.
    pub fn from_raw(
        buffer: Arc<wgpu::Buffer>,
        width: u32,
        height: u32,
        meta: PixelMeta,
    ) -> Arc<Self> {
        let bpp = meta.format.bytes_per_pixel() as u64;
        let byte_len = width as u64 * height as u64 * bpp;
        Arc::new(ImageBuffer {
            buffer: GpuBuffer::from_raw(buffer, byte_len),
            width,
            height,
            meta,
        })
    }

    // ── Sub-region copy ─────────────────────────────────────────────────

    /// GPU-side copy of `rect` into a new, tightly-packed image buffer.
    pub fn copy_region(&self, rect: Rect, ctx: &Arc<GpuContext>) -> Result<Arc<Self>, Error> {
        let rect = rect.clamp(self.full_rect());
        if rect.is_empty() {
            return Err(Error::Render("copy_region: rect is empty".into()));
        }
        let w = rect.width as u32;
        let h = rect.height as u32;
        let bpp = self.bytes_per_pixel() as u64;
        let dst_size = ((w as u64 * h as u64 * bpp) + 3) & !3;

        let dst_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ImageBuffer::copy_region"),
            size: dst_size.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let src_row_bytes = self.row_bytes();
        let dst_row_bytes = w as u64 * bpp;

        for row in 0..h {
            let src_offset = (rect.y as u64 + row as u64) * src_row_bytes + rect.x as u64 * bpp;
            let dst_offset = row as u64 * dst_row_bytes;
            encoder.copy_buffer_to_buffer(
                &self.buffer.buffer,
                src_offset,
                &dst_buffer,
                dst_offset,
                dst_row_bytes,
            );
        }
        ctx.queue.submit(std::iter::once(encoder.finish()));
        let _ = ctx.device.poll(wgpu::PollType::Poll);
        ctx.allocated_bytes
            .fetch_add(dst_size.max(4), std::sync::atomic::Ordering::Relaxed);

        let byte_len = w as u64 * h as u64 * bpp;
        Ok(Arc::new(ImageBuffer {
            buffer: Arc::new(GpuBuffer {
                buffer: Arc::new(dst_buffer),
                byte_len,
                ctx: Some(ctx.clone()),
            }),
            width: w,
            height: h,
            meta: self.meta,
        }))
    }

    // ── CPU readback ────────────────────────────────────────────────────

    /// Read a sub-rect back to CPU bytes. Blocks until GPU work is done.
    pub fn read_subrect_to_cpu(
        &self,
        rect: Rect,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<u8>, Error> {
        let rect = rect.clamp(self.full_rect());
        if rect.is_empty() {
            return Ok(Vec::new());
        }
        let w = rect.width as u32;
        let h = rect.height as u32;
        let bpp = self.bytes_per_pixel() as u64;
        let dst_size = ((w as u64 * h as u64 * bpp) + 3) & !3;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ImageBuffer::read_subrect"),
            size: dst_size.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        let src_row_bytes = self.row_bytes();
        let dst_row_bytes = w as u64 * bpp;

        for row in 0..h {
            let src_offset = (rect.y as u64 + row as u64) * src_row_bytes + rect.x as u64 * bpp;
            let dst_offset = row as u64 * dst_row_bytes;
            encoder.copy_buffer_to_buffer(
                &self.buffer.buffer,
                src_offset,
                &staging,
                dst_offset,
                dst_row_bytes,
            );
        }
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Render(format!("GPU poll error: {e:?}")))?;
        rx.recv()
            .map_err(|_| Error::Render("GPU readback channel closed".into()))?
            .map_err(|e| Error::Render(format!("GPU map error: {e:?}")))?;

        let bytes = slice.get_mapped_range()[..(w as usize * h as usize * bpp as usize)].to_vec();
        Ok(bytes)
    }

    /// Read back the entire image buffer to CPU bytes.
    pub fn read_to_cpu(&self, ctx: &GpuContext) -> Result<Vec<u8>, Error> {
        self.buffer.read_to_cpu(ctx)
    }
}
