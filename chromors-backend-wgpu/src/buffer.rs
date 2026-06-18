use crate::context::GpuContext;
use crate::Error;
use std::sync::Arc;
use wgpu;

/// A tracked, ref-counted VRAM buffer. Carries no metadata (payload-agnostic).
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
    /// Access the underlying wgpu buffer.
    pub fn buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.buffer
    }

    /// Create an untracked buffer (no VRAM accounting).
    pub fn from_raw(buffer: Arc<wgpu::Buffer>, byte_len: u64) -> Arc<Self> {
        Arc::new(GpuBuffer {
            buffer,
            byte_len,
            ctx: None,
        })
    }

    /// Create a buffer with VRAM accounting. On drop, subtracts the buffer's
    /// size from the context's `allocated_bytes` counter.
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

    /// Downloads the buffer contents from VRAM to CPU via a staging buffer.
    /// Blocks until the GPU copy completes.
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
            .map_err(|e| Error::Backend(format!("GPU poll error: {:?}", e)))?;
        rx.recv()
            .map_err(|_| Error::Backend("GPU readback channel closed".into()))?
            .map_err(|e| Error::Backend(format!("GPU map error: {:?}", e)))?;

        let bytes = slice.get_mapped_range()[..self.byte_len as usize].to_vec();
        Ok(bytes)
    }
}
