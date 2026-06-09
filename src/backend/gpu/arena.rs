use std::sync::Mutex;
use wgpu::Buffer;

pub struct BufferArena {
    free_buffers: Mutex<Vec<Buffer>>,
}

impl BufferArena {
    pub fn new() -> Self {
        Self {
            free_buffers: Mutex::new(Vec::new()),
        }
    }

    pub fn allocate(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
        label: Option<&str>,
    ) -> Buffer {
        let mut lock = self.free_buffers.lock().unwrap();
        // find a buffer that fits
        if let Some(idx) = lock
            .iter()
            .position(|b| b.size() >= size && b.usage().contains(usage) && b.size() <= size * 2)
        {
            lock.remove(idx)
        } else {
            device.create_buffer(&wgpu::BufferDescriptor {
                label,
                size,
                usage,
                mapped_at_creation: false,
            })
        }
    }

    pub fn free(&self, buffer: Buffer) {
        let mut lock = self.free_buffers.lock().unwrap();
        if lock.len() < 128 {
            // arbitrary limit
            lock.push(buffer);
        } else {
            buffer.destroy();
        }
    }
}
