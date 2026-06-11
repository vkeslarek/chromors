use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use wgpu::Buffer;

/// Default byte budget for buffers held idle in the free list before they are
/// destroyed and the VRAM returned to the OS.
const DEFAULT_ARENA_BUDGET: u64 = 512 * 1024 * 1024;

/// Recycling pool for transient compute buffers (temps, params, outputs).
///
/// Freed buffers are kept around for reuse by a same-or-larger allocation
/// (within 2x), FIFO order. Once the idle pool exceeds `budget` bytes, the
/// oldest buffers are destroyed until it fits again — bounding the VRAM held
/// by buffers nothing currently references.
pub struct BufferArena {
    free_buffers: Mutex<VecDeque<Buffer>>,
    free_bytes: AtomicU64,
    budget: AtomicU64,
}

impl BufferArena {
    pub fn new() -> Self {
        Self::with_budget(DEFAULT_ARENA_BUDGET)
    }

    pub fn with_budget(budget: u64) -> Self {
        Self {
            free_buffers: Mutex::new(VecDeque::new()),
            free_bytes: AtomicU64::new(0),
            budget: AtomicU64::new(budget),
        }
    }

    /// Set the idle-pool byte budget and immediately evict down to it.
    pub fn set_budget(&self, budget: u64) {
        self.budget.store(budget, Ordering::Relaxed);
        let mut lock = self.free_buffers.lock().unwrap();
        self.evict_to_budget(&mut lock);
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
            let buf = lock.remove(idx).unwrap();
            self.free_bytes.fetch_sub(buf.size(), Ordering::Relaxed);
            buf
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
        let size = buffer.size();
        let mut lock = self.free_buffers.lock().unwrap();
        lock.push_back(buffer);
        self.free_bytes.fetch_add(size, Ordering::Relaxed);
        self.evict_to_budget(&mut lock);
    }

    /// Destroy oldest idle buffers until `free_bytes <= budget`.
    fn evict_to_budget(&self, lock: &mut VecDeque<Buffer>) {
        let budget = self.budget.load(Ordering::Relaxed);
        while self.free_bytes.load(Ordering::Relaxed) > budget {
            let Some(buf) = lock.pop_front() else { break };
            self.free_bytes.fetch_sub(buf.size(), Ordering::Relaxed);
            buf.destroy();
        }
    }
}
