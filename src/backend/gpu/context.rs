use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, RwLock};

use crate::error::Error;

pub type RegionCache = Arc<Mutex<super::cache::TieredCache>>;

/// GPU wgpu context — wraps device, queue, pipeline storage, and the
/// unified materialized-tile cache with LRU eviction.
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub pipeline_cache: Arc<RwLock<LruCache<u64, Arc<CachedPipelines>>>>,
    pub arena: super::arena::BufferArena,
    pub allocated_bytes: std::sync::atomic::AtomicU64,
    /// The unified materialized-tile cache with LRU eviction, shared across
    /// all graphs using this device.
    pub cache: RegionCache,
    /// The `max_storage_buffers_per_shader_stage` value that wgpu validates
    /// against for this device.
    ///
    /// In wgpu 29, `Device::limits()` returns adapter capabilities, not the
    /// `required_limits` passed to `request_device`. The validation is done
    /// against the requested limits, so we store them explicitly here at
    /// device-creation time instead of querying the device afterward.
    pub max_storage_buffers: u32,

    /// The compute workgroup dimension (wg_dim x wg_dim) computed from the GPU limits.
    pub wg_dim: u32,
}

/// PHASE 2: cached pipelines now hold 4 bind group layouts (one per group)
pub struct CachedPipelines {
    pub bgls: Vec<Arc<wgpu::BindGroupLayout>>,
    pub pipelines: Vec<Arc<wgpu::ComputePipeline>>,
}

/// PHASE 6: default pipeline cache size increased from 32 to 256
const DEFAULT_PIPELINE_CACHE_SIZE: usize = 256;

/// WebGPU baseline: `max_storage_buffers_per_shader_stage = 8`.
/// Used as the safe default for any externally-created device where we do not
/// control the `required_limits` (e.g. the Iced compositor device).
const WEBGPU_BASELINE_STORAGE_BUFFERS: u32 = 8;

impl GpuContext {
    pub fn new() -> Result<Arc<GpuContext>, Error> {
        pollster::block_on(Self::new_async(DEFAULT_PIPELINE_CACHE_SIZE))
    }

    /// PHASE 6: create context with a configurable pipeline cache size
    pub fn with_pipeline_cache_size(n: usize) -> Result<Arc<GpuContext>, Error> {
        pollster::block_on(Self::new_async(n))
    }

    async fn new_async(cache_size: usize) -> Result<Arc<GpuContext>, Error> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| Error::Render(format!("no GPU adapter: {e}")))?;
        let limits = adapter.limits();
        // Capture the storage-buffer limit BEFORE consuming `limits`.
        let max_storage_buffers = limits.max_storage_buffers_per_shader_stage;
        let max_invocations = limits.max_compute_invocations_per_workgroup;
        let wg_dim = if max_invocations >= 1024 { 32 } else { 16 };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fused-backend"),
                required_limits: limits,
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Render(format!("GPU device error: {e}")))?;
        let device = Arc::new(device);
        let queue = Arc::new(queue);

        // Background WGPU Garbage Collector
        // Reactive apps (like Iced) may sleep for long periods. WGPU only reclaims
        // VRAM from dropped buffers/textures when the device is polled. This ticker
        // ensures VRAM is actually returned to the OS when the user is idle.
        let device_gc = device.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let _ = device_gc.poll(wgpu::PollType::Poll);
            }
        });

        let cache_cap = NonZeroUsize::new(cache_size.max(1)).unwrap();
        let ctx = Arc::new(GpuContext {
            device,
            queue,
            pipeline_cache: Arc::new(RwLock::new(LruCache::new(cache_cap))),
            arena: super::arena::BufferArena::new(),
            allocated_bytes: std::sync::atomic::AtomicU64::new(0),
            cache: Arc::new(Mutex::new(super::cache::TieredCache::new())),
            max_storage_buffers,
            wg_dim,
        });
        ctx.cache.lock().unwrap().bind_ctx(Arc::downgrade(&ctx));
        Ok(ctx)
    }

    /// Set the GPU cache budget in bytes. Must be called after construction
    /// and before any materialization.
    pub fn with_cache_budget(self: &Arc<Self>, budget: u64) {
        self.cache.lock().unwrap().set_budget(budget);
    }

    /// Set the buffer-arena idle-pool byte budget. Buffers held in the free
    /// list beyond this are destroyed immediately.
    pub fn with_arena_budget(self: &Arc<Self>, budget: u64) {
        self.arena.set_budget(budget);
    }

    /// End the current preview/interaction generation: tiles pinned to it are no
    /// longer protected from eviction. Call on commit / tool switch. (Pinning a
    /// tile to the current generation is done via the cache's `touch_pin`.)
    pub fn bump_cache_generation(&self) {
        self.cache.lock().unwrap().bump_generation();
    }

    /// Wrap an existing wgpu device+queue (e.g. the Iced compositor device).
    ///
    /// Assumes the WebGPU baseline limit of 8 storage buffers per shader stage.
    /// We cannot query which `required_limits` were used when the device was
    /// created, and the validation error confirms the enforced limit is 8.
    pub fn from_device(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Arc<Self> {
        let device_gc = device.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let _ = device_gc.poll(wgpu::PollType::Poll);
            }
        });

        let cache_cap = NonZeroUsize::new(DEFAULT_PIPELINE_CACHE_SIZE).unwrap();
        let wg_dim = if device.limits().max_compute_invocations_per_workgroup >= 1024 {
            32
        } else {
            16
        };
        let ctx = Arc::new(GpuContext {
            device,
            queue,
            pipeline_cache: Arc::new(RwLock::new(LruCache::new(cache_cap))),
            arena: super::arena::BufferArena::new(),
            allocated_bytes: std::sync::atomic::AtomicU64::new(0),
            cache: Arc::new(Mutex::new(super::cache::TieredCache::new())),
            max_storage_buffers: WEBGPU_BASELINE_STORAGE_BUFFERS,
            wg_dim,
        });
        ctx.cache.lock().unwrap().bind_ctx(Arc::downgrade(&ctx));
        ctx
    }

    pub fn max_texture_dim(&self) -> u32 {
        self.device.limits().max_texture_dimension_2d
    }

    pub fn max_storage_buffer_binding_size(&self) -> u64 {
        let limits = self.device.limits();
        limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
    }

    /// Maximum number of source buffers allowed in a single shader pass.
    ///
    /// Group 0 binds `[source_0 .. source_n, params]`. The effective source
    /// budget is `max_storage_buffers - 1` (one slot reserved for the params
    /// buffer).
    pub fn max_sources_per_pass(&self) -> usize {
        self.max_storage_buffers.saturating_sub(1) as usize
    }
}
