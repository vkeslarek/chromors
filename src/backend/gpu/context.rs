use crate::error::Error;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, RwLock};
use wgpu;

const DEFAULT_PIPELINE_CACHE_SIZE: usize = 256;

/// Compiled pipeline storage, keyed by Slang IR text hash.
pub struct CachedPipelines {
    pub bgls: Vec<Arc<wgpu::BindGroupLayout>>,
    pub pipelines: Vec<Arc<wgpu::ComputePipeline>>,
}

/// GPU wgpu context — holds the device, queue, pipeline cache, and
/// hardware-derived limits. Created once at engine init; shared across
/// all GPU operations via `Arc`.
///
/// Note: The data cache is NOT here in v2. The caller wraps Sources in `Cached`.
pub struct GpuContext {
    /// The wgpu device (Vulkan/Metal/DX12 abstraction).
    pub device: Arc<wgpu::Device>,
    /// The wgpu command queue for submitting compute dispatches.
    pub queue: Arc<wgpu::Queue>,
    /// Compiled pipeline cache keyed by Slang IR text hash.
    pub pipeline_cache: Arc<RwLock<LruCache<u64, Arc<CachedPipelines>>>>,
    /// Running total of allocated VRAM bytes, tracked for diagnostics.
    /// Updated atomically by `GpuBuffer` on creation/drop.
    pub allocated_bytes: AtomicU64,
    /// Device limit: max storage buffers per shader stage.
    /// Used by `CutFinder` to decide when to split a fused pass.
    pub max_storage_buffers: u32,
    /// Workgroup tile dimension (32 for most GPUs, 16 for low-end).
    /// Derived from `max_compute_invocations_per_workgroup`.
    pub wg_dim: u32,
}

impl GpuContext {
    pub fn new() -> Result<Arc<GpuContext>, Error> {
        pollster::block_on(Self::new_async(DEFAULT_PIPELINE_CACHE_SIZE))
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
            .map_err(|e| Error::Backend(format!("no GPU adapter: {:?}", e)))?;

        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fused-backend"),
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| Error::Backend(format!("GPU device error: {:?}", e)))?;

        Ok(Self::from_device_with_cache(
            Arc::new(device),
            Arc::new(queue),
            &limits,
            cache_size,
        ))
    }

    /// Builds a `GpuContext` around an externally-created device/queue (e.g.
    /// a windowed app's surface device), so the DAG and the presentation
    /// pipeline share one device and can pass `wgpu::Buffer`/`Texture`
    /// directly without cross-device copies.
    pub fn from_device(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        limits: &wgpu::Limits,
    ) -> Arc<GpuContext> {
        Self::from_device_with_cache(device, queue, limits, DEFAULT_PIPELINE_CACHE_SIZE)
    }

    fn from_device_with_cache(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        limits: &wgpu::Limits,
        cache_size: usize,
    ) -> Arc<GpuContext> {
        let max_storage_buffers = limits.max_storage_buffers_per_shader_stage;
        let max_invocations = limits.max_compute_invocations_per_workgroup;
        let wg_dim = if max_invocations >= 1024 { 32 } else { 16 };

        let device_gc = device.clone();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let _ = device_gc.poll(wgpu::PollType::Poll);
            }
        });

        let cache_cap = NonZeroUsize::new(cache_size.max(1)).unwrap();
        Arc::new(GpuContext {
            device,
            queue,
            pipeline_cache: Arc::new(RwLock::new(LruCache::new(cache_cap))),
            allocated_bytes: AtomicU64::new(0),
            max_storage_buffers,
            wg_dim,
        })
    }
}
