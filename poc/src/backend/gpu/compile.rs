use std::sync::Arc;
use crate::error::Error;
use super::{GpuContext, GpuBuilder};
use super::context::CachedPipelines;
use super::buffer::GpuBuffer;
use wgpu;

pub struct DispatchPass {
    pub bgls: Vec<Arc<wgpu::BindGroupLayout>>,
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub slang_text: String,
}

/// Content fingerprint of every `.slang` file under `shaders/`, memoized once
/// per process. Folded into the pipeline-cache key so editing a kernel (whose
/// text never appears in the emitted `main()`) still invalidates the cached
/// SPIR-V — otherwise a stale shader silently survives a source edit.
fn shader_fingerprint() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::OnceLock;
    static FP: OnceLock<u64> = OnceLock::new();
    *FP.get_or_init(|| {
        let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
        let dir = std::path::PathBuf::from(&manifest).join("shaders");
        let mut files = Vec::new();
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            if let Ok(rd) = std::fs::read_dir(dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    if p.is_dir() {
                        walk(&p, out);
                    } else if p.extension().and_then(|s| s.to_str()) == Some("slang") {
                        out.push(p);
                    }
                }
            }
        }
        walk(&dir, &mut files);
        files.sort(); // stable order → stable hash
        let mut h = DefaultHasher::new();
        for f in files {
            f.hash(&mut h);
            if let Ok(bytes) = std::fs::read(&f) {
                bytes.hash(&mut h);
            }
        }
        h.finish()
    })
}

pub fn compile(ctx: &GpuContext, builder: &GpuBuilder, slang: String, hash: u64) -> Result<DispatchPass, Error> {
    // Fold the shader-tree fingerprint in: the emitted `main()` text alone does
    // not change when an imported kernel's body does.
    let hash = hash ^ shader_fingerprint();
    let cached = {
        let mut cache = ctx.pipeline_cache.write().unwrap();
        cache.get(&hash).cloned()
    };

    if let Some(cached) = cached {
        return Ok(DispatchPass {
            bgls: cached.bgls.clone(),
            pipeline: cached.pipelines[0].clone(),
            slang_text: slang,
        });
    }

    // 1. Emite o SPIR-V a partir do JIT String usando o SlangCompiler
    let spirv = compile_spirv(&slang, hash)?;

    // 2. Carrega o SPIR-V no WGPU (exatamente como na engine original)
    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("jit_fused_module"),
        source: wgpu::util::make_spirv(&spirv),
    });

    let mut bgl_entries = Vec::new();
    let mut binding_idx = 0;

    // Output buffer
    bgl_entries.push(wgpu::BindGroupLayoutEntry {
        binding: binding_idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    binding_idx += 1;

    // Params buffer
    bgl_entries.push(wgpu::BindGroupLayoutEntry {
        binding: binding_idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    });
    binding_idx += 1;

    // Working-space scratch buffers (sandwich ping-pong) — image outputs only.
    for _ in 0..builder.work_buffer_count() {
        bgl_entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding_idx,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        binding_idx += 1;
    }

    for _ in &builder.input_views {
        bgl_entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding_idx,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        binding_idx += 1;
    }

    let bgl0 = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bgl_targets"),
        entries: &bgl_entries,
    });

    let pl = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline_layout"),
        bind_group_layouts: &[Some(&bgl0)],
        immediate_size: 0,
    });

    // We assume the Slang entry point is always "main"
    let entry_point = "main";

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("main_compute"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    let bgls = vec![Arc::new(bgl0)];
    let pipeline = Arc::new(pipeline);

    // Salva no pipeline cache do context
    let new_cache = Arc::new(CachedPipelines {
        bgls: bgls.clone(),
        pipelines: vec![pipeline.clone()],
    });
    ctx.pipeline_cache.write().unwrap().put(hash, new_cache);

    Ok(DispatchPass { bgls, pipeline, slang_text: slang })
}

/// Invoca o compilador C++ FFI para compilar o shader JIT gerado para SPIR-V
fn compile_spirv(text: &str, hash: u64) -> Result<Vec<u8>, Error> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| {
        std::path::PathBuf::from(&manifest)
            .join("target")
            .to_str()
            .unwrap()
            .to_string()
    }));
    let shader_dir = std::path::PathBuf::from(&manifest).join("shaders");

    let compiler = super::slang::SlangCompiler::new(shader_dir, out_dir);
    compiler.compile_ir(text, hash).map_err(|e| Error::Backend(format!("Slang error: {}", e)))
}

pub fn dispatch(ctx: &GpuContext, pass: &DispatchPass, builder: &GpuBuilder, out_bytes: u64, dims: (u32, u32)) -> Result<Arc<GpuBuffer>, Error> {
    // Output buffer holds the result, GPU-resident. Sized by the agnostic
    // `AnyKind::byte_size(wu)` resolved during the demand walk.
    let byte_len = out_bytes.max(16);
    let out_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_buffer"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }));

    use wgpu::util::DeviceExt;

    let mut bg_entries = Vec::new();
    let mut binding_idx = 0;

    bg_entries.push(wgpu::BindGroupEntry {
        binding: binding_idx,
        resource: out_buffer.as_entire_binding(),
    });
    binding_idx += 1;

    let params_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: &builder.params.bytes,
        usage: wgpu::BufferUsages::STORAGE, // Matches Storage { read_only: true }
    });
    bg_entries.push(wgpu::BindGroupEntry {
        binding: binding_idx,
        resource: params_buf.as_entire_binding(),
    });
    binding_idx += 1;

    // Working scratch buffers (float4, output-sized) — image outputs only.
    let work_len = (dims.0 as u64 * dims.1 as u64 * 16).max(16);
    let work_buffers: Vec<Arc<wgpu::Buffer>> = (0..builder.work_buffer_count())
        .map(|k| {
            Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("work_{k}")),
                size: work_len,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }))
        })
        .collect();
    for work in &work_buffers {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: binding_idx,
            resource: work.as_entire_binding(),
        });
        binding_idx += 1;
    }

    for (i, _) in builder.input_views.iter().enumerate() {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: binding_idx,
            resource: builder.source_buffers[i].buffer.as_entire_binding(),
        });
        binding_idx += 1;
    }

    let bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg0"),
        layout: &pass.bgls[0],
        entries: &bg_entries,
    });

    let wg = ctx.wg_dim;
    let gx = (dims.0 + wg - 1) / wg;
    let gy = (dims.1 + wg - 1) / wg;

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dispatch_encoder") });
    {
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("compute_pass"), timestamp_writes: None });
        cp.set_pipeline(&pass.pipeline);
        cp.set_bind_group(0, &bg0, &[]);
        cp.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));

    Ok(GpuBuffer::from_raw(out_buffer, byte_len))
}
