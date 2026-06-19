use super::context::CachedPipelines;
use super::emit::{self, Slot};
use super::{GpuBuilder, GpuContext};
use crate::Error;
use crate::buffer::GpuBuffer;
use std::sync::Arc;
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
        let manifest = env!("CARGO_MANIFEST_DIR");
        let dir = std::path::PathBuf::from(manifest)
            .parent()
            .map(|p| p.join("shaders"))
            .unwrap_or_else(|| std::path::PathBuf::from(manifest).join("shaders"));
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

pub fn compile(
    ctx: &GpuContext,
    builder: &GpuBuilder,
    slang: String,
    hash: u64,
) -> Result<DispatchPass, Error> {
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
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("jit_fused_module"),
            source: wgpu::util::make_spirv(&spirv),
        });

    // Binding layout follows `emit::slots()` exactly — target/work buffers are
    // read-write, params/source buffers are read-only.
    let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = emit::slots(builder)
        .enumerate()
        .map(|(binding, slot)| {
            let read_only = match slot {
                Slot::Target | Slot::Work(_, _) => false,
                Slot::Params | Slot::Source(_, _) => true,
            };
            wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }
        })
        .collect();

    let bgl0 = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bgl_targets"),
            entries: &bgl_entries,
        });

    let pl = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[Some(&bgl0)],
            immediate_size: 0,
        });

    // We assume the Slang entry point is always "main"
    let entry_point = "main";

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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

    Ok(DispatchPass {
        bgls,
        pipeline,
        slang_text: slang,
    })
}

/// Invoca o compilador C++ FFI para compilar o shader JIT gerado para SPIR-V
fn compile_spirv(text: &str, hash: u64) -> Result<Vec<u8>, Error> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap_or_else(|_| {
        std::path::PathBuf::from(manifest)
            .join("target")
            .to_str()
            .unwrap()
            .to_string()
    }));
    let shader_dir = std::path::PathBuf::from(manifest).join("shaders");

    let compiler = super::slang::SlangCompiler::new(shader_dir, out_dir);
    compiler
        .compile_ir(text, hash)
        .map_err(|e| Error::Backend(format!("Slang error: {}", e)))
}

/// wgpu requires storage-buffer binding sizes to be a multiple of 4. Logical
/// region byte sizes (e.g. RGB8 tiles where `w*h*3` is odd-aligned) often
/// aren't, so allocations are padded up; `byte_len` (the logical size used
/// for `as_entire_binding`/download) is rounded separately from the
/// caller-visible `GpuBuffer::byte_len`.
fn align4(size: u64) -> u64 {
    (size + 3) & !3
}

pub fn dispatch(
    ctx: &GpuContext,
    pass: &DispatchPass,
    builder: &GpuBuilder,
    out_bytes: u64,
    dims: (u32, u32),
) -> Result<Arc<GpuBuffer>, Error> {
    // Output buffer holds the result, GPU-resident. Sized by the agnostic
    // `AnyKind::byte_size(wu)` resolved during the demand walk.
    let byte_len = out_bytes.max(16);
    let out_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output_buffer"),
        size: align4(byte_len),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    }));

    use wgpu::util::DeviceExt;

    let params_buf = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &builder.params.bytes,
            usage: wgpu::BufferUsages::STORAGE, // Matches Storage { read_only: true }
        });

    // Working scratch buffers, one per `Slot::Work`, sized by each step's
    // `TempElem::byte_size`.
    let work_buffers: Vec<Arc<wgpu::Buffer>> = emit::slots(builder)
        .filter_map(|slot| match slot {
            Slot::Work(k, elem) => {
                let work_len = (dims.0 as u64 * dims.1 as u64 * elem.byte_size).max(16);
                Some(Arc::new(ctx.device.create_buffer(
                    &wgpu::BufferDescriptor {
                        label: Some(&format!("work_{k}")),
                        size: align4(work_len),
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    },
                )))
            }
            _ => None,
        })
        .collect();

    let mut work_iter = work_buffers.iter();
    let bg_entries: Vec<wgpu::BindGroupEntry> = emit::slots(builder)
        .enumerate()
        .map(|(binding, slot)| {
            let binding = binding as u32;
            let resource = match slot {
                Slot::Target => out_buffer.as_entire_binding(),
                Slot::Params => params_buf.as_entire_binding(),
                Slot::Work(_, _) => work_iter.next().unwrap().as_entire_binding(),
                Slot::Source(i, _) => builder.source_buffers[i].buffer.as_entire_binding(),
            };
            wgpu::BindGroupEntry { binding, resource }
        })
        .collect();

    let bg0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg0"),
        layout: &pass.bgls[0],
        entries: &bg_entries,
    });

    let wg = ctx.wg_dim;
    let gx = (dims.0 + wg - 1) / wg;
    let gy = (dims.1 + wg - 1) / wg;

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("dispatch_encoder"),
        });
    {
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cp.set_pipeline(&pass.pipeline);
        cp.set_bind_group(0, &bg0, &[]);
        cp.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(std::iter::once(encoder.finish()));

    Ok(GpuBuffer::from_raw(out_buffer, byte_len))
}
