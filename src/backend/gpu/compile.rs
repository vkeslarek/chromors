use std::sync::Arc;

use rayon::prelude::*;

use super::context::GpuContext;
use super::datatype::DataType;
use super::emit::EmittedIr;
use super::materialize::MaterializePlan;
use super::op::working_image_type;

// ── CompiledShader — cacheable artifact ──────────────────────────────────────

/// The wgpu pipelines and bind-group layouts for a compiled shader pass.
///
/// Keyed by `EmittedIr::cache_key()` in `GpuContext::pipeline_cache`.
/// Structurally identical graphs produce identical IR text → the same cache key
/// → zero recompilation, even as the graph grows with new operations.
pub struct CompiledShader {
    /// 2 bind-group layouts (group 0 = sources+params, group 1 = temps+targets).
    pub bgls: Vec<Arc<wgpu::BindGroupLayout>>,
    /// One compute pipeline per entry point (one per needed node in topo order).
    pub pipelines: Vec<Arc<wgpu::ComputePipeline>>,
    /// `(entry_name, dispatch_w, dispatch_h)` in topo order.
    pub entry_points: Vec<(String, u32, u32)>,
    pub slang_text: String,
    pub source_count: u32,
    pub target_count: u32,
}

// ── DispatchPass — one-shot per-tile dispatch ─────────────────────────────────

/// Fully compiled and buffer-allocated shader pass, ready to encode + dispatch.
///
/// Created fresh for every materialized tile — the temporary/output buffers are
/// single-use and returned to the arena in `encode()`.  The cached shader
/// machinery lives in the `shader` field and is shared across passes.
pub struct DispatchPass {
    pub shader: Arc<CompiledShader>,
    pub temp_bufs: Vec<wgpu::Buffer>,
    pub out_bufs: Vec<wgpu::Buffer>,
    pub params_gpu: wgpu::Buffer,
    pub target_rects: Vec<(u32, u32)>,
    pub target_output_kinds: Vec<Arc<dyn DataType>>,
    pub temp_buffer_sizes: Vec<u64>,
}

impl DispatchPass {
    /// Compile a graph's IR into wgpu pipelines and allocate per-pass GPU buffers.
    pub fn compile(
        ir: EmittedIr,
        plan: &MaterializePlan,
        shader_dir: &std::path::Path,
        out_dir: &std::path::Path,
        ctx: &GpuContext,
    ) -> Result<Self, String> {
        let _t_start = std::time::Instant::now();
        let hash_val = ir.cache_key();

        // Debug dump — only in debug builds to avoid unconditional I/O panics.
        #[cfg(debug_assertions)]
        let _ = std::fs::write(format!("/tmp/ir_debug_{hash_val:016x}.txt"), &ir.text);

        let device = &ctx.device;
        let cached = ctx.pipeline_cache.write().unwrap().get(&hash_val).cloned();

        let shader = if let Some(cached) = cached {
            let entry_points = ir.entry_points.clone();
            Arc::new(CompiledShader {
                bgls: cached.bgls.clone(),
                pipelines: cached.pipelines.clone(),
                entry_points,
                slang_text: ir.text.clone(),
                source_count: ir.source_count as u32,
                target_count: ir.target_count as u32,
            })
        } else {
            let spirv = Self::compile_spirv(&ir.text, hash_val, shader_dir, out_dir)?;
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("compiled"),
                source: wgpu::util::make_spirv(&spirv),
            });

            let g0 = ir.source_count + 1;
            let g1 = ir.temp_buffer_sizes.len() + ir.target_count;
            let limit = ctx.max_storage_buffers as usize;
            if g0 > limit || g1 > limit {
                tracing::warn!(
                    target: "gpu_budget",
                    "compile: g0={g0} g1={g1} limit={limit} — BFS should have staged this"
                );
            }

            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let bgls = Self::create_bind_group_layouts(device, &ir);
            let bgl_refs: Vec<Option<&wgpu::BindGroupLayout>> =
                bgls.iter().map(|b| Some(b.as_ref())).collect();
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("pl"),
                bind_group_layouts: &bgl_refs,
                immediate_size: 0,
            });
            if let Some(err) = pollster::block_on(scope.pop()) {
                return Err(format!("GPU bind-group/pipeline validation: {err}"));
            }

            let pipelines: Vec<Arc<wgpu::ComputePipeline>> =
                Self::create_pipelines(device, &module, &pl, &ir)
                    .into_iter()
                    .map(Arc::new)
                    .collect();

            let new_cache = Arc::new(super::context::CachedPipelines {
                bgls: bgls.clone(),
                pipelines: pipelines.iter().map(Arc::clone).collect(),
            });
            ctx.pipeline_cache.write().unwrap().put(hash_val, new_cache);

            Arc::new(CompiledShader {
                bgls,
                pipelines,
                entry_points: ir.entry_points.clone(),
                slang_text: ir.text.clone(),
                source_count: ir.source_count as u32,
                target_count: ir.target_count as u32,
            })
        };

        let target_rects: Vec<(u32, u32)> = plan
            .targets
            .iter()
            .map(|t| (t.rect.width as u32, t.rect.height as u32))
            .collect();

        let (temp_bufs, out_bufs) = Self::allocate_buffers(ctx, &ir, &target_rects);
        let params_gpu = Self::create_params_buffer(ctx, &ir.params_bytes);

        Ok(DispatchPass {
            shader,
            temp_bufs,
            out_bufs,
            params_gpu,
            target_rects,
            target_output_kinds: ir.target_output_kinds,
            temp_buffer_sizes: ir.temp_buffer_sizes,
        })
    }

    /// Encode one compute pass into `encoder` and return the output buffers.
    ///
    /// Temp and params buffers are freed back to the arena on completion.
    pub fn encode(
        self,
        ctx: &GpuContext,
        in_buffers: &[Arc<super::buffer::GpuBuffer>],
        params_bytes: &[u8],
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(Vec<wgpu::Buffer>, Vec<wgpu::Buffer>, wgpu::Buffer), String> {
        let device = &ctx.device;
        let queue = &ctx.queue;

        let mut contents = params_bytes.to_vec();
        if contents.is_empty() {
            contents.resize(16, 0);
        }
        queue.write_buffer(&self.params_gpu, 0, &contents);

        let num_temps = self.temp_bufs.len() as u32;

        let g0_entries: Vec<wgpu::BindGroupEntry> = in_buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.buffer().as_entire_binding(),
            })
            .chain(std::iter::once(wgpu::BindGroupEntry {
                binding: in_buffers.len() as u32,
                resource: self.params_gpu.as_entire_binding(),
            }))
            .collect();
        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_sources_params"),
            layout: &self.shader.bgls[0],
            entries: &g0_entries,
        });

        let g1_entries: Vec<wgpu::BindGroupEntry> = self
            .temp_bufs
            .iter()
            .enumerate()
            .map(|(i, tb)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: tb.as_entire_binding(),
            })
            .chain(
                self.out_bufs
                    .iter()
                    .enumerate()
                    .map(|(i, ob)| wgpu::BindGroupEntry {
                        binding: num_temps + i as u32,
                        resource: ob.as_entire_binding(),
                    }),
            )
            .collect();
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_temps_targets"),
            layout: &self.shader.bgls[1],
            entries: &g1_entries,
        });

        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("all_regions"),
                timestamp_writes: None,
            });
            cp.set_bind_group(0, &bg0, &[]);
            cp.set_bind_group(1, &bg1, &[]);
            for (i, (_entry_name, w, h)) in self.shader.entry_points.iter().enumerate() {
                let wg_x = w.div_ceil(ctx.wg_dim);
                let wg_y = h.div_ceil(ctx.wg_dim);
                cp.set_pipeline(&self.shader.pipelines[i]);
                cp.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }

        Ok((self.out_bufs, self.temp_bufs, self.params_gpu))
    }
}

// ── Type alias for call-site backwards compat ─────────────────────────────────

/// Back-compat alias — prefer `DispatchPass` in new code.
pub type Compiled = DispatchPass;

// ── Build-time shader paths ───────────────────────────────────────────────────

/// Returns `(shader_dir, slang_cache_dir)` relative to this crate's Cargo.toml.
///
/// Centralised here so both `data.rs` (warmup thread) and `materialize.rs`
/// (actual dispatch) use the same paths without duplicating the `env!` calls.
pub fn shader_paths() -> (std::path::PathBuf, std::path::PathBuf) {
    let base = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    (base.join("shaders"), base.join("../target/slang-cache"))
}

impl DispatchPass {
    // ── Internal helpers ──────────────────────────────────────────────────────────

    fn compile_spirv(
        text: &str,
        hash_val: u64,
        shader_dir: &std::path::Path,
        out_dir: &std::path::Path,
    ) -> Result<Vec<u8>, String> {
        let _ = std::fs::create_dir_all(out_dir);
        let compiler =
            super::slang::SlangCompiler::new(shader_dir.to_path_buf(), out_dir.to_path_buf());
        let spirv = compiler.compile_ir(text, hash_val)?;
        Ok(spirv)
    }

    /// Create 2 bind group layouts (group 0 = sources+params, group 1 = temps+targets).
    fn create_bind_group_layouts(
        device: &wgpu::Device,
        ir: &EmittedIr,
    ) -> Vec<Arc<wgpu::BindGroupLayout>> {
        let num_temps = ir.temp_buffer_sizes.len() as u32;

        let g0_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..ir.source_count as u32)
            .map(|i| Self::bgl_entry(i, true))
            .chain(std::iter::once(Self::bgl_entry(
                ir.source_count as u32,
                true,
            )))
            .collect();
        let bgl0 = Arc::new(
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bgl_sources_params"),
                entries: &g0_entries,
            }),
        );

        let g1_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..num_temps)
            .map(|i| Self::bgl_entry(i, false))
            .chain((0..ir.target_count as u32).map(|i| Self::bgl_entry(num_temps + i, false)))
            .collect();
        let bgl1 = Arc::new(
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bgl_temps_targets"),
                entries: &g1_entries,
            }),
        );

        vec![bgl0, bgl1]
    }

    fn bgl_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn create_pipelines(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        pl: &wgpu::PipelineLayout,
        ir: &EmittedIr,
    ) -> Vec<wgpu::ComputePipeline> {
        ir.entry_points
            .par_iter()
            .map(|(entry_name, _, _)| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_name),
                    layout: Some(pl),
                    module,
                    entry_point: Some(entry_name),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect()
    }

    fn allocate_buffers(
        ctx: &GpuContext,
        ir: &EmittedIr,
        target_rects: &[(u32, u32)],
    ) -> (Vec<wgpu::Buffer>, Vec<wgpu::Buffer>) {
        let temp_bufs = ir
            .temp_buffer_sizes
            .iter()
            .map(|&sz| {
                ctx.arena
                    .allocate(&ctx.device, sz, wgpu::BufferUsages::STORAGE, Some("temp"))
            })
            .collect();

        let out_bufs = target_rects
            .iter()
            .enumerate()
            .map(|(i, &(tw, th))| {
                let wu = super::work_unit::WorkUnit::Region {
                    rect: crate::geometry::Rect::new(0, 0, tw as i32, th as i32),
                    lod: super::Lod::FULL,
                };
                let sz = match ir.target_output_kinds.get(i) {
                    Some(dt) => dt.byte_size(&wu),
                    None => working_image_type().byte_size(&wu),
                };
                ctx.arena.allocate(
                    &ctx.device,
                    sz,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    Some("out"),
                )
            })
            .collect();

        (temp_bufs, out_bufs)
    }

    fn create_params_buffer(ctx: &GpuContext, bytes: &[u8]) -> wgpu::Buffer {
        let mut contents = bytes.to_vec();
        if contents.is_empty() {
            contents.resize(16, 0);
        }
        let buf = ctx.arena.allocate(
            &ctx.device,
            contents.len() as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            Some("params"),
        );
        ctx.queue.write_buffer(&buf, 0, &contents);
        buf
    }
}
