//! The GPU Materialization Pipeline
//!
//! This module orchestrates the entire execution pipeline for the GPU backend.
//! It converts a high-level request (a NodeId and a Rect) into compiled, executed pixels.
//!
//! Architecture:
//! - **MaterializePipeline**: The entry point that orchestrates typestate transitions.
//! - **Typestates**: `CachedBatch` -> `PlannedBatch` -> `CompiledBatch` -> `SubmittedBatch`.
//! - **Capabilities**: `CacheKey` namespace.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::color::space::ColorSpace;
use crate::geometry::{Rect, merge_overlapping};
use crate::pixel::{AlphaPolicy, PixelFormat, PixelMeta};

use super::Lod;
use super::buffer::{GpuBuffer, ImageBuffer};
use super::datatype::DataType;
use super::graph::{Graph, NodeId, RegionKey};
use super::op::working_image_type;
use super::region::GpuRegion;
use super::source::AnyGpuSource;
use super::value::{MaterializedValue, Storage};
use super::work_unit::{AnyWorkUnit, Region, WorkUnit, WorkUnitKind};

// ── ERROR HANDLING ────────────────────────────────────────────────────────────

#[derive(thiserror::Error, Debug)]
pub enum MaterializeError {
    #[error("Mutex lock poisoned")]
    LockPoisoned,
    #[error(
        "Staged cut node produced a non-image output; staging is only supported for Image2D nodes"
    )]
    InvalidCutOutput,
    #[error("Compile error: {0}")]
    Compile(String),
    #[error("Encode error: {0}")]
    Encode(String),
    #[error(transparent)]
    Inner(#[from] crate::error::Error),
}

impl From<MaterializeError> for crate::error::Error {
    fn from(err: MaterializeError) -> Self {
        match err {
            MaterializeError::Inner(e) => e,
            _ => crate::error::Error::Render(err.to_string()),
        }
    }
}

// ── CACHE KEYS (NEWTYPE PATTERN FOR DOMAIN CAPABILITY) ────────────────────────

pub struct CacheKey;

impl CacheKey {
    /// Fold a `lod` + domain discriminator into a content hash.
    #[inline]
    fn fold(content: u64, disc: u64) -> u64 {
        (content ^ disc).wrapping_mul(0x0000_0100_0000_01b3) ^ content.rotate_left(17)
    }

    /// Cache key for an op-output tile, addressed by the producing subgraph's
    /// `content` hash + `lod` + `rect`. Identical computations across graph
    /// forks/sessions therefore share the entry.
    #[inline]
    pub fn region(content: u64, lod: Lod, rect: Rect) -> RegionKey {
        (
            Self::fold(content, lod.0 as u64),
            rect.x,
            rect.y,
            rect.width,
            rect.height,
        )
    }

    /// Cache key for a source-fetch tile. A domain bit (`1 << 40`) keeps it from
    /// ever colliding with an op-output key of the same content + lod + rect.
    #[inline]
    pub fn source_fetch(content: u64, lod: Lod, rect: Rect) -> RegionKey {
        (
            Self::fold(content, (lod.0 as u64) | (1 << 40)),
            rect.x,
            rect.y,
            rect.width,
            rect.height,
        )
    }
}

// ── CORE DATA STRUCTURES ──────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub struct RegionRequest {
    pub target: NodeId,
    pub rect: Rect,
}

#[derive(Clone, Debug)]
pub struct BufferTarget {
    pub node_id: NodeId,
    pub rect: Rect,
    pub buffer: Option<Arc<GpuBuffer>>,
}

#[derive(Clone, Debug)]
pub struct BufferRegion {
    pub stride: u32,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct MaterializePlan {
    pub sources: Vec<NodeId>,
    pub targets: Vec<BufferTarget>,
    pub source_fetches: Vec<(NodeId, Vec<(Rect, BufferRegion)>)>,
    pub node_outputs: Vec<(NodeId, Rect)>,
}

// ── GRAPH MATERIALIZATION ─────────────────────────────────────────────────────

impl Graph {
    /// Build a materialization plan for the given `(node_id, WorkUnit)` targets.
    pub fn materialize(
        &self,
        targets_info: &[(NodeId, super::work_unit::WorkUnit)],
        lod: Lod,
    ) -> MaterializePlan {
        let node_rects = self.walk_inverse(targets_info, lod);
        let sources = self.filter_reachable_sources(&node_rects);
        let source_fetches = self.build_source_fetches(&sources, &node_rects, lod);
        let node_outputs = self.sort_node_outputs(&node_rects);
        let targets = self.build_targets(targets_info, &node_rects);

        MaterializePlan {
            sources,
            targets,
            source_fetches,
            node_outputs,
        }
    }

    fn filter_reachable_sources(&self, node_rects: &HashMap<NodeId, Rect>) -> Vec<NodeId> {
        self.sources
            .iter()
            .map(|s| s.id)
            .filter(|id| node_rects.contains_key(id))
            .collect()
    }

    fn build_source_fetches(
        &self,
        sources: &[NodeId],
        node_rects: &HashMap<NodeId, Rect>,
        lod: Lod,
    ) -> Vec<(NodeId, Vec<(Rect, BufferRegion)>)> {
        sources
            .iter()
            .map(|&id| {
                let rects: Vec<Rect> = node_rects.get(&id).into_iter().copied().collect();
                let merged = merge_overlapping(rects);
                let regions = merged
                    .into_iter()
                    .map(|mut r| {
                        let source = self.get_source(id).unwrap();
                        let max_w = (source.source.width() / (1u32 << lod.0)).max(1) as i32;
                        let max_h = (source.source.height() / (1u32 << lod.0)).max(1) as i32;
                        let bounds = Rect::new(0, 0, max_w, max_h);

                        let tile_size = 256;
                        let aligned_x = (r.x as f64 / tile_size as f64).floor() as i32 * tile_size;
                        let aligned_y = (r.y as f64 / tile_size as f64).floor() as i32 * tile_size;
                        let aligned_w = ((r.x + r.width - aligned_x) as f64 / tile_size as f64)
                            .ceil() as i32
                            * tile_size;
                        let aligned_h = ((r.y + r.height - aligned_y) as f64 / tile_size as f64)
                            .ceil() as i32
                            * tile_size;
                        r = Rect::new(aligned_x, aligned_y, aligned_w, aligned_h)
                            .intersection(bounds)
                            .unwrap_or(r);

                        let br = BufferRegion {
                            stride: r.width as u32,
                            x: 0,
                            y: 0,
                            width: r.width as u32,
                            height: r.height as u32,
                        };
                        (r, br)
                    })
                    .collect();
                (id, regions)
            })
            .collect()
    }

    fn sort_node_outputs(&self, node_rects: &HashMap<NodeId, Rect>) -> Vec<(NodeId, Rect)> {
        let order = self.topo_order();
        let mut node_outputs: Vec<(NodeId, Rect)> = node_rects
            .iter()
            .filter(|(id, _)| self.get_source(**id).is_none())
            .map(|(&id, &rect)| (id, rect))
            .collect();

        node_outputs
            .sort_by_key(|(id, _)| order.iter().position(|o| o == id).unwrap_or(usize::MAX));
        node_outputs
    }

    fn build_targets(
        &self,
        targets_info: &[(NodeId, super::work_unit::WorkUnit)],
        node_rects: &HashMap<NodeId, Rect>,
    ) -> Vec<BufferTarget> {
        targets_info
            .iter()
            .map(|(node_id, _)| {
                let rect = node_rects
                    .get(node_id)
                    .copied()
                    .unwrap_or(Rect::new(0, 0, 0, 0));
                BufferTarget {
                    node_id: *node_id,
                    rect,
                    buffer: None,
                }
            })
            .collect()
    }
}

// ── TYPESTATE PIPELINE STRUCTS ────────────────────────────────────────────────

struct PlannedJob {
    i: usize,
    rect: Rect,
    lod: Lod,
    key: RegionKey,
    plan: MaterializePlan,
    sources_snapshot: Vec<(NodeId, Arc<super::source::GpuSource>)>,
    ir: super::emit::EmittedIr,
    layout: super::emit::LayoutPlan,
    datatype: Arc<dyn DataType>,
    source_fetch_rects: HashMap<NodeId, Rect>,
}

struct CompiledJob {
    i: usize,
    rect: Rect,
    lod: Lod,
    key: RegionKey,
    sources_snapshot: Vec<(NodeId, Arc<super::source::GpuSource>)>,
    datatype: Arc<dyn DataType>,
    compiled: super::compile::DispatchPass,
    fetched_buffers: Vec<Arc<ImageBuffer>>,
    params_bytes: Vec<u8>,
}

struct SubmittedJob {
    i: usize,
    rect: Rect,
    lod: Lod,
    key: RegionKey,
    datatype: Arc<dyn DataType>,
    out_bufs: Vec<wgpu::Buffer>,
}

// ── PIPELINE STATE INITIAL ──
pub struct MaterializePipeline<'a> {
    region: &'a GpuRegion,
    wus: &'a [Region],
}

impl<'a> MaterializePipeline<'a> {
    pub fn new(region: &'a GpuRegion, wus: &'a [Region]) -> Self {
        Self { region, wus }
    }

    pub fn execute(self) -> Result<Vec<Arc<MaterializedValue>>, MaterializeError> {
        if self.wus.is_empty() {
            return Ok(Vec::new());
        }

        let cached_batch = self.check_cache()?;
        if cached_batch.is_fully_cached() {
            return cached_batch.collect_results();
        }

        match cached_batch.plan_jobs()? {
            PlanResult::NeedsCut(cut_batch) => cut_batch.execute_cuts(),
            PlanResult::Ready(planned_batch) => planned_batch
                .compile_and_fetch()?
                .encode_and_submit()?
                .readback(),
        }
    }

    fn check_cache(self) -> Result<CachedBatch<'a>, MaterializeError> {
        let mut results = vec![None; self.wus.len()];
        let mut uncached = Vec::new();
        let mut cache_lock = self
            .region
            .cache
            .lock()
            .map_err(|_| MaterializeError::LockPoisoned)?;

        for (i, wu) in self.wus.iter().enumerate() {
            let key = CacheKey::region(self.region.content, wu.lod, wu.rect);
            if let Some(hit) = cache_lock.get(&key) {
                results[i] = Some(hit);
            } else {
                uncached.push((i, wu.clone(), key));
            }
        }

        Ok(CachedBatch {
            region: self.region,
            results,
            uncached,
        })
    }
}

// ── PIPELINE STATE CACHED ──
struct CachedBatch<'a> {
    region: &'a GpuRegion,
    results: Vec<Option<Arc<MaterializedValue>>>,
    uncached: Vec<(usize, Region, RegionKey)>,
}

enum PlanResult<'a> {
    Ready(PlannedBatch<'a>),
    NeedsCut(CutBatch<'a>),
}

impl<'a> CachedBatch<'a> {
    fn is_fully_cached(&self) -> bool {
        self.uncached.is_empty()
    }

    fn collect_results(self) -> Result<Vec<Arc<MaterializedValue>>, MaterializeError> {
        Ok(self.results.into_iter().flatten().collect())
    }

    fn plan_jobs(self) -> Result<PlanResult<'a>, MaterializeError> {
        let mut jobs = Vec::with_capacity(self.uncached.len());
        let graph = self
            .region
            .graph
            .lock()
            .map_err(|_| MaterializeError::LockPoisoned)?;

        for (i, wu, key) in &self.uncached {
            let (i, key) = (*i, *key);
            let rect = wu.rect;
            let lod = wu.lod;
            let plan = graph.materialize(&[(self.region.node_id, wu.to_work_unit())], lod);
            let (ir, layout) = plan.emit_ir_with_layout(&graph, self.region.ctx.wg_dim, lod);

            let limit = self.region.ctx.max_storage_buffers as usize;
            let g0 = ir.source_count + 1;
            let g1 = ir.temp_buffer_sizes.len() + ir.target_count;

            // Only invoke the cut-finder when emitted IR actually exceeds the
            // device storage-buffer limit.  The cut-finder uses a simpler
            // "one temp per node" budget model that does not account for the
            // liveness-based slot reuse in alloc_temps; running it when IR
            // already fits would produce unnecessary cuts.
            if g0 > limit || g1 > limit {
                tracing::warn!(target: "gpu_region", "IR exceeds storage-buffer limit (g0={}, g1={}, limit={}). Staging cuts.", g0, g1, limit);
                let budget = limit.saturating_sub(1);
                let cuts =
                    super::pass::CutFinder::new(&graph, self.region.node_id, rect, lod, budget)
                        .execute();
                if !cuts.staging.is_empty() {
                    return Ok(PlanResult::NeedsCut(CutBatch {
                        region: self.region,
                        results: self.results,
                        uncached: self.uncached.clone(),
                        cut_i: i,
                        cut_wu: wu.clone(),
                        cuts,
                    }));
                }
            }

            let topo = graph.topo_order();
            let datatype = graph
                .get_node(self.region.node_id)
                .map(|n| n.datatype.clone())
                .unwrap_or_else(working_image_type);

            let sources_snapshot: Vec<_> = topo
                .iter()
                .filter_map(|&id| graph.get_source(id).map(|s| (id, s.source.clone())))
                .collect();

            let source_fetch_rects: HashMap<NodeId, Rect> = plan
                .source_fetches
                .iter()
                .filter_map(|(id, fetches)| fetches.first().map(|(r, _)| (*id, *r)))
                .collect();

            jobs.push(PlannedJob {
                i,
                rect,
                lod,
                key,
                plan,
                sources_snapshot,
                ir,
                layout,
                datatype,
                source_fetch_rects,
            });
        }

        Ok(PlanResult::Ready(PlannedBatch {
            region: self.region,
            results: self.results,
            jobs,
        }))
    }
}

// ── CUT EXECUTION ──
struct CutBatch<'a> {
    region: &'a GpuRegion,
    results: Vec<Option<Arc<MaterializedValue>>>,
    uncached: Vec<(usize, Region, RegionKey)>,
    cut_i: usize,
    cut_wu: Region,
    cuts: super::pass::StagingCuts,
}

impl<'a> CutBatch<'a> {
    fn execute_cuts(mut self) -> Result<Vec<Arc<MaterializedValue>>, MaterializeError> {
        if self.uncached.len() == 1 {
            let mat = StagingCutter::execute(self.region, self.cut_wu, self.cuts)?;
            self.results[self.cut_i] = Some(mat);
            Ok(self.results.into_iter().flatten().collect())
        } else {
            let sequential: Result<Vec<_>, MaterializeError> = self
                .uncached
                .into_iter()
                .map(|(i, wu, _key)| {
                    let child = GpuRegion {
                        graph: self.region.graph.clone(),
                        cache: self.region.cache.clone(),
                        node_id: self.region.node_id,
                        wu: std::sync::Mutex::new(Some(wu)),
                        ctx: self.region.ctx.clone(),
                        // Same root node + graph as the parent → same content hash.
                        content: self.region.content,
                    };
                    child
                        .materialize()
                        .map(|m| (i, m))
                        .map_err(MaterializeError::Inner)
                })
                .collect();

            for (i, mat) in sequential? {
                self.results[i] = Some(mat);
            }
            Ok(self.results.into_iter().flatten().collect())
        }
    }
}

struct StagingCutter;

impl StagingCutter {
    fn execute(
        region: &GpuRegion,
        wu: Region,
        cuts: super::pass::StagingCuts,
    ) -> Result<Arc<MaterializedValue>, MaterializeError> {
        let rect = wu.rect;
        let lod = wu.lod;
        let results: Result<Vec<_>, MaterializeError> = cuts
            .staging
            .par_iter()
            .map(|(cut_id, cut_rect)| {
                let content = region.graph.lock().unwrap().content_hash(*cut_id);
                let child = GpuRegion {
                    graph: region.graph.clone(),
                    cache: region.cache.clone(),
                    node_id: *cut_id,
                    wu: std::sync::Mutex::new(Some(Region::new(*cut_rect, lod))),
                    ctx: region.ctx.clone(),
                    content,
                };
                child
                    .materialize()
                    .map(|m| (*cut_id, m))
                    .map_err(MaterializeError::Inner)
            })
            .collect();

        let mut overrides = HashMap::new();
        let lod_scale = 1i32 << lod.0;
        for ((cut_id, cut_rect), mat) in cuts
            .staging
            .iter()
            .zip(results?.into_iter().map(|(_, m)| m))
        {
            match &mat.storage {
                Storage::Vram(buffer) => {
                    // The staging buffer was captured at region.lod, so its
                    // image_rect in full-resolution coords is cut_rect * lod_scale.
                    let image_rect = crate::geometry::Rect::new(
                        cut_rect.x * lod_scale,
                        cut_rect.y * lod_scale,
                        cut_rect.width * lod_scale,
                        cut_rect.height * lod_scale,
                    );
                    let mat_rect = Region::from_work_unit(&mat.extent)
                        .map(|r| r.rect)
                        .unwrap_or(*cut_rect);
                    let meta = mat
                        .datatype
                        .as_any()
                        .downcast_ref::<super::datatype::ImageType>()
                        .map(|it| PixelMeta::new(it.format, it.color_space, AlphaPolicy::Straight))
                        .unwrap_or_else(|| {
                            PixelMeta::new(
                                PixelFormat::RgbaF32,
                                ColorSpace::ACES_CG,
                                AlphaPolicy::Straight,
                            )
                        });
                    let img_buf = ImageBuffer::from_raw(
                        buffer.buffer.clone(),
                        mat_rect.width as u32,
                        mat_rect.height as u32,
                        meta,
                    );
                    overrides.insert(
                        *cut_id,
                        super::source::GpuSource::new_buffer_positioned(
                            img_buf,
                            region.ctx.clone(),
                            image_rect,
                        ),
                    );
                }
                Storage::Host(_) => {
                    return Err(MaterializeError::InvalidCutOutput);
                }
            }
        }

        let (subgraph, sub_root_id) = {
            let graph = region
                .graph
                .lock()
                .map_err(|_| MaterializeError::LockPoisoned)?;
            graph.subgraph_with_overrides(region.node_id, &overrides)
        };

        let sub_content = subgraph.content_hash(sub_root_id);
        let sub_region = GpuRegion {
            graph: Arc::new(std::sync::Mutex::new(subgraph)),
            cache: region.ctx.cache.clone(),
            node_id: sub_root_id,
            wu: std::sync::Mutex::new(Some(Region::new(rect, lod))),
            ctx: region.ctx.clone(),
            content: sub_content,
        };

        let result = sub_region.materialize().map_err(MaterializeError::Inner)?;
        // Cache the staged result under the PARENT region's content identity, so a
        // later non-staged materialize of the same root reuses it.
        let key = CacheKey::region(region.content, lod, rect);

        region
            .cache
            .lock()
            .map_err(|_| MaterializeError::LockPoisoned)?
            .insert(key, result.clone());

        Ok(result)
    }
}

// ── PIPELINE STATE PLANNED ──
struct PlannedBatch<'a> {
    region: &'a GpuRegion,
    results: Vec<Option<Arc<MaterializedValue>>>,
    jobs: Vec<PlannedJob>,
}

impl<'a> PlannedBatch<'a> {
    fn compile_and_fetch(self) -> Result<CompiledBatch<'a>, MaterializeError> {
        let (shader_dir, out_dir) = super::compile::shader_paths();

        let compiled_jobs: Result<Vec<CompiledJob>, MaterializeError> = self
            .jobs
            .into_par_iter()
            .map(|job| {
                let params_bytes = job.ir.params_bytes.clone();

                let (compiled_res, fetched_res) = rayon::join(
                    || {
                        super::compile::DispatchPass::compile(
                            job.ir,
                            &job.plan,
                            &shader_dir,
                            &out_dir,
                            &self.region.ctx,
                        )
                        .map_err(|e| MaterializeError::Compile(e.to_string()))
                    },
                    || {
                        job.layout
                            .sources
                            .par_iter()
                            .map(|src_slot| {
                                let id = src_slot.node_id;
                                if let Some((_, s)) =
                                    job.sources_snapshot.iter().find(|(sid, _)| *sid == id)
                                {
                                    let fetch_rect = job
                                        .source_fetch_rects
                                        .get(&id)
                                        .copied()
                                        .unwrap_or(job.rect);
                                    let src_key = CacheKey::source_fetch(
                                        super::source::source_identity(s),
                                        job.lod,
                                        fetch_rect,
                                    );

                                    if let Ok(mut cache) = self.region.cache.lock()
                                        && let Some(cached) = cache.get(&src_key)
                                        && let Storage::Vram(buffer) = &cached.storage
                                    {
                                        let meta = PixelMeta::new(
                                            s.format(),
                                            s.color_space(),
                                            AlphaPolicy::Straight,
                                        );
                                        return Ok(ImageBuffer::from_raw(
                                            buffer.buffer.clone(),
                                            fetch_rect.width as u32,
                                            fetch_rect.height as u32,
                                            meta,
                                        ));
                                    }

                                    let image_type = super::datatype::ImageType {
                                        color_space: s.color_space(),
                                        format: s.format(),
                                    };
                                    let storage = super::datatype::Sourceable::fetch_region(
                                        &image_type,
                                        s,
                                        &super::work_unit::Region::new(fetch_rect, job.lod),
                                        &self.region.ctx,
                                    )?;
                                    let Storage::Vram(gpu_buf) = storage else {
                                        return Err(crate::error::Error::Gpu(
                                            "source fetch_region returned Host storage".into(),
                                        ));
                                    };
                                    let meta = PixelMeta::new(
                                        image_type.format,
                                        image_type.color_space,
                                        AlphaPolicy::Straight,
                                    );
                                    let buf = ImageBuffer::from_raw(
                                        gpu_buf.buffer.clone(),
                                        fetch_rect.width as u32,
                                        fetch_rect.height as u32,
                                        meta,
                                    );

                                    if let Ok(mut cache) = self.region.cache.lock() {
                                        let datatype: Arc<dyn DataType> = Arc::new(image_type);
                                        cache.insert(
                                            src_key,
                                            Arc::new(MaterializedValue::vram(
                                                gpu_buf,
                                                datatype,
                                                WorkUnit::Region {
                                                    rect: fetch_rect,
                                                    lod: job.lod,
                                                },
                                            )),
                                        );
                                    }

                                    Ok(buf)
                                } else {
                                    Ok(ImageBuffer::alloc(
                                        1,
                                        1,
                                        crate::pixel::PixelMeta::new(
                                            crate::pixel::PixelFormat::Rgba8,
                                            ColorSpace::SRGB,
                                            crate::pixel::AlphaPolicy::Straight,
                                        ),
                                        &self.region.ctx,
                                    ))
                                }
                            })
                            .collect::<Result<Vec<Arc<ImageBuffer>>, crate::error::Error>>()
                            .map_err(MaterializeError::Inner)
                    },
                );

                Ok(CompiledJob {
                    i: job.i,
                    rect: job.rect,
                    lod: job.lod,
                    key: job.key,
                    sources_snapshot: job.sources_snapshot,
                    datatype: job.datatype,
                    compiled: compiled_res?,
                    fetched_buffers: fetched_res?,
                    params_bytes,
                })
            })
            .collect();

        Ok(CompiledBatch {
            region: self.region,
            results: self.results,
            compiled_jobs: compiled_jobs?,
        })
    }
}

// ── PIPELINE STATE COMPILED ──
struct CompiledBatch<'a> {
    region: &'a GpuRegion,
    results: Vec<Option<Arc<MaterializedValue>>>,
    compiled_jobs: Vec<CompiledJob>,
}

impl<'a> CompiledBatch<'a> {
    fn encode_and_submit(mut self) -> Result<SubmittedBatch<'a>, MaterializeError> {
        let mut encoder =
            self.region
                .ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("batch_materialize"),
                });

        let mut pending_outputs = Vec::new();
        let mut buffers_to_free = Vec::new();

        for job in self.compiled_jobs {
            if job.compiled.shader.entry_points.is_empty()
                && !job.sources_snapshot.is_empty()
                && let Some(buf) = job.fetched_buffers.first()
            {
                let datatype: Arc<dyn DataType> = Arc::new(super::datatype::ImageType {
                    color_space: buf.color_space(),
                    format: buf.format(),
                });
                let out = Arc::new(MaterializedValue::vram(
                    buf.buffer.clone(),
                    datatype,
                    WorkUnit::Region {
                        rect: job.rect,
                        lod: job.lod,
                    },
                ));
                self.results[job.i] = Some(out.clone());

                self.region
                    .cache
                    .lock()
                    .map_err(|_| MaterializeError::LockPoisoned)?
                    .insert(job.key, out);
                continue;
            }

            let raw_buffers: Vec<_> = job
                .fetched_buffers
                .iter()
                .map(|b| b.buffer.clone())
                .collect();
            let (out_bufs, mut temps, params) = job
                .compiled
                .encode(
                    &self.region.ctx,
                    &raw_buffers,
                    &job.params_bytes,
                    &mut encoder,
                )
                .map_err(|e| MaterializeError::Encode(e.to_string()))?;

            buffers_to_free.append(&mut temps);
            buffers_to_free.push(params);

            pending_outputs.push(SubmittedJob {
                i: job.i,
                rect: job.rect,
                lod: job.lod,
                key: job.key,
                datatype: job.datatype,
                out_bufs,
            });
        }

        self.region.ctx.queue.submit(Some(encoder.finish()));
        let _ = self
            .region
            .ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely());

        for b in buffers_to_free {
            self.region.ctx.arena.free(b);
        }

        Ok(SubmittedBatch {
            region: self.region,
            results: self.results,
            pending_outputs,
        })
    }
}

// ── PIPELINE STATE SUBMITTED ──
struct SubmittedBatch<'a> {
    region: &'a GpuRegion,
    results: Vec<Option<Arc<MaterializedValue>>>,
    pending_outputs: Vec<SubmittedJob>,
}

impl<'a> SubmittedBatch<'a> {
    fn readback(mut self) -> Result<Vec<Arc<MaterializedValue>>, MaterializeError> {
        for mut job in self.pending_outputs {
            let buf = job
                .out_bufs
                .pop()
                .expect("Encode should have produced at least one buffer");

            let out = if job.datatype.needs_fused_temp() {
                let wu = WorkUnit::Region {
                    rect: job.rect,
                    lod: job.lod,
                };
                let byte_len = job.datatype.byte_size(&wu);
                let gpu_buf = GpuBuffer::from_raw(Arc::new(buf), byte_len);
                Arc::new(MaterializedValue::vram(gpu_buf, job.datatype.clone(), wu))
            } else {
                let size = buf.size();
                let staging = self
                    .region
                    .ctx
                    .device
                    .create_buffer(&wgpu::BufferDescriptor {
                        label: Some("staging"),
                        size,
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });

                let mut enc = self
                    .region
                    .ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, size);
                self.region.ctx.queue.submit(Some(enc.finish()));

                let slice = staging.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();

                slice.map_async(wgpu::MapMode::Read, move |res| {
                    let _ = tx.send(res);
                });
                let _ = self
                    .region
                    .ctx
                    .device
                    .poll(wgpu::PollType::wait_indefinitely());

                rx.recv()
                    .map_err(|_| MaterializeError::Encode("Readback channel failed".into()))?
                    .map_err(|e| MaterializeError::Encode(e.to_string()))?;

                let bytes = slice.get_mapped_range().to_vec();
                staging.unmap();

                let extent = match job.datatype.work_unit_kind() {
                    WorkUnitKind::Region => WorkUnit::Region {
                        rect: job.rect,
                        lod: job.lod,
                    },
                    WorkUnitKind::Range => WorkUnit::Range {
                        start: job.rect.x as u32,
                        end: (job.rect.x + job.rect.width) as u32,
                    },
                    WorkUnitKind::Atomic => WorkUnit::Atomic,
                };
                Arc::new(MaterializedValue::host(bytes, job.datatype.clone(), extent))
            };

            self.region
                .cache
                .lock()
                .map_err(|_| MaterializeError::LockPoisoned)?
                .insert(job.key, out.clone());

            self.results[job.i] = Some(out);
        }

        Ok(self.results.into_iter().flatten().collect())
    }
}

// ── ENTRYPOINT ────────────────────────────────────────────────────────────────

pub fn execute_batch(
    region: &GpuRegion,
    wus: &[Region],
) -> Result<Vec<Arc<MaterializedValue>>, crate::error::Error> {
    MaterializePipeline::new(region, wus)
        .execute()
        .map_err(Into::into)
}
