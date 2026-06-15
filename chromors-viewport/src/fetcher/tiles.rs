use crate::atlas::TILE;
use crate::rect::Rect;
use poc::backend::gpu::{GpuBackend, GpuBuffer};
use poc::color::model::ColorModel;
use poc::data::image::{GpuBufferTarget, Image2D as GenImage};
use poc::pixel::{AlphaState, PixelLayout, Storage};
use poc::work_unit::{Lod, Region};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Condvar, Mutex};

use super::{FetchPayload, FetchTask};

#[derive(Clone, PartialEq, Eq, Hash)]
struct JobKey {
    layer_id: u64,
    mip: u32,
    is_patch: bool,
}

enum FetchJob {
    Full {
        priority: u8,
        version: u64,
        mip_img: GenImage<GpuBackend>,
        missing: Vec<(u32, u32)>,
        mip_w: u32,
        mip_h: u32,
    },
    Patch {
        priority: u8,
        version: u64,
        mip_img: GenImage<GpuBackend>,
        rect: Rect,
        patches: Vec<(u32, u32, u32, u32, u32, u32)>,
    },
}

impl FetchJob {
    /// Lower = fetched sooner (0 = the on-screen mip).
    fn priority(&self) -> u8 {
        match self {
            FetchJob::Full { priority, .. } | FetchJob::Patch { priority, .. } => *priority,
        }
    }
}

struct FetchState {
    jobs: HashMap<JobKey, FetchJob>,
}

pub struct TileFetcher {
    version: Arc<AtomicU64>,
    state: Arc<(Mutex<FetchState>, Condvar)>,
}

/// Converts `img` to straight-alpha RGBA8 (the atlas upload format) unless it
/// already is.
fn ensure_rgba8(img: GenImage<GpuBackend>) -> GenImage<GpuBackend> {
    let layout = img.layout();
    let target = PixelLayout {
        storage: Storage::U8,
        model: ColorModel::Rgb,
        alpha: AlphaState::Straight,
        color_space: layout.color_space,
    };
    if layout != target {
        img.convert(target)
    } else {
        img
    }
}

impl TileFetcher {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, tx: Sender<FetchTask>) -> Self {
        // Bounded concurrency: each in-flight tile pull can pre-materialize a
        // full-resolution slab (coarse mips of a large image), so too many
        // parallel fetches exhaust VRAM. Cap the pool so peak GPU memory stays
        // a small multiple of one slab.
        let pool = Arc::new(
            rayon::ThreadPoolBuilder::new()
                .num_threads(32)
                .thread_name(|i| format!("vp-fetch-{}", i))
                .build()
                .expect("failed to create fetch thread pool"),
        );

        let state = Arc::new((
            Mutex::new(FetchState {
                jobs: HashMap::new(),
            }),
            Condvar::new(),
        ));

        let version = Arc::new(AtomicU64::new(0));

        let state_clone = state.clone();
        let v_clone = version.clone();
        let q_clone = queue.clone();

        std::thread::Builder::new()
            .name("vp-fetch-worker".to_string())
            .spawn(move || {
                Self::worker_loop(state_clone, tx, device, q_clone, pool, v_clone);
            })
            .unwrap();

        Self { version, state }
    }

    pub fn bump_version(&mut self) {
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    pub fn spawn_fetch(
        &self,
        layer_id: u64,
        mip: u32,
        mip_img: GenImage<GpuBackend>,
        missing: Vec<(u32, u32)>,
        mip_w: u32,
        mip_h: u32,
        priority: u8,
    ) {
        if missing.is_empty() {
            return;
        }
        let mip_img = ensure_rgba8(mip_img);
        self.submit(
            JobKey {
                layer_id,
                mip,
                is_patch: false,
            },
            FetchJob::Full {
                priority,
                version: self.version(),
                mip_img,
                missing,
                mip_w,
                mip_h,
            },
        );
    }

    pub fn spawn_patch_fetch(
        &self,
        layer_id: u64,
        mip: u32,
        mip_img: GenImage<GpuBackend>,
        rect: Rect,
        patches: Vec<(u32, u32, u32, u32, u32, u32)>,
        priority: u8,
    ) {
        if patches.is_empty() {
            return;
        }
        let mip_img = ensure_rgba8(mip_img);
        self.submit(
            JobKey {
                layer_id,
                mip,
                is_patch: true,
            },
            FetchJob::Patch {
                priority,
                version: self.version(),
                mip_img,
                rect,
                patches,
            },
        );
    }

    fn submit(&self, key: JobKey, job: FetchJob) {
        let (lock, cvar) = &*self.state;
        lock.lock().unwrap().jobs.insert(key, job);
        cvar.notify_one();
    }

    fn worker_loop(
        state: Arc<(Mutex<FetchState>, Condvar)>,
        sender: Sender<FetchTask>,
        device: wgpu::Device,
        queue: wgpu::Queue,
        pool: Arc<rayon::ThreadPool>,
        current_version: Arc<AtomicU64>,
    ) {
        let (lock, cvar) = &*state;
        loop {
            let (key, job) = {
                let mut guard = lock.lock().unwrap();
                loop {
                    // Highest priority (lowest value) first — the on-screen mip
                    // beats fallback/predictive fetches.
                    if let Some(k) = guard
                        .jobs
                        .iter()
                        .min_by_key(|(_, j)| j.priority())
                        .map(|(k, _)| k.clone())
                    {
                        let j = guard.jobs.remove(&k).unwrap();
                        break (k, j);
                    }
                    guard = cvar.wait(guard).unwrap();
                }
            };

            let global = current_version.load(Ordering::Relaxed);

            match job {
                FetchJob::Full {
                    priority: _,
                    version,
                    mip_img,
                    missing,
                    mip_w,
                    mip_h,
                } => {
                    if version != global {
                        continue;
                    }

                    let rects: Vec<(Rect, u32, u32)> = missing
                        .into_iter()
                        .map(|(tx, ty)| {
                            (
                                Rect::new(
                                    (tx * TILE) as i32,
                                    (ty * TILE) as i32,
                                    TILE.min(mip_w.saturating_sub(tx * TILE)) as i32,
                                    TILE.min(mip_h.saturating_sub(ty * TILE)) as i32,
                                ),
                                tx,
                                ty,
                            )
                        })
                        .collect();

                    if rects.is_empty() {
                        continue;
                    }

                    let sender = sender.clone();
                    let (layer_id, mip) = (key.layer_id, key.mip);
                    let dev = device.clone();
                    let q = queue.clone();
                    let cv = current_version.clone();

                    pool.install(move || {
                        let _span = tracing::trace_span!("tile.worker_batch").entered();
                        if cv.load(Ordering::Relaxed) != global {
                            return;
                        }
                        rects
                            .into_par_iter()
                            .for_each_with(sender, |sender, (rect, tx, ty)| {
                                if cv.load(Ordering::Relaxed) != global
                                    || rect.width <= 0
                                    || rect.height <= 0
                                {
                                    return;
                                }
                                let region = Region {
                                    x: rect.x,
                                    y: rect.y,
                                    w: rect.width,
                                    h: rect.height,
                                    // Pull at the mip's LOD: the source
                                    // shrink-on-loads to this level.
                                    lod: Lod(mip),
                                };
                                match mip_img.pull(&GpuBufferTarget, region) {
                                    Ok(buf) => emit_tile(
                                        sender, &buf, layer_id, global, mip, tx, ty, rect, rect,
                                        &dev, &q,
                                    ),
                                    Err(e) => {
                                        tracing::error!(
                                            "pull error on tile ({},{}) mip={}: {:?}",
                                            tx,
                                            ty,
                                            mip,
                                            e
                                        );
                                    }
                                }
                            });
                    });
                }

                FetchJob::Patch {
                    priority: _,
                    version,
                    mip_img,
                    rect,
                    patches,
                } => {
                    if version != global || rect.width <= 0 || rect.height <= 0 {
                        continue;
                    }
                    if patches.is_empty() {
                        continue;
                    }

                    let region = Region {
                        x: rect.x,
                        y: rect.y,
                        w: rect.width,
                        h: rect.height,
                        lod: Lod(key.mip),
                    };
                    match mip_img.pull(&GpuBufferTarget, region) {
                        Ok(buf) => {
                            let sender = sender.clone();
                            let dev = device.clone();
                            let q = queue.clone();
                            let (layer_id, mip) = (key.layer_id, key.mip);
                            let cv = current_version.clone();

                            pool.install(|| {
                                patches.into_par_iter().for_each_with(
                                    sender,
                                    |sender, (tx, ty, px, py, pw, ph)| {
                                        if cv.load(Ordering::Relaxed) != global {
                                            return;
                                        }
                                        let patch_rect = Rect::new(
                                            (tx * TILE + px) as i32,
                                            (ty * TILE + py) as i32,
                                            pw as i32,
                                            ph as i32,
                                        );
                                        emit_tile(
                                            sender, &buf, layer_id, global, mip, tx, ty, rect,
                                            patch_rect, &dev, &q,
                                        );
                                    },
                                );
                            });
                        }
                        Err(e) => {
                            tracing::error!("pull error on patch rect {:?}: {:?}", rect, e);
                        }
                    }
                }
            }
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
    }
}

/// Emits the `target` sub-rectangle of a tightly-packed GPU buffer covering
/// `region` (both in mip-pixel coordinates) as a `FetchTask` for tile
/// `(tx, ty)`. `region` and `target` are the same rect for full fetches;
/// `target` is a sub-rect of `region` for patch fetches.
#[allow(clippy::too_many_arguments)]
fn emit_tile(
    sender: &Sender<FetchTask>,
    buf: &Arc<GpuBuffer>,
    layer_id: u64,
    version: u64,
    mip: u32,
    tx: u32,
    ty: u32,
    region: Rect,
    target: Rect,
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
) {
    let _span = tracing::trace_span!("tile.emit").entered();
    let start_x = target.x.max(region.x);
    let start_y = target.y.max(region.y);
    let end_x = (target.x + target.width).min(region.x + region.width);
    let end_y = (target.y + target.height).min(region.y + region.height);
    if start_x >= end_x || start_y >= end_y {
        return;
    }
    let int_w = (end_x - start_x) as u32;
    let int_h = (end_y - start_y) as u32;

    // Destination offset inside the tile, source offset inside the region buffer.
    let out_px = (start_x - (tx * TILE) as i32) as u32;
    let out_py = (start_y - (ty * TILE) as i32) as u32;

    let src_x = (start_x - region.x) as usize;
    let src_y = (start_y - region.y) as usize;

    const BPP: u32 = 4; // straight-alpha RGBA8 (ensure_rgba8)
    let src_row_bytes = region.width as u32 * BPP;
    let offset = src_y as u64 * src_row_bytes as u64 + src_x as u64 * BPP as u64;

    let final_buf = buf.buffer.clone();

    let aligned = src_row_bytes.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        && offset.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64);

    let kind = if aligned {
        FetchPayload::Staged {
            buffer: final_buf,
            offset,
            bytes_per_row: src_row_bytes,
        }
    } else {
        FetchPayload::Raw {
            buffer: final_buf,
            offset,
            src_row_bytes,
            bpp: BPP,
        }
    };
    let _ = sender.send(FetchTask {
        layer_id,
        version,
        mip,
        tx,
        ty,
        slot_offset_x: out_px,
        slot_offset_y: out_py,
        width: int_w,
        height: int_h,
        kind,
    });
}
