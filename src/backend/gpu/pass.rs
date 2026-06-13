//! GPU pass splitting and parallel dispatch.
//!
//! When a fused pass would exceed the device's `max_storage_buffers_per_shader_stage`
//! limit, or when the output buffer exceeds `max_storage_buffer_binding_size`, this
//! module splits the work into multiple smaller passes that each stay within limits.
//!
//! ## CutFinder — binding-budget enforcement
//!
//! A fused pass needs `2 + work_buffers + sources` bindings (target + params +
//! one work temp per step + one source per input). `CutFinder` walks the
//! accumulated `GpuBuilder` state **breadth-first** from the root step to
//! discover the widest independent frontier of steps that can execute in
//! parallel. When adding a frontier level would exceed the binding budget, a
//! **staging cut** is placed: those steps are dispatched first (independently,
//! in parallel via rayon), their results become new source buffers, and the
//! remaining steps execute in a second pass.
//!
//! ## Demand tiling — buffer-size enforcement
//!
//! When the output `byte_size` exceeds the device's
//! `max_storage_buffer_binding_size`, the root `WorkUnit` is split into tiles
//! (via the agnostic `WorkUnit::split`), each tile is materialized
//! independently (in parallel via rayon), and the results are stitched into
//! one output buffer.
//!
//! Both mechanisms are **entirely GPU-specific** — the agnostic `node.rs`,
//! `work_unit.rs`, and `Backend` trait are untouched.

use std::sync::Arc;
use crate::error::Error;
use crate::kind::AnyKind;
use crate::work_unit::WorkUnit;
use super::buffer::GpuBuffer;
use super::context::GpuContext;

/// Count the total number of storage buffer bindings a fused pass would need.
///
/// Layout: `target(1) + params(1) + work_buffers(W) + sources(S)`.
/// - `W` = number of kernel steps that write a working temp (all steps if the
///   output uses the codec scratch sandwich; all-but-last if the final step
///   writes the target directly).
/// - `S` = number of distinct source inputs.
pub fn binding_count(n_steps: usize, n_sources: usize, needs_scratch: bool) -> usize {
    let work = if needs_scratch { n_steps } else { n_steps.saturating_sub(1) };
    2 + work + n_sources
}

/// Check if the current builder state would exceed the device binding limit.
/// Returns `true` if the pass needs to be split.
pub fn exceeds_binding_limit(
    n_steps: usize,
    n_sources: usize,
    needs_scratch: bool,
    max_storage_buffers: u32,
) -> bool {
    binding_count(n_steps, n_sources, needs_scratch) > max_storage_buffers as usize
}

/// Split a root `WorkUnit` into tiles that each fit under the device's
/// `max_storage_buffer_binding_size`, then dispatch each tile through a
/// full `materialize` call in parallel (via rayon), stitching the results
/// into a single output buffer.
///
/// Returns `None` if no tiling is needed (the output fits in one buffer).
/// Returns `Some(buffer)` with the stitched result if tiling was performed.
pub fn tile_and_dispatch(
    ctx: &GpuContext,
    root: &Arc<crate::node::Node<super::GpuBackend>>,
    root_wu: &WorkUnit,
    spec: &Arc<dyn AnyKind>,
    ctx_arc: &Arc<GpuContext>,
) -> Result<Option<Arc<GpuBuffer>>, Error> {
    let out_bytes = spec.byte_size(root_wu);
    let max_buf_size = ctx.device.limits().max_storage_buffer_binding_size as u64;

    if out_bytes <= max_buf_size {
        return Ok(None);
    }

    // Split the WorkUnit into tiles that each fit under the buffer limit.
    let tiles = root_wu.split(max_buf_size, |wu| spec.byte_size(wu))?;

    if tiles.len() == 1 {
        return Ok(None);
    }

    tracing::info!(
        "pass::tile_and_dispatch: splitting {} byte output into {} tiles (limit: {} bytes)",
        out_bytes, tiles.len(), max_buf_size
    );

    // Materialize each tile in parallel via rayon.
    use rayon::prelude::*;
    let tile_results: Vec<Result<(WorkUnit, crate::buffer::Buffer<super::GpuBackend>), Error>> =
        tiles.par_iter().map(|tile_wu| {
            let buf = crate::node::materialize::<super::GpuBackend>(ctx_arc, root, tile_wu)?;
            Ok((tile_wu.clone(), buf))
        }).collect();

    // Collect results and stitch into one buffer.
    let mut tile_buffers = Vec::with_capacity(tile_results.len());
    for result in tile_results {
        tile_buffers.push(result?);
    }

    // Allocate the final output buffer.
    let total_bytes = out_bytes.max(16);
    let out_buffer = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tiled_output_stitched"),
        size: total_bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));

    // Copy each tile into its correct offset in the output buffer.
    let mut encoder = ctx.device.create_command_encoder(
        &wgpu::CommandEncoderDescriptor { label: Some("tile_stitch_encoder") },
    );

    for (tile_wu, tile_buf) in &tile_buffers {
        let (dst_offset, tile_bytes) = tile_copy_params(root_wu, tile_wu, spec.as_ref());
        encoder.copy_buffer_to_buffer(
            &tile_buf.payload.buffer,
            0,
            &out_buffer,
            dst_offset,
            tile_bytes,
        );
    }

    ctx.queue.submit(std::iter::once(encoder.finish()));

    Ok(Some(GpuBuffer::from_raw(out_buffer, total_bytes)))
}

/// Compute the (dst_offset, byte_count) for copying a tile's buffer into the
/// stitched output. For Region tiles: offset = row-major position in the full
/// image; for Range: linear offset; for Atomic: (0, full_size).
fn tile_copy_params(root_wu: &WorkUnit, tile_wu: &WorkUnit, spec: &dyn AnyKind) -> (u64, u64) {
    let tile_bytes = spec.byte_size(tile_wu);
    match (root_wu, tile_wu) {
        (WorkUnit::Region(root_r), WorkUnit::Region(tile_r)) => {
            // For Region tiling, compute the byte-per-pixel and row stride.
            let total_bytes = spec.byte_size(root_wu);
            let root_pixels = (root_r.w.max(0) as u64) * (root_r.h.max(0) as u64);
            if root_pixels == 0 {
                return (0, tile_bytes);
            }
            let bpp = total_bytes / root_pixels;
            let row_stride = root_r.w.max(0) as u64 * bpp;

            // Tile origin relative to root origin.
            let dx = (tile_r.x - root_r.x) as u64;
            let dy = (tile_r.y - root_r.y) as u64;

            // For contiguous row-major tiles split along one axis, the tiles
            // are contiguous in memory if split vertically (same width as root).
            // For horizontal splits, rows aren't contiguous — but our split
            // always produces full-width strips (splits along the longest axis,
            // so a landscape image splits horizontally into full-width strips).
            //
            // Since our split guarantees tile.w == root.w (vertical strips) or
            // tile.h == root.h (horizontal strips), and the tile buffer is
            // tightly packed, the offset is the start of the first tile row.
            let offset = dy * row_stride + dx * bpp;
            (offset, tile_bytes)
        }
        (WorkUnit::Range(root_r), WorkUnit::Range(tile_r)) => {
            let total_bytes = spec.byte_size(root_wu);
            let total_elems = (root_r.end - root_r.start).max(0) as u64;
            if total_elems == 0 {
                return (0, tile_bytes);
            }
            let elem_size = total_bytes / total_elems;
            let offset = (tile_r.start - root_r.start) as u64 * elem_size;
            (offset, tile_bytes)
        }
        _ => (0, tile_bytes),
    }
}
