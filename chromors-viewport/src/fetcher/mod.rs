use std::sync::Arc;

mod stage;
mod tiles;

pub(crate) use stage::stage_aligned;
pub use tiles::TileFetcher;

/// GPU-side fetch result: carries the wgpu::Buffer handle so the renderer can
/// `copy_buffer_to_texture` directly, avoiding the GPU→CPU→GPU roundtrip.
pub struct FetchTask {
    pub layer_id: u64,
    pub version: u64,
    pub mip: u32,
    pub tx: u32,
    pub ty: u32,
    pub slot_offset_x: u32,
    pub slot_offset_y: u32,
    pub width: u32,
    pub height: u32,
    pub kind: FetchPayload,
}

pub enum FetchPayload {
    /// Raw GPU buffer, not yet staged for texture upload.
    Raw {
        buffer: Arc<wgpu::Buffer>,
        offset: u64,
        src_row_bytes: u32,
        bpp: u32,
    },
    /// Pre-staged GPU buffer ready for copy_buffer_to_texture (zero-copy path).
    Staged {
        buffer: Arc<wgpu::Buffer>,
        offset: u64,
        bytes_per_row: u32,
    },
}

/// Compute a cache key that captures (mip, image identity).
/// Transform is intentionally excluded — the shader applies it in the vertex
/// stage, so tiles don't need re-fetch on position/scale changes.
pub fn compute_cache_key(mip: u32, image_version: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    h ^= mip as u64;
    h = h.wrapping_mul(0x100000001b3);
    h ^= image_version;
    h = h.wrapping_mul(0x100000001b3);
    h
}
