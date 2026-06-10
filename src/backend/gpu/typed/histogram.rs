use std::sync::Arc;

// ── HistogramBuffer ───────────────────────────────────────────────────────────

/// GPU-computed histogram result — `bins` × u32 atomic counters, CPU-resident.
///
/// Analogous to [`super::super::buffer::ImageBuffer`]: a typed container
/// that carries the data together with its structural metadata.
pub struct HistogramBuffer {
    bytes: Vec<u8>,
    pub bins: u32,
}

impl HistogramBuffer {
    pub(crate) fn from_bytes(bytes: Vec<u8>, bins: u32) -> Arc<Self> {
        Arc::new(Self { bytes, bins })
    }

    /// View the bin counts as a typed slice.
    pub fn as_slice(&self) -> &[u32] {
        bytemuck::cast_slice(&self.bytes)
    }

    /// Raw byte view (little-endian u32 per bin).
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}
