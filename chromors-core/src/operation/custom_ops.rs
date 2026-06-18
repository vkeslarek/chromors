// ── HistogramSink ─────────────────────────────────────────────────────────────

/// Counts per intensity (0..=255) for each band.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CustomHistogram {
    /// `bins[band][value]` = number of pixels with that value in that band.
    pub bins: Vec<[u32; 256]>,
}

impl CustomHistogram {
    pub fn count(&self, band: usize) -> u64 {
        self.bins
            .get(band)
            .map_or(0, |b| b.iter().map(|&c| c as u64).sum())
    }
}

#[derive(Clone)]
pub struct HistogramSink;

// ── VectorscopeSink ──────────────────────────────────────────────────────────

pub const VECTORSCOPE_GRID: usize = 128;

#[derive(Clone)]
pub struct VectorscopeSink;

pub fn vectorscope_from_rgba8(bytes: &[u8], bpp: usize) -> Vec<u32> {
    if bpp < 3 || bytes.is_empty() {
        return vec![0; VECTORSCOPE_GRID * VECTORSCOPE_GRID];
    }
    let mut grid = vec![0u32; VECTORSCOPE_GRID * VECTORSCOPE_GRID];
    let n = (VECTORSCOPE_GRID - 1) as f32;
    for px in bytes.chunks_exact(bpp) {
        let r = px[0] as f32 / 255.0;
        let g = px[1] as f32 / 255.0;
        let b = px[2] as f32 / 255.0;
        let cb = -0.168736 * r - 0.331264 * g + 0.5 * b;
        let cr = 0.500000 * r - 0.418688 * g - 0.081312 * b;
        let gx = ((cb + 0.5) * n) as usize;
        let gy = ((cr + 0.5) * n) as usize;
        grid[gy.min(VECTORSCOPE_GRID - 1) * VECTORSCOPE_GRID + gx.min(VECTORSCOPE_GRID - 1)] += 1;
    }
    grid
}

// ── Invert (Custom) ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct CustomInvert;

// ── Checkerboard ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Checkerboard {
    pub square_size: u32,
    pub dark: u8,
    pub light: u8,
}

impl Default for Checkerboard {
    fn default() -> Self {
        Self {
            square_size: 8,
            dark: 198,
            light: 209,
        }
    }
}
