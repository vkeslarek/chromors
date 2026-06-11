use crate::backend::vips::custom::{CustomRegion, VipsCustomOperation, VipsCustomSink};
use crate::error::Error;

// ── HistogramSink ─────────────────────────────────────────────────────────────

/// Counts per intensity (0..=255) for each band.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Histogram {
    /// `bins[band][value]` = number of pixels with that value in that band.
    pub bins: Vec<[u32; 256]>,
}

impl Histogram {
    pub fn count(&self, band: usize) -> u64 {
        self.bins
            .get(band)
            .map_or(0, |b| b.iter().map(|&c| c as u64).sum())
    }
}

#[derive(Clone)]
pub struct HistogramSink;

impl VipsCustomSink for HistogramSink {
    type Output = Histogram;
    type Acc = Histogram;

    fn fold(&self, acc: &mut Histogram, region: &CustomRegion) {
        let (_, top, _, h) = region.rect();
        let bands = region.pixel_bytes();
        if acc.bins.len() < bands {
            acc.bins.resize(bands, [0u32; 256]);
        }
        for y in top..top + h {
            for pixel in region.row(y).chunks_exact(bands) {
                for (b, &v) in pixel.iter().enumerate() {
                    acc.bins[b][v as usize] += 1;
                }
            }
        }
    }

    fn merge(&self, total: &mut Histogram, part: Histogram) {
        if total.bins.len() < part.bins.len() {
            total.bins.resize(part.bins.len(), [0u32; 256]);
        }
        for (t, p) in total.bins.iter_mut().zip(&part.bins) {
            for i in 0..256 {
                t[i] += p[i];
            }
        }
    }

    fn finish(&self, acc: Histogram) -> Histogram {
        acc
    }
}

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

impl VipsCustomSink for VectorscopeSink {
    type Output = Vec<u32>;
    type Acc = Vec<u32>;

    fn fold(&self, acc: &mut Vec<u32>, region: &CustomRegion) {
        if acc.is_empty() {
            acc.resize(VECTORSCOPE_GRID * VECTORSCOPE_GRID, 0);
        }
        let (_, top, _, h) = region.rect();
        let psize = region.pixel_bytes();
        if psize < 3 {
            return;
        }
        let n = (VECTORSCOPE_GRID - 1) as f32;
        for y in top..top + h {
            for px in region.row(y).chunks_exact(psize) {
                let r = px[0] as f32 / 255.0;
                let g = px[1] as f32 / 255.0;
                let b = px[2] as f32 / 255.0;
                let cb = -0.168736 * r - 0.331264 * g + 0.5 * b;
                let cr = 0.500000 * r - 0.418688 * g - 0.081312 * b;
                let gx = ((cb + 0.5) * n) as usize;
                let gy = ((cr + 0.5) * n) as usize;
                acc[gy.min(VECTORSCOPE_GRID - 1) * VECTORSCOPE_GRID
                    + gx.min(VECTORSCOPE_GRID - 1)] += 1;
            }
        }
    }

    fn merge(&self, total: &mut Vec<u32>, part: Vec<u32>) {
        if total.is_empty() {
            *total = part;
            return;
        }
        for (t, p) in total.iter_mut().zip(part.iter()) {
            *t += p;
        }
    }

    fn finish(&self, acc: Vec<u32>) -> Vec<u32> {
        acc
    }
}

// ── Invert (Custom) ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct CustomInvert;

impl VipsCustomOperation for CustomInvert {
    fn generate(&self, out: &mut CustomRegion, input: &CustomRegion) -> Result<(), Error> {
        let (_, top, _, h) = out.rect();
        for y in top..top + h {
            let src = input.row(y);
            let dst = out.row_mut(y);
            for (d, s) in dst.iter_mut().zip(src) {
                *d = 255 - *s;
            }
        }
        Ok(())
    }
}

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

impl VipsCustomOperation for Checkerboard {
    fn generate(&self, out: &mut CustomRegion, _input: &CustomRegion) -> Result<(), Error> {
        let (left, top, w, h) = out.rect();
        let bands = out.pixel_bytes();
        let sq = self.square_size as i32;
        for y in top..top + h {
            let dst = out.row_mut(y);
            let y_sq = y / sq;
            for x in left..left + w {
                let x_sq = x / sq;
                let is_light = ((x_sq + y_sq) & 1) != 0;
                let v = if is_light { self.light } else { self.dark };
                let off = ((x - left) as usize) * bands;
                for b in 0..bands {
                    dst[off + b] = v;
                }
            }
        }
        Ok(())
    }
}
