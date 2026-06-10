//! Placeholder custom operations, one per embedded mould, until real algorithms
//! land:
//!
//! - [`HistogramSink`] — a [`VipsCustomSink`] reduction whose output is an
//!   arbitrary Rust value (per-band [`Histogram`]); no image produced.
//! - [`Invert`] — a [`VipsCustomOperation`] producing an output `Image2D`.
//!
//! Both run region by region inside the vips pipeline (no download).
//!
//! ```ignore
//! let hist = img.sink(HistogramSink)?;     // Histogram { bins: Vec<[u32; 256]> }
//! let inv  = img.custom(Invert)?;          // Image2D
//! ```

use crate::backend::Operation;
use crate::backend::vips::{CustomRegion, VipsBackend, VipsCustomOperation, VipsCustomSink};
use crate::data::image::Image2D;
use crate::error::Error;

// ── execute() bridge ─────────────────────────────────────────────────────────
//
// `Image2D::execute` takes any `Operation<VipsBackend>`. A blanket
// `impl<T: VipsOperation> Operation<VipsBackend> for T` already exists, and Rust
// coherence forbids a second blanket (or a bare concrete impl) for the same
// backend. These local wrappers sidestep it: Rust knows `Custom<O>` / `Reduce<S>`
// are local types that don't implement `VipsOperation`, so the impls don't
// overlap. Usage: `img.execute(&Custom(Invert))`, `img.execute(&Reduce(HistogramSink))`.

/// Wraps a [`VipsCustomOperation`] so it runs through [`Image2D::execute`].
pub struct Custom<O>(pub O);

impl<O: VipsCustomOperation + Clone> Operation<Image2D<VipsBackend>> for Custom<O> {
    type Output = Image2D<VipsBackend>;

    fn execute(&self, image: &Image2D<VipsBackend>) -> Result<Self::Output, Error> {
        image.custom(self.0.clone())
    }
}

/// Wraps a [`VipsCustomSink`] so it runs through [`Image2D::execute`].
pub struct Reduce<S>(pub S);

impl<S: VipsCustomSink + Clone> Operation<Image2D<VipsBackend>> for Reduce<S> {
    type Output = S::Output;

    fn execute(&self, image: &Image2D<VipsBackend>) -> Result<Self::Output, Error> {
        image.sink(self.0.clone())
    }
}

/// Counts per intensity (0..=255) for each band.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Histogram {
    /// `bins[band][value]` = number of pixels with that value in that band.
    pub bins: Vec<[u32; 256]>,
}

impl Histogram {
    /// Total samples counted in `band` (== pixel count for a fully-scanned image).
    pub fn count(&self, band: usize) -> u64 {
        self.bins
            .get(band)
            .map_or(0, |b| b.iter().map(|&c| c as u64).sum())
    }
}

/// 8-bit per-band histogram reduction. Only meaningful for u8 formats (256
/// bins); each byte of a pixel is one band sample.
#[derive(Clone)]
pub struct HistogramSink;

impl VipsCustomSink for HistogramSink {
    type Output = Histogram;
    type Acc = Histogram;

    fn fold(&self, acc: &mut Histogram, region: &CustomRegion) {
        let (_, top, _, h) = region.rect();
        let bands = region.pixel_bytes(); // u8: 1 byte per band
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

/// Grid size for the vectorscope density map (GRID × GRID cells).
pub const VECTORSCOPE_GRID: usize = 128;

/// 2-D Cb/Cr density grid reduction (u8 images only; other formats skipped).
///
/// Maps each pixel's BT.601 Cb/Cr chrominance to a cell in a
/// [`VECTORSCOPE_GRID`]×[`VECTORSCOPE_GRID`] density grid.  Cb is the X axis
/// (blue-yellow), Cr is the Y axis (red-cyan).  Cell (0,0) is bottom-left
/// (Cb=-0.5, Cr=-0.5).
#[derive(Clone)]
pub struct VectorscopeSink;

/// Compute a vectorscope density grid from raw RGBA8 bytes (any stride ≥ 3 bpp).
/// Input: packed rows of `bpp` bytes per pixel (only first 3 = RGB are used).
/// Output: flattened [`VECTORSCOPE_GRID`]×[`VECTORSCOPE_GRID`] density `Vec<u32>`.
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

// ── Invert ───────────────────────────────────────────────────────────────────

/// Per-band 8-bit invert (`255 - x`) producing an output image. A
/// [`VipsCustomOperation`]: the output region is filled from the input region.
#[derive(Clone)]
pub struct Invert;

impl VipsCustomOperation for Invert {
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

/// Generates a checkerboard transparency pattern. Applied via `img.custom()`.
/// The input image is ignored — only the output geometry matters.
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
