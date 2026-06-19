use crate::prelude::*;
use chromors_core::operation::custom_ops::{
    Checkerboard, CustomHistogram, CustomInvert, HistogramSink, VECTORSCOPE_GRID, VectorscopeSink,
};

impl VipsCustomSink for HistogramSink {
    type Output = CustomHistogram;
    type Acc = CustomHistogram;

    fn fold(&self, acc: &mut CustomHistogram, region: &CustomRegion) {
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

    fn merge(&self, total: &mut CustomHistogram, part: CustomHistogram) {
        if total.bins.len() < part.bins.len() {
            total.bins.resize(part.bins.len(), [0u32; 256]);
        }
        for (t, p) in total.bins.iter_mut().zip(&part.bins) {
            for i in 0..256 {
                t[i] += p[i];
            }
        }
    }

    fn finish(&self, acc: CustomHistogram) -> CustomHistogram {
        acc
    }
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
