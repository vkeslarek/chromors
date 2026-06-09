use std::marker::PhantomData;

use crate::backend::{Backend, HistogramTargetCapability};

pub struct Histogram<B: Backend + HistogramTargetCapability> {
    pub handle: B::HistogramHandle,
    _b: PhantomData<B>,
}

impl<B: Backend + HistogramTargetCapability> Histogram<B> {
    pub fn from_handle(handle: B::HistogramHandle) -> Self {
        Histogram {
            handle,
            _b: PhantomData,
        }
    }
}

impl<B: Backend + HistogramTargetCapability> Clone for Histogram<B>
where
    B::HistogramHandle: Clone,
{
    fn clone(&self) -> Self {
        Histogram {
            handle: self.handle.clone(),
            _b: PhantomData,
        }
    }
}

unsafe impl<B: Backend + HistogramTargetCapability> Send for Histogram<B> {}
unsafe impl<B: Backend + HistogramTargetCapability> Sync for Histogram<B> {}

pub struct HistogramResult {
    pub bins: Vec<u32>,
    pub total_pixels: u64,
}

impl HistogramResult {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let bins: Vec<u32> = bytes
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let total_pixels = bins.iter().map(|&b| b as u64).sum();
        Self { bins, total_pixels }
    }

    pub fn normalized(&self) -> Vec<f32> {
        let total = self.total_pixels.max(1) as f32;
        self.bins.iter().map(|&b| b as f32 / total).collect()
    }

    pub fn percentile(&self, p: f32) -> f32 {
        let target = (self.total_pixels as f32 * p.clamp(0.0, 1.0)) as u64;
        let mut acc = 0u64;
        for (i, &b) in self.bins.iter().enumerate() {
            acc += b as u64;
            if acc >= target {
                return i as f32 / (self.bins.len() - 1) as f32;
            }
        }
        1.0
    }
}
