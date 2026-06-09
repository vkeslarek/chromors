use crate::backend::{Backend, HistogramTargetCapability, ImageTargetCapability};
use crate::data::histogram::Histogram;
use crate::data::image::Image;
use crate::error::Error;
use crate::geometry::Rect;
use crate::pixel::PixelMeta;

// ── Materialized Data Wrappers ────────────────────────────────────────────────

/// A physical, materialized 2D image buffer, strongly typed with semantic metadata.
pub struct MaterializedImage<B: Backend> {
    pub buffer: B::Buffer,
    pub meta: PixelMeta,
    pub rect: Rect,
    /// The bounding box within `buffer` that corresponds to the pixels for `rect`.
    pub buffer_rect: Rect,
}

/// A physical, materialized 1D histogram buffer.
pub struct MaterializedHistogram<B: Backend> {
    pub buffer: Vec<u8>,
    pub bins: u32,
    pub _marker: std::marker::PhantomData<B>,
}

// ── Strongly Typed Frontend Targets ───────────────────────────────────────────

/// A logical output sink for 2D Images.
///
/// Use this target to force the evaluation of an `Image` graph and
/// extract a concrete pixel buffer.
pub struct ImageTarget<B: Backend> {
    image: Image<B>,
}

impl<B: Backend> ImageTarget<B>
where
    B: ImageTargetCapability,
{
    pub fn new(image: Image<B>) -> Self {
        Self { image }
    }

    /// Evaluates the subgraph bound to this target and extracts the requested region.
    pub fn pull(&self, rect: Rect, lod: u32) -> Result<MaterializedImage<B>, Error> {
        B::pull_image(&self.image.handle, rect, lod)
    }

    /// Evaluates the subgraph bound to this target and extracts the requested regions in a batch.
    pub fn pull_batch(&self, rects: &[Rect], lod: u32) -> Result<Vec<MaterializedImage<B>>, Error> {
        B::pull_image_batch(&self.image.handle, rects, lod)
    }
}

/// A logical output sink for Histograms.
///
/// Use this target to force the evaluation of a `Histogram` graph and
/// extract the bin counts.
pub struct HistogramTarget<B: Backend + HistogramTargetCapability> {
    histogram: Histogram<B>,
}

impl<B: Backend> HistogramTarget<B>
where
    B: HistogramTargetCapability,
{
    pub fn new(histogram: Histogram<B>) -> Self {
        Self { histogram }
    }

    /// Evaluates the histogram reduction and extracts the bins array.
    pub fn pull(&self) -> Result<MaterializedHistogram<B>, Error> {
        B::pull_histogram(&self.histogram.handle)
    }
}
