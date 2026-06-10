pub mod gpu;
pub mod raw;
pub mod vips;

use crate::data::image::Image2D;
use crate::error::Error;

/// Marker trait for image processing backends.
///
/// Implement this on a plain unit struct to introduce a new backend.
/// `Handle` is the backend-specific resource handle behind `Image2D<B>` — not
/// necessarily image-shaped data itself (e.g. `GpuBackend::Handle` wraps a
/// graph-node reference). Other typed wrappers (`Histogram<B>`, …) carry
/// their own associated handle types (e.g. `HistogramTargetCapability::HistogramHandle`).
pub trait Backend: Sized + Send + Sync + 'static {
    type Handle: Send + Sync;
    type Buffer: Send + Sync;
}

/// Backend capability: open an image from a filesystem path.
///
/// Enables `Image2D::<B>::open`.
pub trait OpenFile: Backend {
    fn open_file(path: &str) -> Result<Self::Handle, Error>;
}

/// Backend capability: decode an encoded byte buffer (JPEG, PNG, …).
///
/// Enables `Image2D::<B>::from_buffer`.
pub trait OpenBuffer: Backend {
    fn open_buffer(data: &[u8]) -> Result<Self::Handle, Error>;
}

/// An image operation that backend `B` knows how to execute.
///
/// `Operation::execute` is gated on this trait — if `B` does not implement
/// `Operation<YourOp>`, the call fails to compile.
pub trait Operation<Input> {
    type Output;
    fn execute(&self, input: &Input) -> Result<Self::Output, Error>;
}

/// Backend capability: open an image from a stream source.
///
/// Enables `Image2D::<B>::new_from_source`.
pub trait SourceInput: Backend {
    type Source: Send + Sync;
    fn open_source(source: &Self::Source) -> Result<Self::Handle, Error>;
}

/// Backend capability: write an image to a stream target.
///
/// Enables `Image2D::<B>::write_to_target`.
pub trait TargetOutput<Input>: Backend {
    type Target: Send + Sync;
    fn write_to_target(input: &Input, target: &Self::Target) -> Result<(), Error>;
}

pub trait ImageTargetCapability: Backend {
    fn pull_image(
        handle: &Self::Handle,
        rect: crate::geometry::Rect,
        lod: u32,
    ) -> Result<crate::target::MaterializedImage<Self>, Error>;

    fn pull_image_batch(
        handle: &Self::Handle,
        rects: &[crate::geometry::Rect],
        lod: u32,
    ) -> Result<Vec<crate::target::MaterializedImage<Self>>, Error>;
}

/// Backend capability: extract a materialized 1D histogram buffer.
pub trait HistogramTargetCapability: Backend {
    type HistogramHandle: Clone + Send + Sync;

    fn create_histogram(handle: &Self::Handle) -> Result<Self::HistogramHandle, Error>;

    fn pull_histogram(
        handle: &Self::HistogramHandle,
    ) -> Result<crate::target::MaterializedHistogram<Self>, Error>;
}

impl<H: Backend> Image2D<H>
where
    H: HistogramTargetCapability,
{
    pub fn histogram(&self) -> Result<crate::data::histogram::Histogram<H>, Error> {
        let handle = H::create_histogram(&self.handle)?;
        Ok(crate::data::histogram::Histogram::from_handle(handle))
    }
}

/// Backend capability: query and convert pixel metadata (format + color space + alpha policy).
///
/// Implementing this on a backend enables the generic `Image2D<B>::pixel_meta()` and
/// `Image2D<B>::convert()` methods, making color conversion a first-class part of the
/// core image API rather than an operation.
///
/// Both VipsBackend and GpuBackend implement this.  RawBackend does not (pixel data
/// requires `materialize()` first; use `Image2D<RawBackend>::meta()` for static metadata).
pub trait ColorConversionCapability: Backend {
    /// Returns the pixel metadata (format, color space, alpha policy) for this image.
    fn pixel_meta(handle: &Self::Handle) -> crate::pixel::PixelMeta;

    /// Convert the image to the given target `PixelMeta`.
    ///
    /// A no-op conversion (same meta) must be efficient (ideally zero-copy).
    fn convert(
        handle: &Self::Handle,
        target: crate::pixel::PixelMeta,
    ) -> Result<Self::Handle, Error>;
}

impl<B: Backend + ColorConversionCapability> Image2D<B> {
    /// Returns the current pixel metadata: format, color space, and alpha policy.
    pub fn pixel_meta(&self) -> crate::pixel::PixelMeta {
        B::pixel_meta(&self.handle)
    }

    /// Convert to a different pixel format, color space, or alpha policy.
    ///
    /// Returns a new image; `self` is unchanged.
    pub fn convert(&self, target: crate::pixel::PixelMeta) -> Result<Self, Error> {
        Ok(Image2D::from_handle(B::convert(&self.handle, target)?))
    }
}
