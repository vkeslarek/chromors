use std::marker::PhantomData;

use crate::backend::{Backend, OpenBuffer, OpenFile, SourceInput, TargetOutput};
use crate::error::Error;

/// Backend-generic image handle.
///
/// The type parameter `B` selects which backend owns this image; `B::Handle`
/// stores the actual image data.  Backend-agnostic operations (open, execute)
/// are available on any `Image<B>` with the right capability bounds.
/// Vips-specific operations live in `impl Image<VipsBackend>`.
pub struct Image<B: Backend> {
    pub handle: B::Handle,
    _b: PhantomData<B>,
}

impl<B: Backend> Image<B> {
    pub fn from_handle(handle: B::Handle) -> Self {
        Image {
            handle,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Clone for Image<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Image {
            handle: self.handle.clone(),
            _b: PhantomData,
        }
    }
}

unsafe impl<B: Backend> Send for Image<B> {}
unsafe impl<B: Backend> Sync for Image<B> {}

impl<B: Backend + OpenFile> Image<B> {
    pub fn open(path: &str) -> Result<Self, Error> {
        Ok(Image::from_handle(B::open_file(path)?))
    }
}

impl<B: Backend + OpenBuffer> Image<B> {
    pub fn from_buffer(data: &[u8]) -> Result<Self, Error> {
        Ok(Image::from_handle(B::open_buffer(data)?))
    }
}

impl<B: Backend> Image<B> {
    pub fn new_from_source(source: &B::Source) -> Result<Self, Error>
    where
        B: SourceInput,
    {
        Ok(Image::from_handle(B::open_source(source)?))
    }

    pub fn write_to_target(&self, target: &B::Target) -> Result<(), Error>
    where
        B: TargetOutput<Self>,
    {
        B::write_to_target(self, target)
    }
}
