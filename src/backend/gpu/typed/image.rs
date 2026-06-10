//! `Image2D<GpuBackend>` — typed host-facing wrapper around a [`super::super::handle::GraphNodeHandle`]
//! whose root node produces [`super::super::datatype::ImageType`].
//!
//! Not to be confused with [`super::super::datatype::image::ImageType`] — that's
//! the graph-node datatype *tag*; this is the ergonomic handle callers hold.

use crate::backend::Operation;
use crate::color::space::ColorSpace;
use crate::data::image::Image2D;
use crate::pixel::PixelFormat;
use std::sync::{Arc, Mutex};

use super::super::graph::{Graph, NodeId};
use super::super::op::TypedOperation;
use super::super::{Executable, GpuBackend, GraphNodeHandle, ImageType, Lod};

impl Image2D<GpuBackend> {
    /// Output pixel dimensions of this image's root node, derived from the
    /// graph (see [`Graph::node_dims`]). Panics if the root node has no
    /// spatial output — the typed [`Image2D`] wrapper guarantees this in
    /// practice (only [`ImageType`]-producing ops can yield one).
    pub fn width(&self) -> u32 {
        let g = self.handle.graph.lock().unwrap();
        g.node_dims(self.handle.root_id)
            .expect("Image2D::width: root node has no spatial output")
            .0
    }

    /// See [`Self::width`].
    pub fn height(&self) -> u32 {
        let g = self.handle.graph.lock().unwrap();
        g.node_dims(self.handle.root_id)
            .expect("Image2D::height: root node has no spatial output")
            .1
    }

    pub fn format(&self) -> PixelFormat {
        self.pixel_meta().format
    }
    pub fn color_space(&self) -> ColorSpace {
        self.pixel_meta().color_space
    }
    pub fn graph(&self) -> &Arc<Mutex<Graph>> {
        &self.handle.graph
    }
    pub fn root_id(&self) -> NodeId {
        self.handle.root_id
    }

    /// Width at a given MIP level.
    pub fn width_at_mip(&self, mip: u32) -> u32 {
        (self.width() as f64 / Lod(mip).scale_factor()).ceil() as u32
    }

    /// Height at a given MIP level.
    pub fn height_at_mip(&self, mip: u32) -> u32 {
        (self.height() as f64 / Lod(mip).scale_factor()).ceil() as u32
    }

    /// Create a clone of this image with a deeply cloned graph.
    /// Subsequent operations on the forked image will not pollute the original graph.
    pub fn fork(&self) -> Self {
        let mut new_graph = self.handle.graph.lock().unwrap().clone();
        new_graph.salt_fork();
        Image2D::from_handle(GraphNodeHandle {
            graph: Arc::new(Mutex::new(new_graph)),
            root_id: self.handle.root_id,
            ctx: self.handle.ctx.clone(),
        })
    }

    /// Execute a GPU operation that produces an image output.
    ///
    /// For operations that produce non-image outputs (histograms etc.) or have
    /// special cross-graph semantics (composite), use the op's own `apply()`
    /// method directly.
    pub fn execute<O: TypedOperation<Output = ImageType> + Clone + 'static>(
        &self,
        op: &O,
    ) -> Result<Image2D<GpuBackend>, crate::error::Error> {
        Ok(Image2D::from_handle(ImageType::execute(op, &self.handle)))
    }
}

impl<T: TypedOperation<Output = ImageType> + Clone + 'static> Operation<Image2D<GpuBackend>> for T {
    type Output = Image2D<GpuBackend>;

    fn execute(
        &self,
        image: &Image2D<GpuBackend>,
    ) -> Result<Image2D<GpuBackend>, crate::error::Error> {
        image.execute(self)
    }
}
