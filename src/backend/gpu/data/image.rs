use std::sync::{Arc, Mutex};
use crate::backend::Operation;
use crate::color::space::ColorSpace;
use crate::data::image::Image;
use crate::pixel::{AlphaPolicy, PixelFormat, PixelMeta};
use crate::geometry::Rect;
use crate::error::Error;

use super::super::buffer::ImageBuffer;
use super::super::context::GpuContext;
use super::super::graph::{Graph, NodeId};
use super::super::materialize::MaterializePlan;
use super::super::value::{GraphValue, ValueKind};
use super::super::{GpuBackend, GpuImageHandle, GraphNodeHandle, Lod};
use super::super::op::GpuOperation;
use super::GpuData;

impl Image<GpuBackend> {
    pub fn width(&self) -> u32 {
        self.handle.width
    }
    pub fn height(&self) -> u32 {
        self.handle.height
    }
    pub fn format(&self) -> PixelFormat {
        self.handle.format
    }
    pub fn color_space(&self) -> ColorSpace {
        self.handle.color_space
    }
    pub fn graph(&self) -> &Arc<Mutex<Graph>> {
        &self.handle.node.graph
    }
    pub fn root_id(&self) -> NodeId {
        self.handle.node.root_id
    }

    /// Width at a given MIP level.
    pub fn width_at_mip(&self, mip: u32) -> u32 {
        (self.handle.width as f64 / Lod(mip).scale_factor()).ceil() as u32
    }

    /// Height at a given MIP level.
    pub fn height_at_mip(&self, mip: u32) -> u32 {
        (self.handle.height as f64 / Lod(mip).scale_factor()).ceil() as u32
    }

    /// Create a clone of this image with a deeply cloned graph.
    /// Subsequent operations on the forked image will not pollute the original graph.
    pub fn fork(&self) -> Self {
        let mut new_graph = self.handle.node.graph.lock().unwrap().clone();
        new_graph.salt_fork();
        Image::from_handle(GpuImageHandle {
            width: self.handle.width,
            height: self.handle.height,
            format: self.handle.format,
            color_space: self.handle.color_space,
            node: GraphNodeHandle {
                graph: Arc::new(Mutex::new(new_graph)),
                root_id: self.handle.node.root_id,
                ctx: self.handle.node.ctx.clone(),
            },
        })
    }

    /// Execute a GPU operation that produces an image output.
    ///
    /// For operations that produce non-image outputs (histograms etc.) or have
    /// special cross-graph semantics (composite), use the op's own `apply()`
    /// method directly.
    pub fn execute<O: GpuOperation + Clone + 'static>(
        &self,
        op: &O,
    ) -> Result<Image<GpuBackend>, crate::error::Error> {
        let spec = op.output_spec(self.handle.width, self.handle.height);
        let (w, h) = spec.image_dims().ok_or_else(|| {
            crate::Error::Gpu(format!(
                "{:?} does not produce an Image output — use op.apply(&image) instead",
                op
            ))
        })?;

        let self_arc: Arc<dyn GpuOperation> = Arc::new(op.clone());
        let node_id = {
            let mut graph = self.handle.node.graph.lock().unwrap();
            op.emit(self.root_id(), &mut graph, self_arc)
        };
        let (out_cs, out_fmt) = {
            // If the op overrides the output codec (e.g. ColorConvertOp), use that.
            // Otherwise inherit the parent image's color space and default to RgbaF32.
            match op.output_codec_override() {
                Some(codec) => (codec.color_space, codec.format),
                None => (self.handle.color_space, PixelFormat::RgbaF32),
            }
        };
        let out_image: Image<GpuBackend> = Image::from_handle(GpuImageHandle {
            node: GraphNodeHandle {
                graph: self.handle.node.graph.clone(),
                root_id: node_id,
                ctx: self.handle.node.ctx.clone(),
            },
            width: w,
            height: h,
            format: out_fmt,
            color_space: out_cs,
        });

        Ok(out_image)
    }

    /// Speculatively compile the shader for this image's current graph root at
    /// the given LOD, so that the first `materialize()` call is not blocked by
    /// slangc.  Returns immediately — compilation happens on a background thread
    /// and is bounded by the global compile-slot semaphore.
    ///
    /// Call this once after the full filter chain is assembled (not after every
    /// individual `execute()`), and only for the final image that will actually
    /// be rendered.
    /// Materialize the image at the given LOD, download pixel bytes to CPU.
    /// Returns `(bytes, width, height)` where `bytes` is in the image's current
    /// `PixelFormat` (row-major, tightly packed).  Primarily used for scopes
    /// and thumbnail generation.
    pub fn download_pixels_at_lod(
        &self,
        lod: Lod,
    ) -> Result<(Vec<u8>, u32, u32), crate::error::Error> {
        let ctx = self.handle.node.ctx.clone();
        let region = crate::backend::gpu::region::GpuRegion::new(
            self.handle.node.graph.clone(),
            self.handle.node.ctx.cache.clone(),
            self.handle.node.root_id,
            ctx.clone(),
            lod,
        );
        let scale = 1.0 / lod.scale_factor();
        let w = (self.handle.width as f64 * scale).ceil() as i32;
        let h = (self.handle.height as f64 * scale).ceil() as i32;
        region.prepare(crate::geometry::Rect::new(0, 0, w, h));
        let mat = region
            .materialize()
            .map_err(|e| crate::error::Error::Gpu(format!("{e:?}")))?;
        let bytes = mat
            .read_bytes(&ctx)
            .map_err(|e| crate::error::Error::Gpu(format!("{e:?}")))?;
        Ok((bytes, w as u32, h as u32))
    }

    pub fn warmup_at_lod(&self, lod: u32) {
        let handle = self.handle.node.clone();
        let lod = Lod(lod);
        std::thread::spawn(move || {
            handle.warmup(lod);
        });
    }
}

impl<T: GpuOperation + Clone + 'static> Operation<Image<GpuBackend>> for T {
    type Output = Image<GpuBackend>;

    fn execute(&self, image: &Image<GpuBackend>) -> Result<Image<GpuBackend>, crate::error::Error> {
        image.execute(self)
    }
}

// ── ImageData ─────────────────────────────────────────────────────────────────

/// Request type for image data. `(lod, rect)` in LOD space.
pub type ImageRequest = (Lod, Rect);

/// [`GpuData`] impl for 2-D pixel images.
pub struct ImageData {
    /// Output color space (derived from the last node's codec override or source).
    pub dst_color_space: ColorSpace,
    /// Output pixel format.
    pub dst_format: PixelFormat,
}

impl ImageData {
    pub fn new(dst_color_space: ColorSpace, dst_format: PixelFormat) -> Self {
        Self {
            dst_color_space,
            dst_format,
        }
    }
}

impl GpuData for ImageData {
    type Value = Arc<ImageBuffer>;
    type Request = ImageRequest;

    fn value_kind(_req: &Self::Request) -> ValueKind {
        ValueKind::Image
    }

    fn plan(graph: &Graph, root: NodeId, req: &Self::Request) -> MaterializePlan {
        let (lod, rect) = *req;
        graph.materialize(&[(root, rect)], lod)
    }

    fn finish(
        &self,
        value: &GraphValue,
        req: &Self::Request,
        _ctx: &GpuContext,
    ) -> Result<Self::Value, Error> {
        let (_lod, rect) = *req;
        let meta = PixelMeta::new(self.dst_format, self.dst_color_space, AlphaPolicy::Straight);
        match value {
            GraphValue::Image { buffer, .. } => Ok(ImageBuffer::from_raw(
                buffer.buffer.buffer.clone(),
                rect.width as u32,
                rect.height as u32,
                meta,
            )),
            GraphValue::Raw { .. } => Err(Error::Gpu(
                "ImageData::finish: expected Image, got Raw".into(),
            )),
        }
    }
}
