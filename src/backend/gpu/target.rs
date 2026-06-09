use crate::backend::TargetOutput;
/// Output handle — placeholder. Will hold a GPU buffer + metadata
/// for writing results back to CPU or to a file.
#[derive(Clone)]
pub struct GpuTarget {
    pub buffer: std::sync::Arc<std::sync::Mutex<Option<Vec<u8>>>>,
}

impl GpuTarget {
    pub fn new() -> Self {
        Self {
            buffer: std::sync::Arc::new(std::sync::Mutex::new(None)),
        }
    }
}

use super::GpuBackend;
use super::handle::GpuImageHandle;
use super::handle::Lod;
use super::value::GraphValue;
use crate::backend::gpu::op::GpuOperation;
use crate::backend::{HistogramTargetCapability, ImageTargetCapability};
use crate::data::image::Image;
use crate::geometry::Rect;
use crate::target::{MaterializedHistogram, MaterializedImage};
// ── TargetOutput ──────────────────────────────────────────────────────────────

impl TargetOutput<Image<GpuBackend>> for GpuBackend {
    type Target = GpuTarget;

    fn write_to_target(image: &Image<Self>, target: &Self::Target) -> Result<(), crate::Error> {
        let rect = Rect::new(0, 0, image.width() as i32, image.height() as i32);
        let region = crate::backend::gpu::region::GpuRegion::new(
            image.handle.node.graph.clone(),
            image.handle.node.ctx.cache.clone(),
            image.handle.node.root_id,
            image.handle.node.ctx.clone(),
            Lod::FULL,
        );
        region.prepare(rect);
        let compiled = region
            .materialize()
            .map_err(|e| crate::Error::Vips(format!("compile error: {:?}", e)))?;

        let bytes = compiled
            .read_bytes(&image.handle.node.ctx)
            .map_err(|e| crate::Error::Vips(e.to_string()))?;
        *target.buffer.lock().unwrap() = Some(bytes);
        Ok(())
    }
}

// ── Target Capabilities ───────────────────────────────────────────────────────

/// Lift a `GraphValue::Image` to a `MaterializedImage`, using the
/// handle's pixel metadata.  Returns `Err` if the buffer is `Raw`.
fn mat_to_image(
    mat: &std::sync::Arc<GraphValue>,
    handle: &GpuImageHandle,
    rect: Rect,
) -> Result<MaterializedImage<GpuBackend>, crate::Error> {
    match &**mat {
        GraphValue::Image { buffer, .. } => {
            let coords = mat.buffer_coords(rect);
            Ok(MaterializedImage {
                buffer: buffer.buffer.clone(),
                meta: handle.pixel_meta(),
                rect,
                buffer_rect: coords,
            })
        }
        GraphValue::Raw { .. } => Err(crate::Error::Gpu(
            "Expected Image buffer but got Raw".into(),
        )),
    }
}

impl ImageTargetCapability for GpuBackend {
    fn pull_image(
        handle: &Self::Handle,
        rect: Rect,
        lod: u32,
    ) -> Result<MaterializedImage<Self>, crate::Error> {
        let region = crate::backend::gpu::region::GpuRegion::new(
            handle.node.graph.clone(),
            handle.node.ctx.cache.clone(),
            handle.node.root_id,
            handle.node.ctx.clone(),
            Lod(lod),
        );
        region.prepare(rect);
        mat_to_image(&region.materialize()?, handle, rect)
    }

    fn pull_image_batch(
        handle: &Self::Handle,
        rects: &[Rect],
        lod: u32,
    ) -> Result<Vec<MaterializedImage<Self>>, crate::Error> {
        let region = crate::backend::gpu::region::GpuRegion::new(
            handle.node.graph.clone(),
            handle.node.ctx.cache.clone(),
            handle.node.root_id,
            handle.node.ctx.clone(),
            Lod(lod),
        );
        region
            .materialize_batch(rects)?
            .into_iter()
            .zip(rects.iter())
            .map(|(mat, &rect)| mat_to_image(&mat, handle, rect))
            .collect()
    }
}

impl HistogramTargetCapability for GpuBackend {
    type HistogramHandle = crate::backend::gpu::GraphNodeHandle;

    fn create_histogram(handle: &Self::Handle) -> Result<Self::HistogramHandle, crate::Error> {
        let op = crate::operation::stats::HistogramOp {
            bins: 256,
            channel: 4,
        };
        let self_arc: std::sync::Arc<dyn crate::backend::gpu::op::GpuOperation> =
            std::sync::Arc::new(op.clone());
        let node_id = {
            let mut graph = handle.node.graph.lock().unwrap();
            op.emit(handle.node.root_id, &mut graph, self_arc)
        };
        Ok(crate::backend::gpu::GraphNodeHandle {
            graph: handle.node.graph.clone(),
            root_id: node_id,
            ctx: handle.node.ctx.clone(),
        })
    }

    fn pull_histogram(
        handle: &Self::HistogramHandle,
    ) -> Result<MaterializedHistogram<Self>, crate::Error> {
        let bins = {
            let graph = handle.graph.lock().unwrap();
            let node = graph.get_node(handle.root_id);
            match node.map(|n| &n.output) {
                Some(crate::backend::gpu::value::ValueKind::Histogram { bins }) => *bins,
                _ => return Err(crate::Error::Gpu("not a histogram node".into())),
            }
        };
        let request = crate::backend::gpu::request::GpuRequest::new(
            handle.graph.clone(),
            handle.ctx.cache.clone(),
            handle.root_id,
            handle.ctx.clone(),
            Lod::FULL,
            crate::backend::gpu::data::HistogramData,
            (Lod::FULL, bins),
        );
        let data = request
            .materialize()
            .map_err(|e| crate::Error::Gpu(format!("{:?}", e)))?;
        let bytes: Vec<u8> = data.into_iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(MaterializedHistogram {
            _marker: std::marker::PhantomData,
            buffer: bytes,
            bins,
        })
    }
}
