use super::GpuBackend;
use super::datatype::Targetable;
use super::handle::GraphNodeHandle;
use super::handle::Lod;
use super::value::{MaterializedValue, Storage};
use crate::backend::gpu::datatype::Executable;
use crate::backend::{ColorConversionCapability, HistogramTargetCapability, ImageTargetCapability};
use crate::geometry::Rect;
use crate::target::{MaterializedHistogram, MaterializedImage};

// ── Target Capabilities ───────────────────────────────────────────────────────

/// Lift a `MaterializedValue` to a `MaterializedImage`, using the handle's
/// pixel metadata. Returns `Err` if storage is `Host` (not an image buffer).
fn mat_to_image(
    mat: &std::sync::Arc<MaterializedValue>,
    handle: &GraphNodeHandle,
    rect: Rect,
) -> Result<MaterializedImage<GpuBackend>, crate::Error> {
    match &mat.storage {
        Storage::Vram(buffer) => {
            // Translate `rect` (source coordinates) into coordinates local to
            // the materialized buffer, which tightly covers `mat`'s extent.
            let buf_rect = mat.region().map(|r| r.rect).unwrap_or(rect);
            let buffer_rect = Rect::new(
                rect.x - buf_rect.x,
                rect.y - buf_rect.y,
                rect.width,
                rect.height,
            );
            Ok(MaterializedImage {
                buffer: buffer.clone(),
                meta: <GpuBackend as ColorConversionCapability>::pixel_meta(handle),
                rect,
                buffer_rect,
            })
        }
        Storage::Host(_) => Err(crate::Error::Gpu(
            "Expected Vram (image) storage but got Host".into(),
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
            handle.graph.clone(),
            handle.ctx.cache.clone(),
            handle.root_id,
            handle.ctx.clone(),
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
            handle.graph.clone(),
            handle.ctx.cache.clone(),
            handle.root_id,
            handle.ctx.clone(),
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
        Ok(crate::backend::gpu::HistogramType::execute(&op, handle))
    }

    fn pull_histogram(
        handle: &Self::HistogramHandle,
    ) -> Result<MaterializedHistogram<Self>, crate::Error> {
        let bins = {
            let graph = handle.graph.lock().unwrap();
            let node = graph.get_node(handle.root_id);
            match node.and_then(|n| {
                n.datatype
                    .as_any()
                    .downcast_ref::<crate::backend::gpu::HistogramType>()
            }) {
                Some(hist) => hist.bins,
                None => return Err(crate::Error::Gpu("not a histogram node".into())),
            }
        };
        let hist_type = crate::backend::gpu::HistogramType { bins };
        let data = hist_type
            .pull(handle, Lod::FULL, &crate::backend::gpu::work_unit::Atomic)
            .map_err(|e| crate::Error::Gpu(format!("{:?}", e)))?;
        Ok(MaterializedHistogram {
            _marker: std::marker::PhantomData,
            buffer: data.as_bytes().to_vec(),
            bins,
        })
    }
}
