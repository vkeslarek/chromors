use std::sync::Arc;

use super::OperationBoolean;
use crate::backend::gpu::graph::{Graph, NodeId, GraphNode, NodeEval, KernelSpec};
use crate::backend::gpu::value::ValueKind;
use crate::backend::Backend;
use crate::geometry::Rect;
use crate::backend::gpu::op::GpuOperation;
use crate::backend::gpu::op::emit_image;
use crate::backend::gpu::param::Param;
use crate::backend::vips::IntoVipsEnum;
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

#[derive(Debug, Clone)]
pub struct BandboolOperation {
    pub boolean: OperationBoolean,
    pub bands: u32,
}
impl VipsOperation for BandboolOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"bandbool\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("boolean", self.boolean.into_vips());
    }
}
impl GpuOperation for BandboolOperation {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_image(
            graph,
            input,
            self_arc,
            "ops.bands",
            "bandbool_kernel",
            vec![Param::U32(self.boolean as u32), Param::U32(self.bands)],
        )
    }
}

#[derive(Debug, Clone)]
pub struct BandfoldOperation {
    pub factor: u32,
}
impl VipsOperation for BandfoldOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"bandfold\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
        o.set_int("factor", self.factor as i32);
    }
}
impl GpuOperation for BandfoldOperation {
    fn output_spec(&self, input_w: u32, input_h: u32) -> crate::backend::gpu::op::OutputSpec {
        crate::backend::gpu::op::OutputSpec::Image {
            width: input_w,
            height: input_h * self.factor,
        }
    }
    fn inverse_map(
        &self,
        output_rect: crate::geometry::Rect,
        _w: u32,
        _h: u32,
        _lod: crate::backend::gpu::handle::Lod,
    ) -> Vec<(usize, crate::geometry::Rect)> {
        let f = self.factor as i32;
        let y = output_rect.y / f;
        let h = ((output_rect.y + output_rect.height + f - 1) / f) - y;
        vec![(0, crate::geometry::Rect::new(output_rect.x, y, output_rect.width, h))]
    }
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_image(
            graph,
            input,
            self_arc,
            "ops.bands",
            "bandfold_kernel",
            vec![Param::U32(self.factor)],
        )
    }
}

#[derive(Debug, Clone)]
pub struct BandunfoldOperation {
    pub factor: u32,
}
impl VipsOperation for BandunfoldOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"bandunfold\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
        o.set_int("factor", self.factor as i32);
    }
}
impl GpuOperation for BandunfoldOperation {
    fn output_spec(&self, input_w: u32, input_h: u32) -> crate::backend::gpu::op::OutputSpec {
        crate::backend::gpu::op::OutputSpec::Image {
            width: input_w,
            height: input_h / self.factor,
        }
    }
    fn inverse_map(
        &self,
        output_rect: crate::geometry::Rect,
        _w: u32,
        _h: u32,
        _lod: crate::backend::gpu::handle::Lod,
    ) -> Vec<(usize, crate::geometry::Rect)> {
        let f = self.factor as i32;
        vec![(0, crate::geometry::Rect::new(output_rect.x, output_rect.y * f, output_rect.width, output_rect.height * f))]
    }
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_image(
            graph,
            input,
            self_arc,
            "ops.bands",
            "bandunfold_kernel",
            vec![Param::U32(self.factor)],
        )
    }
}

#[derive(Debug, Clone)]
pub struct BandmeanOperation {
    pub bands: u32,
}
impl VipsOperation for BandmeanOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"bandmean\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}
impl GpuOperation for BandmeanOperation {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_image(
            graph,
            input,
            self_arc,
            "ops.bands",
            "bandmean_kernel",
            vec![Param::U32(self.bands)],
        )
    }
}

// ── GPU channel operations ────────────────────────────────────────────────────

/// Extracts band `band` and replicates it to all four output channels.
///
/// Output pixel: `float4(val, val, val, 1.0)` where `val = input.channel[band]`.
///
/// Vips equivalent: `ExtractBandOperation { band, count: Some(1) }` (returns a
/// 1-band image; the GPU version keeps RGBA with replicated values for
/// format-uniform comparisons).
#[derive(Debug, Clone)]
pub struct ExtractBandGpuOp {
    pub band: u32,
}

impl GpuOperation for ExtractBandGpuOp {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_image(
            graph,
            input,
            self_arc,
            "ops.bands",
            "extract_band_kernel",
            vec![Param::U32(self.band)],
        )
    }
}

/// Multiplies band `band` by `factor`, leaving all other channels unchanged.
///
/// Equivalent to the Vips chain:
///   `extract_band(0..band-1) + linear(band, factor) + bandjoin`
/// but emitted as a single fused GPU kernel — no intermediate copies.
///
/// `scale_band(band=3, factor=0.5)` is identical to `OpacityOperation(0.5)`.
#[derive(Debug, Clone)]
pub struct ScaleBandGpuOp {
    pub band: u32,
    pub factor: f32,
}

impl GpuOperation for ScaleBandGpuOp {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_image(
            graph,
            input,
            self_arc,
            "ops.bands",
            "scale_band_kernel",
            vec![Param::U32(self.band), Param::F32(self.factor)],
        )
    }
}

/// Adds a constant offset to band `band`, clamping the result to [0, 1].
///
/// Equivalent to extracting the band, applying `linear(a=1, b=offset)`, and
/// joining back — expressed here as a single fused GPU kernel.
#[derive(Debug, Clone)]
pub struct AddToBandGpuOp {
    pub band: u32,
    pub offset: f32,
}

impl GpuOperation for AddToBandGpuOp {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_image(
            graph,
            input,
            self_arc,
            "ops.bands",
            "add_to_band_kernel",
            vec![Param::U32(self.band), Param::F32(self.offset)],
        )
    }
}

#[derive(Debug, Clone)]
pub struct ExtractBandOperation {
    pub band: i32,
    pub count: Option<i32>,
}
impl VipsOperation for ExtractBandOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"extract_band\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("band", self.band);
        if let Some(v) = self.count {
            op.set_int("n", v);
        }
    }
}

impl GpuOperation for ExtractBandOperation {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        let count = self.count.unwrap_or(1).max(1) as u32;
        if count == 1 {
            emit_image(
                graph,
                input,
                self_arc,
                "ops.bands",
                "extract_band_kernel",
                vec![Param::U32(self.band as u32)],
            )
        } else {
            emit_image(
                graph,
                input,
                self_arc,
                "ops.bands",
                "extract_band_range_kernel",
                vec![Param::U32(self.band as u32), Param::U32(count)],
            )
        }
    }
    fn output_spec(&self, w: u32, h: u32) -> crate::backend::gpu::op::OutputSpec {
        crate::backend::gpu::op::OutputSpec::Image {
            width: w,
            height: h,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BandjoinOperation — joins N images band-wise into one output image.
// ═══════════════════════════════════════════════════════════════════════════════

pub struct BandjoinOperation<B: Backend> {
    pub images: Vec<crate::data::image::Image<B>>,
}

impl<B: Backend> std::fmt::Debug for BandjoinOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BandjoinOperation")
            .field("count", &self.images.len())
            .finish()
    }
}

impl<B: Backend> Clone for BandjoinOperation<B>
where
    B::Handle: Clone,
{
    fn clone(&self) -> Self {
        Self {
            images: self.images.clone(),
        }
    }
}

// Vips path — vips_bandjoin takes an array of images directly.
impl crate::backend::Operation<
    crate::data::image::Image<crate::backend::vips::VipsBackend>,
> for BandjoinOperation<crate::backend::vips::VipsBackend>
{
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;

    fn execute(
        &self,
        image: &crate::data::image::Image<crate::backend::vips::VipsBackend>,
    ) -> Result<Self::Output, crate::error::Error> {
        let n = self.images.len() + 1;
        let mut ptrs: Vec<*mut ffi::VipsImage> = vec![image.vips_ptr()];
        for img in &self.images {
            ptrs.push(img.vips_ptr());
        }
        let mut out: *mut ffi::VipsImage = std::ptr::null_mut();
        let ret = unsafe {
            ffi::vips_bandjoin(ptrs.as_mut_ptr(), &mut out, n as i32, std::ptr::null::<std::ffi::c_void>())
        };
        if ret != 0 {
            return Err(crate::error::Error::Vips(crate::backend::vips::vips_error()));
        }
        if out.is_null() {
            return Err(crate::error::Error::Vips(
                "vips_bandjoin returned null".into(),
            ));
        }
        Ok(crate::data::image::Image::from_vips_ptr(out))
    }
}

// GPU path.
impl GpuOperation for BandjoinOperation<crate::backend::gpu::GpuBackend> {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        use crate::backend::gpu::buffer::ImageBuffer;
        use crate::backend::gpu::source::GpuSource;

        let total = self.images.len() + 1;
        assert!(
            total <= 5,
            "BandjoinOperation: max 5 inputs (got {total})"
        );

        let mut input_ids: Vec<NodeId> = vec![input];
        let mut ch_params: Vec<Param> = vec![Param::U32(0u32)];

        for img in &self.images {
            let id = if graph.get_node(img.root_id()).is_some()
                || graph.get_source(img.root_id()).is_some()
            {
                img.root_id()
            } else {
                let (w, h) = (img.width(), img.height());
                let target = crate::target::ImageTarget::new(img.clone());
                let mat = target
                    .pull(crate::geometry::Rect::new(0, 0, w as i32, h as i32), 0)
                    .expect("BandjoinOperation::emit: failed to pull image");
                let img_buf = Arc::new(ImageBuffer {
                    buffer: mat.buffer.clone(),
                    width: mat.buffer_rect.width as u32,
                    height: mat.buffer_rect.height as u32,
                    meta: mat.meta,
                });
                let source = GpuSource::new_buffer(img_buf, img.handle.node.ctx.clone());
                graph.add_source(std::sync::Arc::new(source))
            };
            input_ids.push(id);
            ch_params.push(Param::U32(0u32));
        }

        let func = match total {
            1 => "bandjoin1_kernel",
            2 => "bandjoin2_kernel",
            3 => "bandjoin3_kernel",
            4 => "bandjoin4_kernel",
            5 => "bandjoin5_kernel",
            _ => unreachable!(),
        };

        graph.add_node(GraphNode {
            id: NodeId(0),
            inputs: input_ids,
            eval: NodeEval::Kernel(KernelSpec {
                module: "ops.bands",
                function: func,
            }),
            params: ch_params,
            op: self_arc,
            output: ValueKind::Image,
        })
    }

    fn inverse_map(
        &self,
        output_rect: Rect,
        _w: u32,
        _h: u32,
        _lod: crate::backend::gpu::handle::Lod,
    ) -> Vec<(usize, Rect)> {
        (0..=self.images.len())
            .map(|i| (i, output_rect))
            .collect()
    }
}
