use crate::backend::gpu::datatype::ImageType;
use crate::backend::gpu::graph::{Graph, NodeId};
use crate::backend::gpu::op::emit_image;
use crate::backend::gpu::op::{GpuOperation, TypedOperation};
use crate::backend::gpu::param::Param;
use crate::geometry::Rect;
use std::sync::Arc;

use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

#[derive(Debug, Clone)]
pub struct GaussianBlurOperation {
    pub sigma: f64,
    pub minimum_amplitude: Option<f64>,
    pub precision: Option<i32>,
}

impl GaussianBlurOperation {
    pub(crate) fn radius(&self) -> i32 {
        (self.sigma * 3.0).ceil().max(0.0) as i32
    }
}

impl VipsOperation for GaussianBlurOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"gaussblur\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_double("sigma", self.sigma);
        if let Some(v) = self.minimum_amplitude {
            op.set_double("min_ampl", v);
        }
        if let Some(v) = self.precision {
            op.set_int("precision", v);
        }
    }
}

pub struct SharpenOperation {
    pub sigma: Option<f64>,
    pub flat: Option<f64>,
    pub jagged: Option<f64>,
    pub edge: Option<f64>,
    pub smooth: Option<f64>,
    pub maximum: Option<f64>,
}

impl VipsOperation for SharpenOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"sharpen\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.sigma {
            op.set_double("sigma", v);
        }
        if let Some(v) = self.flat {
            op.set_double("x1", v);
        }
        if let Some(v) = self.jagged {
            op.set_double("y2", v);
        }
        if let Some(v) = self.edge {
            op.set_double("y3", v);
        }
        if let Some(v) = self.smooth {
            op.set_double("m1", v);
        }
        if let Some(v) = self.maximum {
            op.set_double("m2", v);
        }
    }
}

pub struct CannyOperation {
    pub sigma: Option<f64>,
    pub precision: Option<i32>,
}

impl VipsOperation for CannyOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"canny\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.sigma {
            op.set_double("sigma", v);
        }
        if let Some(v) = self.precision {
            op.set_int("precision", v);
        }
    }
}

pub struct MedianOperation {
    pub size: i32,
}

impl VipsOperation for MedianOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"rank\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("width", self.size);
        op.set_int("height", self.size);
        op.set_int("index", self.size * self.size / 2);
    }
}

pub struct HoughLineOperation {
    pub width: Option<i32>,
    pub height: Option<i32>,
}
impl VipsOperation for HoughLineOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hough_line\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.width {
            op.set_int("width", v);
        }
        if let Some(v) = self.height {
            op.set_int("height", v);
        }
    }
}

pub struct HoughCircleOperation {
    pub scale: Option<i32>,
    pub min_radius: Option<i32>,
    pub max_radius: Option<i32>,
}
impl VipsOperation for HoughCircleOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hough_circle\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.scale {
            op.set_int("scale", v);
        }
        if let Some(v) = self.min_radius {
            op.set_int("min_radius", v);
        }
        if let Some(v) = self.max_radius {
            op.set_int("max_radius", v);
        }
    }
}

// ── GaussianBlurOperation ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct BlurHPassOp {
    radius: i32,
}

impl TypedOperation for BlurHPassOp {
    type Output = ImageType;
}

impl GpuOperation for BlurHPassOp {
    fn emit(&self, _: &[NodeId], _: &mut Graph, _: Arc<dyn GpuOperation>) -> NodeId {
        panic!("BlurHPassOp is not emitted directly")
    }
    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        match wu {
            crate::backend::gpu::work_unit::WorkUnit::Region { rect, lod } => {
                let scaled_radius = (self.radius as f64 / lod.scale_factor()).ceil() as i32;
                vec![(
                    0,
                    crate::backend::gpu::work_unit::WorkUnit::Region {
                        rect: Rect::new(
                            rect.x - scaled_radius,
                            rect.y,
                            rect.width + 2 * scaled_radius,
                            rect.height,
                        ),
                        lod: *lod,
                    },
                )]
            }
            _ => vec![(0, wu.clone())],
        }
    }
    fn scale_params_for_lod(
        &self,
        params: &[crate::backend::gpu::param::Param],
        lod: crate::backend::gpu::Lod,
    ) -> Vec<crate::backend::gpu::param::Param> {
        let scale = lod.scale_factor() as f32;
        let mut out = params.to_vec();
        if let Some(crate::backend::gpu::param::Param::F32(v)) = out.first_mut() {
            *v /= scale;
        }
        out
    }
}

impl TypedOperation for GaussianBlurOperation {
    type Output = ImageType;
}

impl GpuOperation for GaussianBlurOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let h_pass: Arc<dyn GpuOperation> = Arc::new(BlurHPassOp {
            radius: self.radius(),
        });
        let node_h = emit_image(
            graph,
            input,
            h_pass,
            "ops.gaussian_blur",
            "blur_h_kernel",
            vec![Param::F32(self.sigma as f32)],
        );
        emit_image(
            graph,
            node_h,
            self_arc,
            "ops.gaussian_blur",
            "blur_v_kernel",
            vec![Param::F32(self.sigma as f32)],
        )
    }

    fn input_demands(
        &self,
        wu: &crate::backend::gpu::work_unit::WorkUnit,
    ) -> Vec<(usize, crate::backend::gpu::work_unit::WorkUnit)> {
        match wu {
            crate::backend::gpu::work_unit::WorkUnit::Region { rect, lod } => {
                let scaled_radius = (self.radius() as f64 / lod.scale_factor()).ceil() as i32;
                vec![(
                    0,
                    crate::backend::gpu::work_unit::WorkUnit::Region {
                        rect: Rect::new(
                            rect.x,
                            rect.y - scaled_radius,
                            rect.width,
                            rect.height + 2 * scaled_radius,
                        ),
                        lod: *lod,
                    },
                )]
            }
            _ => vec![(0, wu.clone())],
        }
    }
    fn scale_params_for_lod(
        &self,
        params: &[crate::backend::gpu::param::Param],
        lod: crate::backend::gpu::Lod,
    ) -> Vec<crate::backend::gpu::param::Param> {
        let scale = lod.scale_factor() as f32;
        let mut out = params.to_vec();
        if let Some(crate::backend::gpu::param::Param::F32(v)) = out.first_mut() {
            *v /= scale;
        }
        out
    }
}
