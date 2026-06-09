use crate::backend::gpu::graph::{Graph, NodeId};
use crate::backend::gpu::op::GpuOperation;
use crate::backend::gpu::op::emit_image;
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
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
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
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
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
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
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
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
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
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
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
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
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

impl GpuOperation for BlurHPassOp {
    fn emit(&self, _: NodeId, _: &mut Graph, _: Arc<dyn GpuOperation>) -> NodeId {
        panic!("BlurHPassOp is not emitted directly")
    }
    fn inverse_map(
        &self,
        output_rect: Rect,
        w: u32,
        h: u32,
        lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, Rect)> {
        // Scale the halo by 1/lod_scale so we don't over-fetch source pixels.
        let scaled_radius = (self.radius as f64 / lod.scale_factor()).ceil() as i32;
        let bounds = Rect::new(0, 0, w as i32, h as i32);
        vec![(
            0,
            Rect::new(
                output_rect.x - scaled_radius,
                output_rect.y,
                output_rect.width + 2 * scaled_radius,
                output_rect.height,
            )
            .clamp(bounds),
        )]
    }
    /// sigma is param index 0 — must be divided by lod.scale_factor() at dispatch.
    fn lod_scale_param_indices(&self) -> &'static [usize] {
        &[0]
    }
}

impl GpuOperation for GaussianBlurOperation {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
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

    fn inverse_map(
        &self,
        output_rect: Rect,
        w: u32,
        h: u32,
        lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, Rect)> {
        let scaled_radius = (self.radius() as f64 / lod.scale_factor()).ceil() as i32;
        let bounds = Rect::new(0, 0, w as i32, h as i32);
        vec![(
            0,
            Rect::new(
                output_rect.x,
                output_rect.y - scaled_radius,
                output_rect.width,
                output_rect.height + 2 * scaled_radius,
            )
            .clamp(bounds),
        )]
    }
    /// sigma is param index 0 (V-pass node).
    fn lod_scale_param_indices(&self) -> &'static [usize] {
        &[0]
    }
}
