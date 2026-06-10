use crate::backend::gpu::datatype::ImageType;
use crate::backend::gpu::graph::{Graph, NodeId};
use crate::backend::gpu::op::emit_image;
use crate::backend::gpu::op::{GpuOperation, TypedOperation};
use crate::backend::gpu::param::Param;
use std::sync::Arc;

use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

pub struct IccImportOperation {
    pub embedded: Option<bool>,
    pub input_profile: Option<String>,
    pub intent: Option<i32>,
}
impl VipsOperation for IccImportOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"icc_import\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.embedded {
            op.set_bool("embedded", v);
        }
        if let Some(ref v) = self.input_profile {
            op.set_string("input_profile", v);
        }
        if let Some(v) = self.intent {
            op.set_int("intent", v);
        }
    }
}

pub struct IccExportOperation {
    pub output_profile: Option<String>,
    pub intent: Option<i32>,
    pub depth: Option<i32>,
}
impl VipsOperation for IccExportOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"icc_export\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(ref v) = self.output_profile {
            op.set_string("output_profile", v);
        }
        if let Some(v) = self.intent {
            op.set_int("intent", v);
        }
        if let Some(v) = self.depth {
            op.set_int("depth", v);
        }
    }
}

pub struct IccTransformOperation {
    pub output_profile: String,
    pub embedded: Option<bool>,
    pub input_profile: Option<String>,
    pub intent: Option<i32>,
    pub depth: Option<i32>,
}
impl VipsOperation for IccTransformOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"icc_transform\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_string("output_profile", &self.output_profile);
        if let Some(v) = self.embedded {
            op.set_bool("embedded", v);
        }
        if let Some(ref v) = self.input_profile {
            op.set_string("input_profile", v);
        }
        if let Some(v) = self.intent {
            op.set_int("intent", v);
        }
        if let Some(v) = self.depth {
            op.set_int("depth", v);
        }
    }
}

#[derive(Debug, Clone)]
pub struct GammaOperation {
    pub exponent: Option<f64>,
}
impl VipsOperation for GammaOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"gamma\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.exponent {
            op.set_double("exponent", v);
        }
    }
}

// ColourspaceOperation removed — use Image2D::convert(PixelMeta) instead.

/// Adjusts saturation by blending the image with its grayscale version.
/// `amount = 0` produces grayscale, `amount = 1` is identity, `amount > 1` boosts.
///
/// Computes luminance-weighted grayscale (Rec. 709) and blends:
/// `output = gray + amount * (original - gray)`.
#[derive(Debug, Clone)]
pub struct SaturationOperation {
    pub amount: f64,
}

impl crate::backend::Operation<crate::data::image::Image2D<crate::backend::vips::VipsBackend>>
    for SaturationOperation
{
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn execute(
        &self,
        image: &crate::data::image::Image2D<crate::backend::vips::VipsBackend>,
    ) -> Result<Self::Output, crate::error::Error> {
        let amount = self.amount;
        let bands = image.bands();
        let rgb_bands = if bands >= 3 { 3 } else { bands };

        let img_f = image.execute(&crate::operation::misc::CastOperation {
            format: crate::PixelFormat::RgbaF32,
            shift: None,
        })?;

        let rgb = img_f.execute(&crate::operation::bands::ExtractBandOperation {
            band: 0,
            count: Some(rgb_bands),
        })?;

        let r = rgb.execute(&crate::operation::bands::ExtractBandOperation {
            band: 0,
            count: Some(1),
        })?;
        let g = rgb.execute(&crate::operation::bands::ExtractBandOperation {
            band: 1.min(rgb_bands - 1),
            count: Some(1),
        })?;
        let b = rgb.execute(&crate::operation::bands::ExtractBandOperation {
            band: 2.min(rgb_bands - 1),
            count: Some(1),
        })?;

        let r_scaled = r.execute(&crate::operation::arithmetic::LinearOperation {
            a: 0.2126,
            b: 0.0,
            uchar: None,
        })?;
        let g_scaled = g.execute(&crate::operation::arithmetic::LinearOperation {
            a: 0.7152,
            b: 0.0,
            uchar: None,
        })?;
        let b_scaled = b.execute(&crate::operation::arithmetic::LinearOperation {
            a: 0.0722,
            b: 0.0,
            uchar: None,
        })?;

        let rg =
            r_scaled.execute(&crate::operation::arithmetic::AddOperation { right: g_scaled })?;
        let gray = rg.execute(&crate::operation::arithmetic::AddOperation { right: b_scaled })?;

        let diff = rgb.execute(&crate::operation::arithmetic::SubtractOperation {
            right: gray.clone(),
        })?;
        let diff_scaled = diff.execute(&crate::operation::arithmetic::LinearOperation {
            a: amount,
            b: 0.0,
            uchar: None,
        })?;
        let out_rgb =
            gray.execute(&crate::operation::arithmetic::AddOperation { right: diff_scaled })?;

        let final_f = if bands > 3 {
            let alpha = img_f.execute(&crate::operation::bands::ExtractBandOperation {
                band: 3,
                count: Some(bands - 3),
            })?;
            let images = [out_rgb.vips_ptr(), alpha.vips_ptr()];
            let mut out: *mut crate::libvips_ffi::VipsImage = std::ptr::null_mut();
            let ret = unsafe {
                crate::libvips_ffi::vips_bandjoin(
                    images.as_ptr() as *mut *mut crate::libvips_ffi::VipsImage,
                    &mut out,
                    2,
                    crate::backend::vips::null(),
                )
            };
            if ret != 0 {
                return Err(crate::error::Error::Vips(crate::backend::vips::vips_error()));
            }
            crate::data::image::Image2D::from_handle(crate::backend::vips::VipsHandle { ptr: out })
        } else {
            out_rgb
        };

        final_f.execute(&crate::operation::misc::CastOperation {
            format: image.pixel_format(),
            shift: None,
        })
    }
}

// ── SaturationOperation ───────────────────────────────────────────────────────

impl TypedOperation for SaturationOperation {
    type Output = ImageType;
}

impl GpuOperation for SaturationOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        emit_image(
            graph,
            input,
            self_arc,
            "ops.saturation",
            "saturation_kernel",
            vec![Param::F32(self.amount as f32)],
        )
    }
}

// ── GammaOperation ────────────────────────────────────────────────────────────

impl TypedOperation for GammaOperation {
    type Output = ImageType;
}

impl GpuOperation for GammaOperation {
    fn emit(
        &self,
        inputs: &[NodeId],
        graph: &mut Graph,
        self_arc: Arc<dyn GpuOperation>,
    ) -> NodeId {
        let input = inputs[0];
        let exponent = self.exponent.unwrap_or(1.0);
        let shader_exp = if exponent != 0.0 {
            (1.0 / exponent) as f32
        } else {
            1.0
        };
        emit_image(
            graph,
            input,
            self_arc,
            "ops.gamma",
            "gamma_kernel",
            vec![Param::F32(shader_exp)],
        )
    }
}
