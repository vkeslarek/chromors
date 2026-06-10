use crate::backend::gpu::datatype::ImageType;
use crate::backend::gpu::graph::{Graph, NodeId};
use crate::backend::gpu::op::emit_image;
use crate::backend::gpu::op::{GpuOperation, TypedOperation};
use crate::backend::gpu::param::Param;
use std::sync::Arc;

use crate::backend::Operation;
use crate::error::Error;
use crate::libvips_ffi as ffi;

#[derive(Debug, Clone)]
pub struct OpacityOperation {
    pub amount: f32,
}

impl Operation<crate::data::image::Image2D<crate::backend::vips::VipsBackend>>
    for OpacityOperation
{
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;

    fn execute(
        &self,
        img: &crate::data::image::Image2D<crate::backend::vips::VipsBackend>,
    ) -> Result<Self::Output, Error> {
        if !img.has_alpha() {
            let max_val =
                match img.pixel_format().bytes_per_pixel() / img.pixel_format().channel_count() {
                    1 => 255.0,
                    2 => 65535.0,
                    _ => 255.0, // VIPS float images traditionally use 255 for opaque alpha
                };
            let with_alpha = img.bandjoin_const(&[max_val])?;
            return self.execute(&with_alpha);
        }

        let bands = img.bands();
        if bands < 2 {
            return img.execute(&super::arithmetic::LinearOperation {
                a: self.amount as f64,
                b: 0.0,
                uchar: Some(true),
            });
        }

        let rgb = img.execute(&super::bands::ExtractBandOperation {
            band: 0,
            count: Some(bands - 1),
        })?;
        let alpha = img.execute(&super::bands::ExtractBandOperation {
            band: bands - 1,
            count: Some(1),
        })?;
        // uchar=true forces uint8 output regardless of input format.
        // For uint16 images (RAW), this converts alpha 65535→255, then bandjoin
        // upcasts it back to uint16 value 255, giving 0.4% opacity (transparent).
        // Only use uchar for uint8 inputs; let vips_linear preserve uint16.
        let bytes_per_sample =
            img.pixel_format().bytes_per_pixel() / img.pixel_format().channel_count();
        let uchar = (bytes_per_sample == 1).then_some(true);
        let scaled_float = alpha.execute(&super::arithmetic::LinearOperation {
            a: self.amount as f64,
            b: 0.0,
            uchar,
        })?;
        let scaled = scaled_float.execute(&super::misc::CastOperation {
            format: img.pixel_format(),
            shift: None,
        })?;

        let images = [rgb.vips_ptr(), scaled.vips_ptr()];
        let mut out: *mut ffi::VipsImage = std::ptr::null_mut();
        let ret = unsafe {
            ffi::vips_bandjoin(
                images.as_ptr() as *mut *mut ffi::VipsImage,
                &mut out,
                2,
                crate::backend::vips::null(),
            )
        };
        if ret != 0 {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(crate::data::image::Image2D::from_vips_ptr(out))
    }
}

// ── OpacityOperation ──────────────────────────────────────────────────────────

impl TypedOperation for OpacityOperation {
    type Output = ImageType;
}

impl GpuOperation for OpacityOperation {
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
            "ops.opacity",
            "opacity_kernel",
            vec![Param::F32(self.amount)],
        )
    }
}
