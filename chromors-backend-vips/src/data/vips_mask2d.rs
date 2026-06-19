use crate::{VipsBackend, VipsBuilder, VipsHandle};
use chromors_core::data::mask2d::{Mask2D, Mask2DKind, RamMaskTarget};
use chromors_core::*;
use std::hash::Hasher;
use std::sync::Arc;

pub trait VipsMask2DExt {
    fn extract_target(&self) -> RawMaskTarget;
    fn from_values(width: i32, height: i32, values: &[f32]) -> Mask2D<VipsBackend>;
    fn from_values_scaled(
        width: i32,
        height: i32,
        values: &[f32],
        scale: f64,
        offset: f64,
    ) -> Mask2D<VipsBackend>;
    fn identity(n: i32) -> Mask2D<VipsBackend>;
}

impl VipsMask2DExt for Mask2D<VipsBackend> {
    fn from_values(width: i32, height: i32, values: &[f32]) -> Self {
        Self::from_values_scaled(width, height, values, 1.0, 0.0)
    }

    fn from_values_scaled(
        width: i32,
        height: i32,
        values: &[f32],
        scale: f64,
        offset: f64,
    ) -> Self {
        let spec = Arc::new(Mask2DKind::new(width, height));
        let data: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        let src = VipsConstantMaskSource {
            spec: spec.clone(),
            data,
            scale,
            offset,
        };
        chromors_core::node::Data::from_source(Arc::new(src), Arc::new(()))
    }

    fn identity(n: i32) -> Self {
        let dim = n.max(0) as usize;
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self::from_values(n, n, &data)
    }

    fn extract_target(&self) -> RawMaskTarget {
        RawMaskTarget
    }
}

pub struct VipsConstantMaskSource {
    pub spec: Arc<Mask2DKind>,
    pub data: Vec<f64>,
    pub scale: f64,
    pub offset: f64,
}

impl Source<VipsBackend> for VipsConstantMaskSource {
    type Kind = Mask2DKind;

    fn spec(&self) -> Arc<Mask2DKind> {
        self.spec.clone()
    }

    fn fetch(
        &self,
        _ctx: &(),
        _wu: &chromors_core::work_unit::Region,
    ) -> Result<Buffer<VipsBackend>, Error> {
        let ptr = unsafe {
            crate::ffi::vips_image_new_from_memory_copy(
                self.data.as_ptr() as *const std::ffi::c_void,
                self.data.len() * 8,
                self.spec.width,
                self.spec.height,
                1,
                crate::ffi::VipsBandFormat_VIPS_FORMAT_DOUBLE,
            )
        };
        if ptr.is_null() {
            return Err(Error::Vips(crate::vips_error()));
        }
        // vips_conv/conva/compass/morph read the mask's "scale"/"offset"
        // double properties (defaulting to 0 if unset, which divides the
        // convolution result by zero).
        unsafe {
            let scale = std::ffi::CString::new("scale").unwrap();
            let offset = std::ffi::CString::new("offset").unwrap();
            let xoffset = std::ffi::CString::new("xoffset").unwrap();
            let yoffset = std::ffi::CString::new("yoffset").unwrap();
            crate::ffi::vips_image_set_double(ptr, scale.as_ptr(), self.scale);
            crate::ffi::vips_image_set_double(ptr, offset.as_ptr(), self.offset);
            crate::ffi::vips_image_set_int(ptr, xoffset.as_ptr(), self.spec.width / 2);
            crate::ffi::vips_image_set_int(ptr, yoffset.as_ptr(), self.spec.height / 2);
        }
        Ok(Buffer {
            payload: Arc::new(VipsHandle { ptr }),
            spec: self.spec.clone(),
        })
    }

    fn lower(&self, cx: &mut VipsBuilder) {
        let region = chromors_core::work_unit::Region::full(
            (self.spec.width, self.spec.height),
            chromors_core::work_unit::Lod(0),
        );
        let buf = self.fetch(&(), &region).unwrap();
        cx.emit((*buf.payload).clone());
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        for &v in &self.data {
            state.write_u64(v.to_bits());
        }
    }
}

pub struct RawMaskTarget;

impl Target<Mask2DKind, VipsBackend> for RawMaskTarget {
    type Out = VipsHandle;

    fn extract(
        &self,
        buf: &Buffer<VipsBackend>,
        _wu: &chromors_core::work_unit::Region,
        _ctx: &(),
    ) -> Result<Self::Out, Error> {
        Ok((*buf.payload).clone())
    }
}

impl Target<Mask2DKind, VipsBackend> for RamMaskTarget {
    type Out = Vec<f32>;

    fn extract(
        &self,
        buf: &Buffer<VipsBackend>,
        _wu: &chromors_core::work_unit::Region,
        _ctx: &<VipsBackend as Backend>::Ctx,
    ) -> Result<Self::Out, Error> {
        let mut size: usize = 0;
        let ptr = unsafe {
            crate::ffi::vips_image_write_to_memory(buf.payload.ptr(), &mut size as *mut usize)
        };
        if ptr.is_null() {
            return Err(Error::Vips(crate::vips_error()));
        }
        // Vips mask images are VIPS_FORMAT_DOUBLE (f64)
        let count = size / 8;
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const f64, count) };
        let values: Vec<f32> = slice.iter().map(|&v| v as f32).collect();
        unsafe { crate::ffi::g_free(ptr as *mut std::ffi::c_void) };
        Ok(values)
    }
}
