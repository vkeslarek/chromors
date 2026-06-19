use crate::{VipsBackend, VipsBand, VipsBuilder, VipsHandle};
use chromors_core::lut::RawLutTarget;
use chromors_core::*;
use std::hash::Hasher;
use std::sync::Arc;

impl VipsBand for LutKind {
    fn band_format(&self) -> i32 {
        crate::ffi::VipsBandFormat_VIPS_FORMAT_DOUBLE
    }
}

pub struct VipsConstantLutSource {
    pub spec: Arc<LutKind>,
    pub data: Vec<f64>,
}

impl Source<VipsBackend> for VipsConstantLutSource {
    type Kind = LutKind;

    fn spec(&self) -> Arc<LutKind> {
        self.spec.clone()
    }

    fn fetch(&self, _ctx: &(), _wu: &Range) -> Result<Buffer<VipsBackend>, Error> {
        let ptr = unsafe {
            crate::ffi::vips_image_new_from_memory_copy(
                self.data.as_ptr() as *const std::ffi::c_void,
                self.data.len() * 8,
                self.spec.bands as i32,
                self.spec.entries as i32,
                1,
                crate::ffi::VipsBandFormat_VIPS_FORMAT_DOUBLE,
            )
        };
        if ptr.is_null() {
            return Err(Error::Vips(crate::vips_error()));
        }
        Ok(Buffer {
            payload: Arc::new(VipsHandle { ptr }),
            spec: self.spec.clone(),
        })
    }

    fn lower(&self, cx: &mut VipsBuilder) {
        let buf = self
            .fetch(
                &(),
                &Range {
                    start: 0,
                    end: self.spec.entries as i32,
                },
            )
            .unwrap();
        cx.emit((*buf.payload).clone());
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        for &v in &self.data {
            state.write_u64(v.to_bits());
        }
    }
}

pub trait VipsLutExt {
    fn from_values(entries: u32, bands: u32, values: &[f32]) -> Self;
}

impl VipsLutExt for Lut<VipsBackend> {
    fn from_values(entries: u32, bands: u32, values: &[f32]) -> Self {
        let spec = Arc::new(LutKind::new(entries, bands));
        let data: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        let src = VipsConstantLutSource {
            spec: spec.clone(),
            data,
        };
        Data::from_source(Arc::new(src), Arc::new(()))
    }
}

impl Target<LutKind, VipsBackend> for RawLutTarget {
    type Out = Vec<u8>;

    fn extract(
        &self,
        buf: &Buffer<VipsBackend>,
        _wu: &Range,
        _ctx: &(),
    ) -> Result<Self::Out, Error> {
        let mut size: usize = 0;
        let ptr = unsafe {
            crate::ffi::vips_image_write_to_memory(buf.payload.ptr, &mut size as *mut usize)
        };
        if ptr.is_null() {
            return Err(Error::Vips(crate::vips_error()));
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        let vec = slice.to_vec();
        unsafe { crate::ffi::g_free(ptr as *mut std::ffi::c_void) };
        Ok(vec)
    }
}
