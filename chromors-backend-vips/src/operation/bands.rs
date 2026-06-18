use crate::prelude::*;

impl Lower<VipsBackend> for crate::Bandbool<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"bandbool\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("boolean", self.boolean.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Bandfold<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"bandfold\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("factor", self.factor as i32);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Bandunfold<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"bandunfold\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("factor", self.factor as i32);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Bandmean<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"bandmean\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::ExtractBand<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"extract_band\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("band", self.band);
        if let Some(c) = self.count {
            op.set_int("n", c);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Bandjoin<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let mut ptrs: Vec<*mut crate::ffi::VipsImage> = vec![];
        for input in &self.images {
            let handle = cx.input(input.src());
            ptrs.push(handle.ptr);
        }
        let mut out: *mut crate::ffi::VipsImage = std::ptr::null_mut();
        let ret = unsafe {
            crate::ffi::vips_bandjoin(
                ptrs.as_mut_ptr(),
                &mut out,
                ptrs.len() as i32,
                std::ptr::null::<std::ffi::c_void>(),
            )
        };
        if ret != 0 {
            panic!("vips_bandjoin failed");
        }
        let out_handle = VipsHandle { ptr: out };
        cx.emit(out_handle);
    }
}

