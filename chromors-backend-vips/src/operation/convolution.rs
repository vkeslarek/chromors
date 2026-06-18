use crate::prelude::*;

impl Lower<VipsBackend> for crate::Convolution<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = VipsGObject::new(b"conv\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        if let Some(v) = self.precision {
            op.set_int("precision", v.into_vips());
        }
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        if let Some(v) = self.cluster {
            op.set_int("cluster", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Compass<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = VipsGObject::new(b"compass\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Morph<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = VipsGObject::new(b"morph\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        op.set_int("morph", self.morph.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Conva<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = VipsGObject::new(b"conva\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        if let Some(v) = self.cluster {
            op.set_int("cluster", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Convf<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = VipsGObject::new(b"convf\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Convi<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = VipsGObject::new(b"convi\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Convsep<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = VipsGObject::new(b"convsep\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        if let Some(v) = self.precision {
            op.set_int("precision", v.into_vips());
        }
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        if let Some(v) = self.cluster {
            op.set_int("cluster", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Convasep<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = VipsGObject::new(b"convasep\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

