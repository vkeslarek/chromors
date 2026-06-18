use crate::prelude::*;

impl Lower<VipsBackend> for crate::ForwardFft<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"fwfft\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::InverseFft<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"invfft\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Spectrum<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"spectrum\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

