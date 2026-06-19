use crate::prelude::*;

impl Lower<VipsBackend> for crate::Gamma<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"gamma\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.exponent {
            op.set_double("exponent", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}
