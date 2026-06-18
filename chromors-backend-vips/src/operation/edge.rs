use crate::prelude::*;

impl Lower<VipsBackend> for crate::Sobel<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"sobel\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();

        let mut cast_op = VipsGObject::new(b"cast\0").unwrap();
        cast_op.set_image("in", out_handle.ptr);
        cast_op.set_int("format", 6); // VIPS_FORMAT_FLOAT
        let cast_handle = cast_op.run().unwrap();

        cx.emit(cast_handle);
    }
}

impl Lower<VipsBackend> for crate::Prewitt<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"prewitt\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();

        let mut cast_op = VipsGObject::new(b"cast\0").unwrap();
        cast_op.set_image("in", out_handle.ptr);
        cast_op.set_int("format", 6); // VIPS_FORMAT_FLOAT
        let cast_handle = cast_op.run().unwrap();

        cx.emit(cast_handle);
    }
}

impl Lower<VipsBackend> for crate::Scharr<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"scharr\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();

        let mut cast_op = VipsGObject::new(b"cast\0").unwrap();
        cast_op.set_image("in", out_handle.ptr);
        cast_op.set_int("format", 6); // VIPS_FORMAT_FLOAT
        let cast_handle = cast_op.run().unwrap();

        cx.emit(cast_handle);
    }
}

impl Lower<VipsBackend> for crate::Invert<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"invert\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Sign<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"sign\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Abs<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"abs\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

