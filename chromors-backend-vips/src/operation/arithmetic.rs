use crate::prelude::*;

impl Lower<VipsBackend> for crate::Add<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"add\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Subtract<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"subtract\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Multiply<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"multiply\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Divide<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"divide\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::MaxPair<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"maxpair\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::MinPair<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"minpair\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Remainder<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"remainder\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Complexform<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"complexform\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Complex2<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"complex2\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        op.set_int("cmplx", self.cmplx.into_vips());
        let out_handle = op.run().expect("vips complex2 failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Math<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"math\0").expect("failed to create vips math op");
        op.set_image("in", input_handle.ptr);
        op.set_int("math", self.math.into_vips());
        let out_handle = op.run().expect("vips math failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Round<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"round\0").expect("failed to create vips round op");
        op.set_image("in", input_handle.ptr);
        op.set_int("round", self.round.into_vips());
        let out_handle = op.run().expect("vips round failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Math2<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"math2\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        op.set_int("math2", self.math2.into_vips());
        let out_handle = op.run().expect("vips math2 failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Linear<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"linear\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_array_double("a", &self.a);
        op.set_array_double("b", &self.b);
        if self.input.spec.layout.storage == crate::Storage::U8 {
            op.set_bool("uchar", true);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Math2Const<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"math2_const\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("math2", self.math2.into_vips());
        op.set_array_double("c", &self.c);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::RemainderConst<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"remainder_const\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_array_double("c", &self.c);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}
