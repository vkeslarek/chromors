use crate::prelude::*;

impl Lower<VipsBackend> for crate::Composite2<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let base_handle = cx.input(self.base.src());
        let mut op = VipsGObject::new(b"composite2\0")
            .expect("failed to create vips composite2 op");
        op.set_image("base", base_handle.ptr);
        let overlay_handle = cx.input(self.overlay.src());
        op.set_image("overlay", overlay_handle.ptr);
        op.set_int("mode", self.mode.into_vips());
        if let Some(v) = self.x {
            op.set_int("x", v);
        }
        if let Some(v) = self.y {
            op.set_int("y", v);
        }
        if let Some(v) = self.premultiplied {
            op.set_bool("premultiplied", v);
        }
        let out_handle = op.run().expect("vips composite2 failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Join<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let in1_handle = cx.input(self.in1.src());
        let mut op = VipsGObject::new(b"join\0")
            .expect("failed to create vips join op");
        op.set_image("in1", in1_handle.ptr);
        let in2_handle = cx.input(self.in2.src());
        op.set_image("in2", in2_handle.ptr);
        op.set_int("direction", self.direction.into_vips());
        if let Some(v) = self.expand {
            op.set_bool("expand", v);
        }
        if let Some(v) = self.shim {
            op.set_int("shim", v);
        }
        if let Some(v) = self.align {
            op.set_int("align", v.into_vips());
        }
        let out_handle = op.run().expect("vips join failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Insert<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let main_handle = cx.input(self.main.src());
        let mut op = VipsGObject::new(b"insert\0")
            .expect("failed to create vips insert op");
        op.set_image("main", main_handle.ptr);
        let sub_handle = cx.input(self.sub.src());
        op.set_image("sub", sub_handle.ptr);
        op.set_int("x", self.x);
        op.set_int("y", self.y);
        if let Some(v) = self.expand {
            op.set_bool("expand", v);
        }
        let out_handle = op.run().expect("vips insert failed");
        cx.emit(out_handle);
    }
}

