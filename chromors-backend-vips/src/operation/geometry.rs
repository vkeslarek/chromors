use crate::prelude::*;

impl Lower<VipsBackend> for crate::Crop<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"crop\0").unwrap();
        op.set_image("input", input_handle.ptr);
        op.set_int("left", self.left);
        op.set_int("top", self.top);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Embed<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"embed\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("x", self.x);
        op.set_int("y", self.y);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(e) = self.extend {
            op.set_int("extend", e.into_vips());
        }
        if let Some(bg) = self.background {
            op.set_array_double("background", &bg);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Flip<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"flip\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("direction", self.direction.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Rot90<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"rot\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("angle", self.angle.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Rot45<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"rot45\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("angle", self.angle.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Rotate<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"rotate\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("angle", self.angle);
        if let Some(bg) = self.background {
            op.set_array_double("background", &bg);
        }
        if let Some(v) = self.offset_input_x {
            op.set_double("idx", v);
        }
        if let Some(v) = self.offset_input_y {
            op.set_double("idy", v);
        }
        if let Some(v) = self.offset_output_x {
            op.set_double("odx", v);
        }
        if let Some(v) = self.offset_output_y {
            op.set_double("ody", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Smartcrop<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"smartcrop\0").unwrap();
        op.set_image("input", input_handle.ptr);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(i) = self.interesting {
            op.set_int("interesting", i.into_vips());
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Gravity<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"gravity\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("direction", self.direction.into_vips());
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(e) = self.extend {
            op.set_int("extend", e.into_vips());
        }
        if let Some(bg) = self.background {
            op.set_array_double("background", &bg);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Resize<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"resize\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("scale", self.scale);
        if let Some(k) = self.kernel {
            op.set_int("kernel", k.into_vips());
        }
        if let Some(v) = self.vertical_scale {
            op.set_double("vscale", v);
        }
        if let Some(g) = self.gap {
            op.set_double("gap", g);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Thumbnail<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"thumbnail_image\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("width", self.width);
        if let Some(h) = self.height {
            op.set_int("height", h);
        }
        if let Some(s) = self.size {
            op.set_int("size", s);
        }
        if let Some(c) = self.crop {
            op.set_int("crop", c.into_vips());
        }
        if let Some(v) = self.linear {
            op.set_bool("linear", v);
        }
        if let Some(v) = self.auto_rotate {
            op.set_bool("auto_rotate", v);
        }
        if let Some(v) = self.no_rotate {
            op.set_bool("no_rotate", v);
        }
        if let Some(ref v) = self.import_profile {
            op.set_string("import_profile", v);
        }
        if let Some(ref v) = self.export_profile {
            op.set_string("export_profile", v);
        }
        if let Some(v) = self.intent {
            op.set_int("intent", v);
        }
        if let Some(v) = self.fail_on {
            op.set_int("fail_on", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Shrink<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"shrink\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("hshrink", self.horizontal);
        op.set_double("vshrink", self.vertical);
        if let Some(c) = self.ceil {
            op.set_bool("ceil", c);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Reduce<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"reduce\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("hshrink", self.horizontal);
        op.set_double("vshrink", self.vertical);
        if let Some(k) = self.kernel {
            op.set_int("kernel", k.into_vips());
        }
        if let Some(g) = self.gap {
            op.set_double("gap", g);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::ReduceHorizontal<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"reduceh\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("hshrink", self.shrink);
        if let Some(k) = self.kernel {
            op.set_int("kernel", k.into_vips());
        }
        if let Some(g) = self.gap {
            op.set_double("gap", g);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::ReduceVertical<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"reducev\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_double("vshrink", self.shrink);
        if let Some(k) = self.kernel {
            op.set_int("kernel", k.into_vips());
        }
        if let Some(g) = self.gap {
            op.set_double("gap", g);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::ExtractArea<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"extract_area\0").unwrap();
        op.set_image("input", input_handle.ptr);
        op.set_int("left", self.left);
        op.set_int("top", self.top);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Subsample<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"subsample\0").unwrap();
        op.set_image("input", input_handle.ptr);
        op.set_int("xfac", self.horizontal);
        op.set_int("yfac", self.vertical);
        if let Some(p) = self.point {
            op.set_bool("point", p);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Zoom<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"zoom\0").unwrap();
        op.set_image("input", input_handle.ptr);
        op.set_int("xfac", self.horizontal);
        op.set_int("yfac", self.vertical);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Replicate<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"replicate\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("across", self.across);
        op.set_int("down", self.down);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::ShrinkHorizontal<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"shrinkh\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("hshrink", self.shrink);
        if let Some(c) = self.ceil {
            op.set_bool("ceil", c);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::ShrinkVertical<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"shrinkv\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("vshrink", self.shrink);
        if let Some(c) = self.ceil {
            op.set_bool("ceil", c);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Grid<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"grid\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("tile_height", self.tile_height);
        op.set_int("across", self.across);
        op.set_int("down", self.down);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}
