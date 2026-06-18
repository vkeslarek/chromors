use crate::prelude::*;

impl Lower<VipsBackend> for crate::Mosaic<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let ref_handle = cx.input(self.input.src());
        let sec_handle = cx.input(self.secondary.src());
        let mut op = VipsGObject::new(b"mosaic\0").unwrap();
        op.set_image("ref", ref_handle.ptr);
        op.set_image("sec", sec_handle.ptr);
        op.set_int("direction", self.direction.into_vips());
        op.set_int("xref", self.x_reference);
        op.set_int("yref", self.y_reference);
        op.set_int("xsec", self.x_secondary);
        op.set_int("ysec", self.y_secondary);
        if let Some(v) = self.half_window {
            op.set_int("hwindow", v);
        }
        if let Some(v) = self.half_area {
            op.set_int("harea", v);
        }
        if let Some(v) = self.max_blend {
            op.set_int("mblend", v);
        }
        if let Some(v) = self.search_band {
            op.set_int("bandno", v);
        }
        let out = op.run().unwrap();
        cx.emit(out);
    }
}

impl Lower<VipsBackend> for crate::Mosaic1<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let ref_handle = cx.input(self.input.src());
        let sec_handle = cx.input(self.secondary.src());
        let mut op = VipsGObject::new(b"mosaic1\0").unwrap();
        op.set_image("ref", ref_handle.ptr);
        op.set_image("sec", sec_handle.ptr);
        op.set_int("direction", self.direction.into_vips());
        op.set_int("xr1", self.x_reference_1);
        op.set_int("yr1", self.y_reference_1);
        op.set_int("xs1", self.x_secondary_1);
        op.set_int("ys1", self.y_secondary_1);
        op.set_int("xr2", self.x_reference_2);
        op.set_int("yr2", self.y_reference_2);
        op.set_int("xs2", self.x_secondary_2);
        op.set_int("ys2", self.y_secondary_2);
        if let Some(v) = self.half_window {
            op.set_int("hwindow", v);
        }
        if let Some(v) = self.half_area {
            op.set_int("harea", v);
        }
        if let Some(v) = self.search {
            op.set_bool("search", v);
        }
        if let Some(v) = self.max_blend {
            op.set_int("mblend", v);
        }
        let out = op.run().unwrap();
        cx.emit(out);
    }
}

impl Lower<VipsBackend> for crate::Match<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let ref_handle = cx.input(self.input.src());
        let sec_handle = cx.input(self.secondary.src());
        let mut op = VipsGObject::new(b"match\0").unwrap();
        op.set_image("ref", ref_handle.ptr);
        op.set_image("sec", sec_handle.ptr);
        op.set_int("xr1", self.x_reference_1);
        op.set_int("yr1", self.y_reference_1);
        op.set_int("xs1", self.x_secondary_1);
        op.set_int("ys1", self.y_secondary_1);
        op.set_int("xr2", self.x_reference_2);
        op.set_int("yr2", self.y_reference_2);
        op.set_int("xs2", self.x_secondary_2);
        op.set_int("ys2", self.y_secondary_2);
        if let Some(v) = self.half_window {
            op.set_int("hwindow", v);
        }
        if let Some(v) = self.half_area {
            op.set_int("harea", v);
        }
        if let Some(v) = self.search {
            op.set_bool("search", v);
        }
        let out = op.run().unwrap();
        cx.emit(out);
    }
}

impl Lower<VipsBackend> for crate::Merge<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let ref_handle = cx.input(self.input.src());
        let sec_handle = cx.input(self.secondary.src());
        let mut op = VipsGObject::new(b"merge\0").unwrap();
        op.set_image("ref", ref_handle.ptr);
        op.set_image("sec", sec_handle.ptr);
        op.set_int("direction", self.direction.into_vips());
        op.set_int("dx", self.dx);
        op.set_int("dy", self.dy);
        if let Some(v) = self.max_blend {
            op.set_int("mblend", v);
        }
        let out = op.run().unwrap();
        cx.emit(out);
    }
}

impl Lower<VipsBackend> for crate::GlobalBalance<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"globalbalance\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.gamma {
            op.set_double("gamma", v);
        }
        if let Some(v) = self.integer_output {
            op.set_bool("int_output", v);
        }
        let out = op.run().unwrap();
        cx.emit(out);
    }
}

