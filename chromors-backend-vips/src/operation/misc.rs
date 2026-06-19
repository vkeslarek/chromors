use crate::prelude::*;

impl Lower<VipsBackend> for crate::Cast<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"cast\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("format", self.target.storage.into_vips_band_format());
        if let Some(v) = self.shift {
            op.set_bool("shift", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Copy<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"copy\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.width {
            op.set_int("width", v);
        }
        if let Some(v) = self.height {
            op.set_int("height", v);
        }
        if let Some(v) = self.bands {
            op.set_int("bands", v);
        }
        if let Some(v) = self.format {
            op.set_int("format", v);
        }
        if let Some(v) = self.interpretation {
            op.set_int("interpretation", v);
        }
        if let Some(v) = self.horizontal_resolution {
            op.set_double("xres", v);
        }
        if let Some(v) = self.vertical_resolution {
            op.set_double("yres", v);
        }
        if let Some(v) = self.offset_x {
            op.set_int("xoffset", v);
        }
        if let Some(v) = self.offset_y {
            op.set_int("yoffset", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::TileCache<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"tilecache\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.tile_width {
            op.set_int("tile_width", v);
        }
        if let Some(v) = self.tile_height {
            op.set_int("tile_height", v);
        }
        if let Some(v) = self.maximum_tiles {
            op.set_int("max_tiles", v);
        }
        if let Some(v) = self.access {
            op.set_int("access", v.into_vips());
        }
        if let Some(v) = self.threaded {
            op.set_bool("threaded", v);
        }
        if let Some(v) = self.persistent {
            op.set_bool("persistent", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Msb<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"msb\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.band {
            op.set_int("band", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Maplut<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let lut_handle = cx.input(self.lut.src());
        let mut op = VipsGObject::new(b"maplut\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("lut", lut_handle.ptr);
        if let Some(v) = self.band {
            op.set_int("band", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Recomb<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        use crate::IntoVipsBandFormat;
        let input_handle = cx.input(self.input.src());
        let matrix_handle = cx.input(self.matrix.src());
        let mut op = VipsGObject::new(b"recomb\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("m", matrix_handle.ptr);
        let out_handle = op.run().unwrap();

        // vips_recomb always promotes to float, but output_spec() keeps the
        // input's format (matching the GPU lowering, which stays in the
        // original format) -- cast back down so both backends' outputs share
        // the same band format/byte layout.
        let mut cast_op = VipsGObject::new(b"cast\0").unwrap();
        cast_op.set_image("in", out_handle.ptr);
        cast_op.set_int(
            "format",
            self.input.spec.layout.storage.into_vips_band_format(),
        );
        cast_op.set_bool("shift", false);
        let cast_handle = cast_op.run().unwrap();
        cx.emit(cast_handle);
    }
}

impl Lower<VipsBackend> for crate::Ifthenelse<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let cond_handle = cx.input(self.cond.src());
        let true_handle = cx.input(self.if_true.src());
        let false_handle = cx.input(self.if_false.src());
        let mut op = VipsGObject::new(b"ifthenelse\0").unwrap();
        op.set_image("cond", cond_handle.ptr);
        op.set_image("in1", true_handle.ptr);
        op.set_image("in2", false_handle.ptr);
        if let Some(v) = self.blend {
            op.set_bool("blend", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Invertlut<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"invertlut\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.size {
            op.set_int("size", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Linecache<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"linecache\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.tile_height {
            op.set_int("tile_height", v);
        }
        if let Some(v) = self.access {
            op.set_int("access", v.into_vips());
        }
        if let Some(v) = self.threaded {
            op.set_bool("threaded", v);
        }
        if let Some(v) = self.persistent {
            op.set_bool("persistent", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Case<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let index_handle = cx.input(self.input.src());
        let mut case_ptrs: Vec<*mut crate::ffi::VipsImage> = vec![];
        for c in &self.cases {
            let handle = cx.input(c.src());
            case_ptrs.push(handle.ptr);
        }
        let mut op = VipsGObject::new(b"case\0").unwrap();
        op.set_image("index", index_handle.ptr);
        op.set_array_image("cases", &case_ptrs);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Exposure<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let gain = 2.0f64.powf(self.stops as f64);
        let mut op = VipsGObject::new(b"linear\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_array_double("a", &[gain]);
        op.set_array_double("b", &[0.0]);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Brightness<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"linear\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_array_double("a", &[self.value as f64]);
        op.set_array_double("b", &[0.0]);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::NoiseReduction<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"rank\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let size = self.size();
        op.set_int("width", size);
        op.set_int("height", size);
        op.set_int("index", size * size / 2);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Saturation<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let ptr = input_handle.ptr;
        let bands = unsafe { crate::ffi::vips_image_get_bands(ptr) };
        if bands < 3 {
            // Grayscale: saturation has no effect
            unsafe { crate::ffi::g_object_ref(ptr as *mut _) };
            cx.emit(VipsHandle { ptr });
            return;
        }

        let ext = |b| {
            let mut o = VipsGObject::new(b"extract_band\0").unwrap();
            o.set_image("in", ptr);
            o.set_int("band", b);
            o.run().unwrap().ptr
        };
        let r = ext(0);
        let g = ext(1);
        let b = ext(2);

        let mul = |p, w: f64| {
            let mut o = VipsGObject::new(b"linear\0").unwrap();
            o.set_image("in", p);
            o.set_array_double("a", &[w]);
            o.set_array_double("b", &[0.0]);
            o.run().unwrap().ptr
        };
        let luma_r = mul(r, 0.2126);
        let luma_g = mul(g, 0.7152);
        let luma_b = mul(b, 0.0722);

        let add = |p1, p2| {
            let mut o = VipsGObject::new(b"add\0").unwrap();
            o.set_image("left", p1);
            o.set_image("right", p2);
            o.run().unwrap().ptr
        };
        let luma1 = add(luma_r, luma_g);
        let luma = add(luma1, luma_b);

        let mut op_rgb = VipsGObject::new(b"extract_band\0").unwrap();
        op_rgb.set_image("in", ptr);
        op_rgb.set_int("band", 0);
        op_rgb.set_int("n", 3);
        let rgb_ptr = op_rgb.run().unwrap().ptr;

        let rgb_scaled = mul(rgb_ptr, self.amount as f64);
        let luma_scaled = mul(luma, 1.0 - self.amount as f64);

        let out_rgb = add(rgb_scaled, luma_scaled);

        let out_ptr = if bands > 3 {
            let mut op_a = VipsGObject::new(b"extract_band\0").unwrap();
            op_a.set_image("in", ptr);
            op_a.set_int("band", 3);
            op_a.set_int("n", bands - 3);
            let a_ptr = op_a.run().unwrap().ptr;

            let mut out: *mut crate::ffi::VipsImage = std::ptr::null_mut();
            let arr = [out_rgb, a_ptr];
            let ret = unsafe {
                crate::ffi::vips_bandjoin(arr.as_ptr() as *mut *mut _, &mut out, 2, crate::null())
            };
            if ret != 0 {
                panic!("vips_bandjoin failed");
            }
            out
        } else {
            out_rgb
        };
        cx.emit(VipsHandle { ptr: out_ptr });
    }
}

impl Lower<VipsBackend> for crate::Boolean<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"boolean\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        op.set_int("boolean", self.boolean_op.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Relational<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let left_handle = cx.input(self.left.src());
        let right_handle = cx.input(self.right.src());
        let mut op = VipsGObject::new(b"relational\0").unwrap();
        op.set_image("left", left_handle.ptr);
        op.set_image("right", right_handle.ptr);
        op.set_int("relational", self.relational.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::BooleanConst<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"boolean_const\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("boolean", self.boolean_op.into_vips());
        op.set_array_double("c", &self.c);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::RelationalConst<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"relational_const\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("relational", self.relational.into_vips());
        op.set_array_double("c", &self.c);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}
