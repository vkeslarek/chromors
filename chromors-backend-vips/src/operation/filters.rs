use crate::prelude::*;

impl Lower<VipsBackend> for crate::Sharpen<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"sharpen\0")
            .expect("failed to create vips sharpen op");
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.sigma {
            op.set_double("sigma", v);
        }
        if let Some(v) = self.flat {
            op.set_double("x1", v);
        }
        if let Some(v) = self.jagged {
            op.set_double("y2", v);
        }
        if let Some(v) = self.edge {
            op.set_double("y3", v);
        }
        if let Some(v) = self.smooth {
            op.set_double("m1", v);
        }
        if let Some(v) = self.maximum {
            op.set_double("m2", v);
        }
        let out_handle = op.run().expect("vips sharpen failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Canny<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"canny\0")
            .expect("failed to create vips canny op");
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.sigma {
            op.set_double("sigma", v);
        }
        if let Some(v) = self.precision {
            op.set_int("precision", v);
        }
        let out_handle = op.run().expect("vips canny failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Median<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"rank\0")
            .expect("failed to create vips rank op");
        op.set_image("in", input_handle.ptr);
        op.set_int("width", self.size);
        op.set_int("height", self.size);
        op.set_int("index", self.size * self.size / 2);
        let out_handle = op.run().expect("vips rank failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::HoughLine<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"hough_line\0")
            .expect("failed to create vips hough_line op");
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.width {
            op.set_int("width", v);
        }
        if let Some(v) = self.height {
            op.set_int("height", v);
        }
        let out_handle = op.run().expect("vips hough_line failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::HoughCircle<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"hough_circle\0")
            .expect("failed to create vips hough_circle op");
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.scale {
            op.set_int("scale", v);
        }
        if let Some(v) = self.min_radius {
            op.set_int("min_radius", v);
        }
        if let Some(v) = self.max_radius {
            op.set_int("max_radius", v);
        }
        let out_handle = op.run().expect("vips hough_circle failed");
        cx.emit(out_handle);
    }
}

impl Lower<VipsBackend> for crate::Blur<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"gaussblur\0")
            .expect("failed to create vips gaussblur op");
        op.set_image("in", input_handle.ptr);
        op.set_double("sigma", self.sigma as f64);
        let out_handle = op.run().expect("vips gaussblur failed");
        cx.emit(out_handle);
    }
}

