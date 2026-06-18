use crate::prelude::*;

impl chromors_core::operation::Lower<VipsBackend> for crate::HistogramFind<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"hist_find\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.band {
            op.set_int("band", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl chromors_core::operation::Lower<VipsBackend> for crate::HistogramEqualize<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"hist_equal\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.band {
            op.set_int("band", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl chromors_core::operation::Lower<VipsBackend> for chromors_core::operation::stats::HistogramCumulative<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"hist_cum\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl chromors_core::operation::Lower<VipsBackend> for chromors_core::operation::stats::HistogramNormalize<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"hist_norm\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl chromors_core::operation::Lower<VipsBackend> for crate::HistogramPlot<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"hist_plot\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl chromors_core::operation::Lower<VipsBackend> for crate::HistFindIndexed<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let index_handle = cx.input(self.index.src());
        let mut op =
            VipsGObject::new(b"hist_find_indexed\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("index", index_handle.ptr);
        if let Some(v) = self.combine {
            op.set_int("combine", v.into_vips());
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl chromors_core::operation::Lower<VipsBackend> for crate::HistFindNdim<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"hist_find_ndim\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.bins {
            op.set_int("bins", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl chromors_core::operation::Lower<VipsBackend> for crate::HistLocal<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"hist_local\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(v) = self.max_slope {
            op.set_int("max_slope", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl chromors_core::operation::Lower<VipsBackend> for crate::HistMatch<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let ref_handle = cx.input(self.ref_image.src());
        let mut op = VipsGObject::new(b"hist_match\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("ref", ref_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

impl chromors_core::operation::Lower<VipsBackend> for crate::Stdif<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = VipsGObject::new(b"stdif\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(v) = self.a {
            op.set_double("a", v);
        }
        if let Some(v) = self.m0 {
            op.set_double("m0", v);
        }
        if let Some(v) = self.b {
            op.set_double("b", v);
        }
        if let Some(v) = self.s0 {
            op.set_double("s0", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

