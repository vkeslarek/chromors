use crate::prelude::*;

impl Lower<GpuBackend> for crate::Merge<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let ref_w = self.input.spec.width;
        let ref_h = self.input.spec.height;
        let sec_w = self.secondary.spec.width;
        let sec_h = self.secondary.spec.height;

        // vips' merge places `sec` at (-dx,-dy) relative to `ref`'s origin
        // (0,0) (see libvips lrmerge/tbmerge: `sarea.left = -dx`); rarea =
        // (0,0,ref_w,ref_h), sarea = (-dx,-dy,sec_w,sec_h), oarea =
        // union(rarea,sarea) translated to start at (0,0).
        let oarea_left = (-self.dx).min(0);
        let oarea_top = (-self.dy).min(0);
        let r_left = -oarea_left;
        let r_top = -oarea_top;
        let s_left = -self.dx - oarea_left;
        let s_top = -self.dy - oarea_top;

        let overlap_left = r_left.max(s_left);
        let overlap_top = r_top.max(s_top);
        let overlap_right = (r_left + ref_w).min(s_left + sec_w);
        let overlap_bottom = (r_top + ref_h).min(s_top + sec_h);

        let (mut first, mut bwidth) = match self.direction {
            Direction::Horizontal => (overlap_left, (overlap_right - overlap_left).max(0)),
            Direction::Vertical => (overlap_top, (overlap_bottom - overlap_top).max(0)),
        };
        // vips_merge defaults `mblend` to 10.
        let mblend = self.max_blend.unwrap_or(10);
        if mblend >= 0 && bwidth > mblend {
            let shrink = (bwidth - mblend) / 2;
            first += shrink;
            bwidth -= shrink * 2;
        }

        cx.param_block(
            ParamBlock::new()
                .param("direction", self.direction.into_vips() as u32)
                .param("r_left", r_left)
                .param("r_top", r_top)
                .param("s_left", s_left)
                .param("s_top", s_top)
                .param("ref_w", ref_w as u32)
                .param("ref_h", ref_h as u32)
                .param("sec_w", sec_w as u32)
                .param("sec_h", sec_h as u32)
                .param("first", first)
                .param("bwidth", bwidth as f32),
        );
        cx.kernel("ops.mosaicing", "merge_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

