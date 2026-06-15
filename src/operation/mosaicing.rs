use std::hash::Hasher;

use super::Direction;
use crate::backend::Backend;
use crate::backend::gpu::view::ParamBlock;
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::vips::{IntoVipsEnum, VipsBackend, VipsBuilder};
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

// ── Mosaic ────────────────────────────────────────────────────────────────────

pub struct Mosaic<B: Backend> {
    pub input: Input<ImageKind, B>, // reference
    pub secondary: Input<ImageKind, B>,
    pub direction: Direction,
    pub x_reference: i32,
    pub y_reference: i32,
    pub x_secondary: i32,
    pub y_secondary: i32,
    pub half_window: Option<i32>,
    pub half_area: Option<i32>,
    pub max_blend: Option<i32>,
    pub search_band: Option<i32>,
}

impl<B: Backend> Operation<B> for Mosaic<B>
where
    Mosaic<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.secondary]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        // Mosaicing generally requires the full images
        vec![
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.input.spec.width,
                h: self.input.spec.height,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.secondary.spec.width,
                h: self.secondary.spec.height,
                lod: out.lod,
            })),
        ]
    }
    // TODO: vips_mosaic's real output is the bounding box of `input` and
    // `secondary` after alignment search (offset is data-dependent, found at
    // run time), so it's larger than either input. There's no struct field to
    // compute that bound from statically; this placeholder (input dims) is
    // wrong but harmless for now.
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.direction.into_vips());
        state.write_i32(self.x_reference);
        state.write_i32(self.y_reference);
        state.write_i32(self.x_secondary);
        state.write_i32(self.y_secondary);
        if let Some(v) = self.half_window {
            state.write_i32(v);
        }
        if let Some(v) = self.half_area {
            state.write_i32(v);
        }
        if let Some(v) = self.max_blend {
            state.write_i32(v);
        }
        if let Some(v) = self.search_band {
            state.write_i32(v);
        }
    }
}

impl Lower<VipsBackend> for Mosaic<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let ref_handle = cx.input(self.input.src());
        let sec_handle = cx.input(self.secondary.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"mosaic\0").unwrap();
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

// ── Mosaic1 ───────────────────────────────────────────────────────────────────

pub struct Mosaic1<B: Backend> {
    pub input: Input<ImageKind, B>, // reference
    pub secondary: Input<ImageKind, B>,
    pub direction: Direction,
    pub x_reference_1: i32,
    pub y_reference_1: i32,
    pub x_secondary_1: i32,
    pub y_secondary_1: i32,
    pub x_reference_2: i32,
    pub y_reference_2: i32,
    pub x_secondary_2: i32,
    pub y_secondary_2: i32,
    pub half_window: Option<i32>,
    pub half_area: Option<i32>,
    pub search: Option<bool>,
    // Removed interpolate for now
    pub max_blend: Option<i32>,
}

impl<B: Backend> Operation<B> for Mosaic1<B>
where
    Mosaic1<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.secondary]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.input.spec.width,
                h: self.input.spec.height,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.secondary.spec.width,
                h: self.secondary.spec.height,
                lod: out.lod,
            })),
        ]
    }
    // TODO: vips_mosaic1's real output is the bounding box of `input` and
    // `secondary` after alignment search (offset is data-dependent), larger
    // than either input. No struct field gives a static bound; placeholder.
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.direction.into_vips());
        state.write_i32(self.x_reference_1);
        state.write_i32(self.y_reference_1);
        state.write_i32(self.x_secondary_1);
        state.write_i32(self.y_secondary_1);
        state.write_i32(self.x_reference_2);
        state.write_i32(self.y_reference_2);
        state.write_i32(self.x_secondary_2);
        state.write_i32(self.y_secondary_2);
        if let Some(v) = self.half_window {
            state.write_i32(v);
        }
        if let Some(v) = self.half_area {
            state.write_i32(v);
        }
        if let Some(v) = self.search {
            state.write_u8(v as u8);
        }
        if let Some(v) = self.max_blend {
            state.write_i32(v);
        }
    }
}

impl Lower<VipsBackend> for Mosaic1<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let ref_handle = cx.input(self.input.src());
        let sec_handle = cx.input(self.secondary.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"mosaic1\0").unwrap();
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

// ── Match ─────────────────────────────────────────────────────────────────────

pub struct Match<B: Backend> {
    pub input: Input<ImageKind, B>, // reference
    pub secondary: Input<ImageKind, B>,
    pub x_reference_1: i32,
    pub y_reference_1: i32,
    pub x_secondary_1: i32,
    pub y_secondary_1: i32,
    pub x_reference_2: i32,
    pub y_reference_2: i32,
    pub x_secondary_2: i32,
    pub y_secondary_2: i32,
    pub half_window: Option<i32>,
    pub half_area: Option<i32>,
    pub search: Option<bool>,
}

impl<B: Backend> Operation<B> for Match<B>
where
    Match<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.secondary]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.input.spec.width,
                h: self.input.spec.height,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.secondary.spec.width,
                h: self.secondary.spec.height,
                lod: out.lod,
            })),
        ]
    }
    // TODO: vips_match's real output is the bounding box of `input` and
    // `secondary` after the tie-point alignment, larger than either input.
    // No struct field gives a static bound; placeholder.
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.x_reference_1);
        state.write_i32(self.y_reference_1);
        state.write_i32(self.x_secondary_1);
        state.write_i32(self.y_secondary_1);
        state.write_i32(self.x_reference_2);
        state.write_i32(self.y_reference_2);
        state.write_i32(self.x_secondary_2);
        state.write_i32(self.y_secondary_2);
        if let Some(v) = self.half_window {
            state.write_i32(v);
        }
        if let Some(v) = self.half_area {
            state.write_i32(v);
        }
        if let Some(v) = self.search {
            state.write_u8(v as u8);
        }
    }
}

impl Lower<VipsBackend> for Match<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let ref_handle = cx.input(self.input.src());
        let sec_handle = cx.input(self.secondary.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"match\0").unwrap();
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

// ── Merge ─────────────────────────────────────────────────────────────────────

pub struct Merge<B: Backend> {
    pub input: Input<ImageKind, B>, // reference
    pub secondary: Input<ImageKind, B>,
    pub direction: Direction,
    pub dx: i32,
    pub dy: i32,
    pub max_blend: Option<i32>,
}

impl<B: Backend> Operation<B> for Merge<B>
where
    Merge<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.secondary]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.input.spec.width,
                h: self.input.spec.height,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.secondary.spec.width,
                h: self.secondary.spec.height,
                lod: out.lod,
            })),
        ]
    }
    // vips_merge places `secondary` at offset (dx, dy) relative to `input`;
    // the output canvas is the bounding box of both.
    fn output_spec(&self) -> ImageKind {
        let input = &*self.input.spec;
        let sec = &*self.secondary.spec;
        let left = 0.min(self.dx);
        let top = 0.min(self.dy);
        let right = input.width.max(self.dx + sec.width);
        let bottom = input.height.max(self.dy + sec.height);
        ImageKind {
            width: right - left,
            height: bottom - top,
            layout: input.layout,
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.direction.into_vips());
        state.write_i32(self.dx);
        state.write_i32(self.dy);
        if let Some(v) = self.max_blend {
            state.write_i32(v);
        }
    }
}

impl Lower<VipsBackend> for Merge<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let ref_handle = cx.input(self.input.src());
        let sec_handle = cx.input(self.secondary.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"merge\0").unwrap();
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

impl Lower<GpuBackend> for Merge<GpuBackend> {
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

// ── GlobalBalance ─────────────────────────────────────────────────────────────

pub struct GlobalBalance<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub gamma: Option<f64>,
    pub integer_output: Option<bool>,
}

impl<B: Backend> Operation<B> for GlobalBalance<B>
where
    GlobalBalance<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    // TODO: vips_globalbalance's real output is the bounding box of all
    // mosaiced pieces in `input`'s assembly graph, which can differ from
    // `input`'s own dims. `input` here is a single image with no per-piece
    // bbox info available statically; placeholder.
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.gamma {
            state.write(&v.to_ne_bytes());
        }
        if let Some(v) = self.integer_output {
            state.write_u8(v as u8);
        }
    }
}

impl Lower<VipsBackend> for GlobalBalance<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"globalbalance\0").unwrap();
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

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Mosaic<B>: crate::operation::Lower<B>,
{
    pub fn mosaic(
        &self,
        secondary: Input<ImageKind, B>,
        direction: Direction,
        x_reference: i32,
        y_reference: i32,
        x_secondary: i32,
        y_secondary: i32,
        half_window: Option<i32>,
        half_area: Option<i32>,
        max_blend: Option<i32>,
        search_band: Option<i32>,
    ) -> Self {
        self.push(Mosaic {
            input: self.as_input(),
            secondary,
            direction,
            x_reference,
            y_reference,
            x_secondary,
            y_secondary,
            half_window,
            half_area,
            max_blend,
            search_band,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Mosaic1<B>: crate::operation::Lower<B>,
{
    pub fn mosaic1(
        &self,
        secondary: Input<ImageKind, B>,
        direction: Direction,
        x_reference_1: i32,
        y_reference_1: i32,
        x_secondary_1: i32,
        y_secondary_1: i32,
        x_reference_2: i32,
        y_reference_2: i32,
        x_secondary_2: i32,
        y_secondary_2: i32,
        half_window: Option<i32>,
        half_area: Option<i32>,
        search: Option<bool>,
        max_blend: Option<i32>,
    ) -> Self {
        self.push(Mosaic1 {
            input: self.as_input(),
            secondary,
            direction,
            x_reference_1,
            y_reference_1,
            x_secondary_1,
            y_secondary_1,
            x_reference_2,
            y_reference_2,
            x_secondary_2,
            y_secondary_2,
            half_window,
            half_area,
            search,
            max_blend,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Match<B>: crate::operation::Lower<B>,
{
    pub fn match_op(
        &self,
        secondary: Input<ImageKind, B>,
        x_reference_1: i32,
        y_reference_1: i32,
        x_secondary_1: i32,
        y_secondary_1: i32,
        x_reference_2: i32,
        y_reference_2: i32,
        x_secondary_2: i32,
        y_secondary_2: i32,
        half_window: Option<i32>,
        half_area: Option<i32>,
        search: Option<bool>,
    ) -> Self {
        self.push(Match {
            input: self.as_input(),
            secondary,
            x_reference_1,
            y_reference_1,
            x_secondary_1,
            y_secondary_1,
            x_reference_2,
            y_reference_2,
            x_secondary_2,
            y_secondary_2,
            half_window,
            half_area,
            search,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Merge<B>: crate::operation::Lower<B>,
{
    pub fn merge(
        &self,
        secondary: Input<ImageKind, B>,
        direction: Direction,
        dx: i32,
        dy: i32,
        max_blend: Option<i32>,
    ) -> Self {
        self.push(Merge {
            input: self.as_input(),
            secondary,
            direction,
            dx,
            dy,
            max_blend,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    GlobalBalance<B>: crate::operation::Lower<B>,
{
    pub fn global_balance(&self, gamma: Option<f64>, integer_output: Option<bool>) -> Self {
        self.push(GlobalBalance {
            input: self.as_input(),
            gamma,
            integer_output,
        })
    }
}
