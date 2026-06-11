use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{VipsBackend, VipsBuilder};
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

// ── Sharpen ───────────────────────────────────────────────────────────────────

pub struct Sharpen<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub sigma: Option<f64>,
    pub flat: Option<f64>,
    pub jagged: Option<f64>,
    pub edge: Option<f64>,
    pub smooth: Option<f64>,
    pub maximum: Option<f64>,
}

impl<B: Backend> Operation<B> for Sharpen<B>
where
    Sharpen<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let halo = self.sigma.unwrap_or(1.0) * 3.0; // typical 3-sigma
        vec![Some(WorkUnit::Region(out.expanded(halo.ceil() as i32)))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.sigma.unwrap_or(0.0).to_bits());
        state.write_u64(self.flat.unwrap_or(0.0).to_bits());
        state.write_u64(self.jagged.unwrap_or(0.0).to_bits());
        state.write_u64(self.edge.unwrap_or(0.0).to_bits());
        state.write_u64(self.smooth.unwrap_or(0.0).to_bits());
        state.write_u64(self.maximum.unwrap_or(0.0).to_bits());
    }
}

impl Lower<VipsBackend> for Sharpen<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"sharpen\0")
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

// ── Canny ─────────────────────────────────────────────────────────────────────

pub struct Canny<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub sigma: Option<f64>,
    pub precision: Option<i32>,
}

impl<B: Backend> Operation<B> for Canny<B>
where
    Canny<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let halo = self.sigma.unwrap_or(1.0) * 3.0;
        vec![Some(WorkUnit::Region(out.expanded(halo.ceil() as i32)))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.sigma.unwrap_or(0.0).to_bits());
        state.write_i32(self.precision.unwrap_or(0));
    }
}

impl Lower<VipsBackend> for Canny<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"canny\0")
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

// ── Median ────────────────────────────────────────────────────────────────────

pub struct Median<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub size: i32,
}

impl<B: Backend> Operation<B> for Median<B>
where
    Median<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let halo = self.size / 2;
        vec![Some(WorkUnit::Region(out.expanded(halo)))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.size);
    }
}

impl Lower<VipsBackend> for Median<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"rank\0")
            .expect("failed to create vips rank op");
        op.set_image("in", input_handle.ptr);
        op.set_int("width", self.size);
        op.set_int("height", self.size);
        op.set_int("index", self.size * self.size / 2);
        let out_handle = op.run().expect("vips rank failed");
        cx.emit(out_handle);
    }
}

// ── HoughLine ─────────────────────────────────────────────────────────────────

pub struct HoughLine<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub width: Option<i32>,
    pub height: Option<i32>,
}

impl<B: Backend> Operation<B> for HoughLine<B>
where
    HoughLine<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        // Hough usually needs full image, but for now we demand the same region
        vec![Some(WorkUnit::Region(out.clone()))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width.unwrap_or(0));
        state.write_i32(self.height.unwrap_or(0));
    }
}

impl Lower<VipsBackend> for HoughLine<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hough_line\0")
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

// ── HoughCircle ───────────────────────────────────────────────────────────────

pub struct HoughCircle<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub scale: Option<i32>,
    pub min_radius: Option<i32>,
    pub max_radius: Option<i32>,
}

impl<B: Backend> Operation<B> for HoughCircle<B>
where
    HoughCircle<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        // Hough usually needs full image, but for now we demand the same region
        vec![Some(WorkUnit::Region(out.clone()))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.scale.unwrap_or(0));
        state.write_i32(self.min_radius.unwrap_or(0));
        state.write_i32(self.max_radius.unwrap_or(0));
    }
}

impl Lower<VipsBackend> for HoughCircle<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hough_circle\0")
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

// ── Blur ──────────────────────────────────────────────────────────────────────

pub struct Blur<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub sigma: f32,
}

impl<B: Backend> Operation<B> for Blur<B>
where
    Blur<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let halo = (self.sigma * 3.0).ceil() as i32;
        vec![Some(WorkUnit::Region(out.expanded(halo)))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.sigma.to_bits());
    }
}

impl Lower<VipsBackend> for Blur<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"gaussblur\0")
            .expect("failed to create vips gaussblur op");
        op.set_image("in", input_handle.ptr);
        op.set_double("sigma", self.sigma as f64);
        let out_handle = op.run().expect("vips gaussblur failed");
        cx.emit(out_handle);
    }
}
