use std::hash::Hasher;

use crate::backend::gpu::view::ParamBlock;
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
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

impl Lower<GpuBackend> for Sharpen<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let scale = if let WorkUnit::Region(r) = &wu { r.lod.scale_factor() as f32 } else { 1.0 };
        let sigma = self.sigma.unwrap_or(0.5) as f32 / scale;
        let m1 = self.smooth.unwrap_or(1.0) as f32;
        cx.param_block(ParamBlock::new().param("sigma", sigma).param("m1", m1));
        cx.kernel("ops.filters", "sharpen_kernel");
        cx.output(self.output_spec().output(cx.wu()));
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

impl Lower<GpuBackend> for Median<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(ParamBlock::new().param("sz", self.size as u32));
        cx.kernel("ops.filters", "median_kernel");
        cx.output(self.output_spec().output(cx.wu()));
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
        let halo = gauss_radius(self.sigma / out.lod.scale_factor() as f32);
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

/// Mask radius matching `vips_gaussmat`'s default (`min_ampl = 0.2`): the
/// largest `x` with `exp(-x^2 / (2*sigma^2)) >= min_ampl`. Much narrower than
/// a naive `sigma*3` window -- e.g. radius 5 (not 9) for sigma=3.
fn gauss_radius(sigma: f32) -> i32 {
    let sig2 = 2.0 * sigma * sigma;
    let max_x = (8.0 * sigma) as i32;
    let mut x = 0;
    while x < max_x {
        let v = (-((x * x) as f32) / sig2).exp();
        if v < 0.2 {
            break;
        }
        x += 1;
    }
    (x - 1).max(0)
}

impl Lower<GpuBackend> for Blur<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let scale = if let WorkUnit::Region(r) = &wu { r.lod.scale_factor() as f32 } else { 1.0 };
        let sigma = self.sigma / scale;
        // Single-pass 2D kernel (not separable H/V): a separable fused
        // two-step pass would have the V step read NEIGHBOR threads' H
        // output, which a single dispatch can't order across workgroups.
        cx.kernel("ops.filters", "blur_kernel");
        cx.param("sigma", sigma);
        cx.param("radius", gauss_radius(sigma));
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl<B: Backend> crate::data::image::Image2D<B>
where
    Blur<B>: Lower<B>,
{
    pub fn blur(&self, sigma: f32) -> Self {
        self.push(Blur { input: self.as_input(), sigma })
    }
}


impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Sharpen<B>: crate::operation::Lower<B>,
{
    pub fn sharpen(&self, sigma: Option<f64>, flat: Option<f64>, jagged: Option<f64>, edge: Option<f64>, smooth: Option<f64>, maximum: Option<f64>) -> Self {
        self.push(Sharpen { input: self.as_input(), sigma, flat, jagged, edge, smooth, maximum })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Canny<B>: crate::operation::Lower<B>,
{
    pub fn canny(&self, sigma: Option<f64>, precision: Option<i32>) -> Self {
        self.push(Canny { input: self.as_input(), sigma, precision })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Median<B>: crate::operation::Lower<B>,
{
    pub fn median(&self, size: i32) -> Self {
        self.push(Median { input: self.as_input(), size })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HoughLine<B>: crate::operation::Lower<B>,
{
    pub fn hough_line(&self, width: Option<i32>, height: Option<i32>) -> Self {
        self.push(HoughLine { input: self.as_input(), width, height })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HoughCircle<B>: crate::operation::Lower<B>,
{
    pub fn hough_circle(&self, scale: Option<i32>, min_radius: Option<i32>, max_radius: Option<i32>) -> Self {
        self.push(HoughCircle { input: self.as_input(), scale, min_radius, max_radius })
    }
}
