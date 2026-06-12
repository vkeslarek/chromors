use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{IntoVipsEnum, VipsBackend, VipsBuilder};
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CombineMode {
    Max,
    Sum,
    Min,
}
impl IntoVipsEnum for CombineMode {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

// ── HistFind ──────────────────────────────────────────────────────────────────

pub struct HistogramFind<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub band: Option<i32>,
}
impl<B: Backend> Operation<B> for HistogramFind<B> where HistogramFind<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        // Histograms typically scan the entire input
        vec![Some(WorkUnit::Region(Region { x: 0, y: 0, w: self.input.spec.width, h: self.input.spec.height, lod: out.lod }))]
    }
    // vips_hist_find: 1D histogram, `bins x 1`, one band per input band
    // (or a single band if `band` selects one). bins is 2^bits for the
    // input's sample depth (256 for 8-bit formats; the POC only models 8-bit
    // histograms here).
    fn output_spec(&self) -> ImageKind {
        let input = &*self.input.spec;
        let bands = match self.band {
            Some(_) => 1,
            None => input.format.channel_count() as i32,
        };
        ImageKind {
            width: 256,
            height: 1,
            format: input.with_band_count(bands),
            color_space: input.color_space,
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.band { state.write_i32(v); }
    }
}
impl Lower<VipsBackend> for HistogramFind<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hist_find\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.band { op.set_int("band", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── HistEqual ─────────────────────────────────────────────────────────────────

pub struct HistogramEqualize<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub band: Option<i32>,
}
impl<B: Backend> Operation<B> for HistogramEqualize<B> where HistogramEqualize<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.band { state.write_i32(v); }
    }
}
impl Lower<VipsBackend> for HistogramEqualize<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hist_equal\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.band { op.set_int("band", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── HistCum ───────────────────────────────────────────────────────────────────

pub struct HistogramCumulative<B: Backend> {
    pub input: Input<ImageKind, B>,
}
impl<B: Backend> Operation<B> for HistogramCumulative<B> where HistogramCumulative<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}
impl Lower<VipsBackend> for HistogramCumulative<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hist_cum\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── HistNorm ──────────────────────────────────────────────────────────────────

pub struct HistogramNormalize<B: Backend> {
    pub input: Input<ImageKind, B>,
}
impl<B: Backend> Operation<B> for HistogramNormalize<B> where HistogramNormalize<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}
impl Lower<VipsBackend> for HistogramNormalize<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hist_norm\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── HistPlot ──────────────────────────────────────────────────────────────────

pub struct HistogramPlot<B: Backend> {
    pub input: Input<ImageKind, B>,
}
impl<B: Backend> Operation<B> for HistogramPlot<B> where HistogramPlot<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    // vips_hist_plot renders a histogram image (e.g. `256 x 1`) into a square
    // chart, `width x width`, preserving the band count.
    fn output_spec(&self) -> ImageKind {
        let input = &*self.input.spec;
        ImageKind {
            width: input.width,
            height: input.width,
            format: input.format,
            color_space: input.color_space,
        }
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}
impl Lower<VipsBackend> for HistogramPlot<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hist_plot\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── HistFindIndexed ───────────────────────────────────────────────────────────

pub struct HistFindIndexed<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub index: Input<ImageKind, B>,
    pub combine: Option<CombineMode>,
}
impl<B: Backend> Operation<B> for HistFindIndexed<B> where HistFindIndexed<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.index] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(Region { x: 0, y: 0, w: self.input.spec.width, h: self.input.spec.height, lod: out.lod })),
            Some(WorkUnit::Region(Region { x: 0, y: 0, w: self.index.spec.width, h: self.index.spec.height, lod: out.lod })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.combine { state.write_i32(v.into_vips()); }
    }
}
impl Lower<VipsBackend> for HistFindIndexed<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let index_handle = cx.input(self.index.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hist_find_indexed\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("index", index_handle.ptr);
        if let Some(v) = self.combine { op.set_int("combine", v.into_vips()); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── HistFindNdim ──────────────────────────────────────────────────────────────

pub struct HistFindNdim<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub bins: Option<i32>,
}
impl<B: Backend> Operation<B> for HistFindNdim<B> where HistFindNdim<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(Region { x: 0, y: 0, w: self.input.spec.width, h: self.input.spec.height, lod: out.lod }))]
    }
    // vips_hist_find_ndim: N-dimensional histogram (N = input band count,
    // default bins = 10), flattened to 2D. For 2 bands: `bins x bins`. For
    // higher band counts vips flattens the extra dims into height
    // (`bins x bins^(N-1)`); only the 2-band case is handled precisely here.
    fn output_spec(&self) -> ImageKind {
        let input = &*self.input.spec;
        let bins = self.bins.unwrap_or(10);
        let bands = input.format.channel_count() as i32;
        // TODO: for bands > 2, vips flattens the extra dimensions into
        // height (bins^(bands-1)); this only covers the bands == 2 case
        // precisely (and bands == 1, where height collapses to 1).
        let height = if bands <= 1 { 1 } else { bins.pow((bands - 1) as u32) };
        ImageKind {
            width: bins,
            height,
            format: input.with_band_count(1),
            color_space: input.color_space,
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.bins { state.write_i32(v); }
    }
}
impl Lower<VipsBackend> for HistFindNdim<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hist_find_ndim\0").unwrap();
        op.set_image("in", input_handle.ptr);
        if let Some(v) = self.bins { op.set_int("bins", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── HistLocal ─────────────────────────────────────────────────────────────────

pub struct HistLocal<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub width: i32,
    pub height: i32,
    pub max_slope: Option<i32>,
}
impl<B: Backend> Operation<B> for HistLocal<B> where HistLocal<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height);
        if let Some(v) = self.max_slope { state.write_i32(v); }
    }
}
impl Lower<VipsBackend> for HistLocal<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hist_local\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(v) = self.max_slope { op.set_int("max_slope", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── HistMatch ─────────────────────────────────────────────────────────────────

pub struct HistMatch<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub ref_image: Input<ImageKind, B>,
}
impl<B: Backend> Operation<B> for HistMatch<B> where HistMatch<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.ref_image] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Region(Region { x: 0, y: 0, w: self.ref_image.spec.width, h: self.ref_image.spec.height, lod: out.lod })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}
impl Lower<VipsBackend> for HistMatch<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let ref_handle = cx.input(self.ref_image.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"hist_match\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("ref", ref_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Stdif ─────────────────────────────────────────────────────────────────────

pub struct Stdif<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub width: i32,
    pub height: i32,
    pub a: Option<f64>,
    pub m0: Option<f64>,
    pub b: Option<f64>,
    pub s0: Option<f64>,
}
impl<B: Backend> Operation<B> for Stdif<B> where Stdif<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> { vec![Some(WorkUnit::Region(out.clone()))] }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height);
        if let Some(v) = self.a { state.write_u64(v.to_bits()); }
        if let Some(v) = self.m0 { state.write_u64(v.to_bits()); }
        if let Some(v) = self.b { state.write_u64(v.to_bits()); }
        if let Some(v) = self.s0 { state.write_u64(v.to_bits()); }
    }
}
impl Lower<VipsBackend> for Stdif<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"stdif\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(v) = self.a { op.set_double("a", v); }
        if let Some(v) = self.m0 { op.set_double("m0", v); }
        if let Some(v) = self.b { op.set_double("b", v); }
        if let Some(v) = self.s0 { op.set_double("s0", v); }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}


impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistogramFind<B>: crate::operation::Lower<B>,
{
    pub fn histogram_find(&self, band: Option<i32>) -> Self {
        self.push(HistogramFind { input: self.as_input(), band })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistogramEqualize<B>: crate::operation::Lower<B>,
{
    pub fn histogram_equalize(&self, band: Option<i32>) -> Self {
        self.push(HistogramEqualize { input: self.as_input(), band })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistogramCumulative<B>: crate::operation::Lower<B>,
{
    pub fn histogram_cumulative(&self) -> Self {
        self.push(HistogramCumulative { input: self.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistogramNormalize<B>: crate::operation::Lower<B>,
{
    pub fn histogram_normalize(&self) -> Self {
        self.push(HistogramNormalize { input: self.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistogramPlot<B>: crate::operation::Lower<B>,
{
    pub fn histogram_plot(&self) -> Self {
        self.push(HistogramPlot { input: self.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistFindIndexed<B>: crate::operation::Lower<B>,
{
    pub fn hist_find_indexed(&self, index: Input<ImageKind, B>, combine: Option<CombineMode>) -> Self {
        self.push(HistFindIndexed { input: self.as_input(), index, combine })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistFindNdim<B>: crate::operation::Lower<B>,
{
    pub fn hist_find_ndim(&self, bins: Option<i32>) -> Self {
        self.push(HistFindNdim { input: self.as_input(), bins })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistLocal<B>: crate::operation::Lower<B>,
{
    pub fn hist_local(&self, width: i32, height: i32, max_slope: Option<i32>) -> Self {
        self.push(HistLocal { input: self.as_input(), width, height, max_slope })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistMatch<B>: crate::operation::Lower<B>,
{
    pub fn hist_match(&self, ref_image: Input<ImageKind, B>) -> Self {
        self.push(HistMatch { input: self.as_input(), ref_image })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Stdif<B>: crate::operation::Lower<B>,
{
    pub fn stdif(&self, width: i32, height: i32, a: Option<f64>, m0: Option<f64>, b: Option<f64>, s0: Option<f64>) -> Self {
        self.push(Stdif { input: self.as_input(), width, height, a, m0, b, s0 })
    }
}
