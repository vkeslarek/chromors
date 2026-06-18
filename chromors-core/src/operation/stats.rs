use std::hash::Hasher;
use crate::operation::IntoVipsEnum;

use crate::backend::Backend;
use crate::data::image::{Image2D, ImageKind};
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Atomic, Lod, Range, Region, WorkUnit};

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
impl<B: Backend> Operation<B> for HistogramFind<B>
where
    HistogramFind<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        // Histograms typically scan the entire input
        vec![Some(WorkUnit::Region(Region {
            x: 0,
            y: 0,
            w: self.input.spec.width,
            h: self.input.spec.height,
            lod: out.lod,
        }))]
    }
    // vips_hist_find: 1D histogram, `bins x 1`, one band per input band
    // (or a single band if `band` selects one). bins is 2^bits for the
    // input's sample depth (256 for 8-bit formats; the POC only models 8-bit
    // histograms here).
    fn output_spec(&self) -> ImageKind {
        let input = &*self.input.spec;
        let bands = match self.band {
            Some(_) => 1,
            None => (input.layout.channel_count() as i32).min(4),
        };
        let mut spec = input.clone();
        spec.layout = spec.layout.to_f32();
        spec.width = 256;
        spec.height = 1;
        spec.set_band_count(bands);
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.band {
            state.write_i32(v);
        }
    }
}

// ── HistEqual ─────────────────────────────────────────────────────────────────

pub struct HistogramEqualize<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub band: Option<i32>,
}
impl<B: Backend> Operation<B> for HistogramEqualize<B>
where
    HistogramEqualize<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.band {
            state.write_i32(v);
        }
    }
}

// ── HistCum (image-level) ─────────────────────────────────────────────────────

pub struct HistogramCumulative<B: Backend> {
    pub input: Input<ImageKind, B>,
}
impl<B: Backend> Operation<B> for HistogramCumulative<B>
where
    HistogramCumulative<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── HistNorm (image-level) ────────────────────────────────────────────────────

pub struct HistogramNormalize<B: Backend> {
    pub input: Input<ImageKind, B>,
}
impl<B: Backend> Operation<B> for HistogramNormalize<B>
where
    HistogramNormalize<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── HistPlot ──────────────────────────────────────────────────────────────────

pub struct HistogramPlot<B: Backend> {
    pub input: Input<ImageKind, B>,
}
impl<B: Backend> Operation<B> for HistogramPlot<B>
where
    HistogramPlot<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        // The plot kernel reads the *whole* histogram-image (e.g. `256 x 1`),
        // not the `width x width` output region.
        vec![Some(WorkUnit::Region(Region {
            x: 0,
            y: 0,
            w: self.input.spec.width,
            h: self.input.spec.height,
            lod: out.lod,
        }))]
    }
    // vips_hist_plot renders a histogram image (e.g. `256 x 1`) into a square
    // chart, `width x width`, preserving the band count.
    fn output_spec(&self) -> ImageKind {
        let input = &*self.input.spec;
        ImageKind {
            width: input.width,
            height: input.width,
            layout: input.layout,
        }
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── HistFindIndexed ───────────────────────────────────────────────────────────

pub struct HistFindIndexed<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub index: Input<ImageKind, B>,
    pub combine: Option<CombineMode>,
}
impl<B: Backend> Operation<B> for HistFindIndexed<B>
where
    HistFindIndexed<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.index]
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
                w: self.index.spec.width,
                h: self.index.spec.height,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.combine {
            state.write_i32(v.into_vips());
        }
    }
}

// ── HistFindNdim ──────────────────────────────────────────────────────────────

pub struct HistFindNdim<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub bins: Option<i32>,
}
impl<B: Backend> Operation<B> for HistFindNdim<B>
where
    HistFindNdim<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(Region {
            x: 0,
            y: 0,
            w: self.input.spec.width,
            h: self.input.spec.height,
            lod: out.lod,
        }))]
    }
    // vips_hist_find_ndim: N-dimensional histogram (N = input band count,
    // default bins = 10), flattened to 2D. For 2 bands: `bins x bins`. For
    // higher band counts vips flattens the extra dims into height
    // (`bins x bins^(N-1)`); only the 2-band case is handled precisely here.
    fn output_spec(&self) -> ImageKind {
        let input = &*self.input.spec;
        let bins = self.bins.unwrap_or(10);
        let bands = input.layout.channel_count() as i32;
        // TODO: for bands > 2, vips flattens the extra dimensions into
        // height (bins^(bands-1)); this only covers the bands == 2 case
        // precisely (and bands == 1, where height collapses to 1).
        let height = if bands <= 1 {
            1
        } else {
            bins.pow((bands - 1) as u32)
        };
        let mut spec = input.clone();
        spec.width = bins;
        spec.height = height;
        spec.set_band_count(1);
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.bins {
            state.write_i32(v);
        }
    }
}

// ── HistLocal ─────────────────────────────────────────────────────────────────

pub struct HistLocal<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub width: i32,
    pub height: i32,
    pub max_slope: Option<i32>,
}
impl<B: Backend> Operation<B> for HistLocal<B>
where
    HistLocal<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height);
        if let Some(v) = self.max_slope {
            state.write_i32(v);
        }
    }
}

// ── HistMatch ─────────────────────────────────────────────────────────────────

pub struct HistMatch<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub ref_image: Input<ImageKind, B>,
}
impl<B: Backend> Operation<B> for HistMatch<B>
where
    HistMatch<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.ref_image]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.ref_image.spec.width,
                h: self.ref_image.spec.height,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
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
impl<B: Backend> Operation<B> for Stdif<B>
where
    Stdif<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height);
        if let Some(v) = self.a {
            state.write_u64(v.to_bits());
        }
        if let Some(v) = self.m0 {
            state.write_u64(v.to_bits());
        }
        if let Some(v) = self.b {
            state.write_u64(v.to_bits());
        }
        if let Some(v) = self.s0 {
            state.write_u64(v.to_bits());
        }
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistogramFind<B>: crate::operation::Lower<B>,
{
    pub fn histogram_find(&self, band: Option<i32>) -> Self {
        self.push(HistogramFind {
            input: self.as_input(),
            band,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistogramEqualize<B>: crate::operation::Lower<B>,
{
    pub fn histogram_equalize(&self, band: Option<i32>) -> Self {
        self.push(HistogramEqualize {
            input: self.as_input(),
            band,
        })
    }
}


impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistogramCumulative<B>: crate::operation::Lower<B>,
{
    pub fn histogram_cumulative(&self) -> Self {
        self.push(HistogramCumulative {
            input: self.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistogramNormalize<B>: crate::operation::Lower<B>,
{
    pub fn histogram_normalize(&self) -> Self {
        self.push(HistogramNormalize {
            input: self.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistogramPlot<B>: crate::operation::Lower<B>,
{
    pub fn histogram_plot(&self) -> Self {
        self.push(HistogramPlot {
            input: self.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistFindIndexed<B>: crate::operation::Lower<B>,
{
    pub fn hist_find_indexed(
        &self,
        index: Input<ImageKind, B>,
        combine: Option<CombineMode>,
    ) -> Self {
        self.push(HistFindIndexed {
            input: self.as_input(),
            index,
            combine,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistFindNdim<B>: crate::operation::Lower<B>,
{
    pub fn hist_find_ndim(&self, bins: Option<i32>) -> Self {
        self.push(HistFindNdim {
            input: self.as_input(),
            bins,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistLocal<B>: crate::operation::Lower<B>,
{
    pub fn hist_local(&self, width: i32, height: i32, max_slope: Option<i32>) -> Self {
        self.push(HistLocal {
            input: self.as_input(),
            width,
            height,
            max_slope,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    HistMatch<B>: crate::operation::Lower<B>,
{
    pub fn hist_match(&self, ref_image: Input<ImageKind, B>) -> Self {
        self.push(HistMatch {
            input: self.as_input(),
            ref_image,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Stdif<B>: crate::operation::Lower<B>,
{
    pub fn stdif(
        &self,
        width: i32,
        height: i32,
        a: Option<f64>,
        m0: Option<f64>,
        b: Option<f64>,
        s0: Option<f64>,
    ) -> Self {
        self.push(Stdif {
            input: self.as_input(),
            width,
            height,
            a,
            m0,
            b,
            s0,
        })
    }
}
