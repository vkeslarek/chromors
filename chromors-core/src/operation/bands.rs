use crate::operation::IntoVipsEnum;
use std::hash::Hasher;

use crate::backend::Backend;
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation, OperationBoolean};
use crate::work_unit::{Region, WorkUnit};

// ── Boolean ───────────────────────────────────────────────────────────────────

pub struct Bandbool<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub boolean: OperationBoolean,
    pub bands: u32,
}

impl<B: Backend> Operation<B> for Bandbool<B>
where
    Bandbool<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.set_band_count(1);
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.boolean.into_vips());
        state.write_u32(self.bands);
    }
}

// ── Bandfold ──────────────────────────────────────────────────────────────────

pub struct Bandfold<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub factor: u32,
}

impl<B: Backend> Operation<B> for Bandfold<B>
where
    Bandfold<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let f = self.factor as i32;
        vec![Some(WorkUnit::Region(Region {
            x: out.x * f,
            y: out.y,
            w: out.w * f,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        let bands = spec.layout.channel_count() as i32;
        spec.width /= self.factor as i32;
        spec.set_band_count(bands * self.factor as i32);
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.factor);
    }
}

// ── Bandunfold ────────────────────────────────────────────────────────────────

pub struct Bandunfold<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub factor: u32,
}

impl<B: Backend> Operation<B> for Bandunfold<B>
where
    Bandunfold<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let f = self.factor as i32;
        let x = out.x / f;
        let w = ((out.x + out.w + f - 1) / f) - x;
        vec![Some(WorkUnit::Region(Region {
            x,
            y: out.y,
            w,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        let bands = spec.layout.channel_count() as i32;
        spec.width *= self.factor as i32;
        spec.set_band_count(bands / self.factor as i32);
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.factor);
    }
}

// ── Bandmean ──────────────────────────────────────────────────────────────────

pub struct Bandmean<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub bands: u32,
}

impl<B: Backend> Operation<B> for Bandmean<B>
where
    Bandmean<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.set_band_count(1);
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.bands);
    }
}

// ── ExtractBand ───────────────────────────────────────────────────────────────

pub struct ExtractBand<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub band: i32,
    pub count: Option<i32>,
}

impl<B: Backend> Operation<B> for ExtractBand<B>
where
    ExtractBand<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.set_band_count(self.count.unwrap_or(1));
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.band);
        if let Some(c) = self.count {
            state.write_i32(c);
        }
    }
}

// ── Bandjoin ──────────────────────────────────────────────────────────────────

pub struct Bandjoin<B: Backend> {
    pub images: Vec<Input<ImageKind, B>>,
}

impl<B: Backend> Operation<B> for Bandjoin<B>
where
    Bandjoin<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        self.images.iter().map(|i| i as &dyn AnyInput<B>).collect()
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); self.images.len()]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.images[0].spec).clone();
        spec.set_band_count(self.images.len() as i32);
        spec
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Bandbool<B>: crate::operation::Lower<B>,
{
    pub fn bandbool(&self, boolean: OperationBoolean, bands: u32) -> Self {
        self.push(Bandbool {
            input: self.as_input(),
            boolean,
            bands,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Bandfold<B>: crate::operation::Lower<B>,
{
    pub fn bandfold(&self, factor: u32) -> Self {
        self.push(Bandfold {
            input: self.as_input(),
            factor,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Bandunfold<B>: crate::operation::Lower<B>,
{
    pub fn bandunfold(&self, factor: u32) -> Self {
        self.push(Bandunfold {
            input: self.as_input(),
            factor,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Bandmean<B>: crate::operation::Lower<B>,
{
    pub fn bandmean(&self, bands: u32) -> Self {
        self.push(Bandmean {
            input: self.as_input(),
            bands,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    ExtractBand<B>: crate::operation::Lower<B>,
{
    pub fn extract_band(&self, band: i32, count: Option<i32>) -> Self {
        self.push(ExtractBand {
            input: self.as_input(),
            band,
            count,
        })
    }
}
