use std::hash::Hasher;
use crate::operation::IntoVipsEnum;

use crate::backend::Backend;
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation, OperationMorphology};
use crate::work_unit::{Region, WorkUnit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    Integer,
    Float,
    Approximate,
}
impl IntoVipsEnum for Precision {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

// ── Convolution ───────────────────────────────────────────────────────────────

pub struct Convolution<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<crate::data::mask2d::Mask2DKind, B>,
    pub precision: Option<Precision>,
    pub layers: Option<i32>,
    pub cluster: Option<i32>,
}

impl<B: Backend> Operation<B> for Convolution<B>
where
    Convolution<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.mask]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.precision {
            state.write_i32(v.into_vips());
        }
        if let Some(v) = self.layers {
            state.write_i32(v);
        }
        if let Some(v) = self.cluster {
            state.write_i32(v);
        }
    }
}

// ── Compass ───────────────────────────────────────────────────────────────────

pub struct Compass<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<crate::data::mask2d::Mask2DKind, B>,
}

impl<B: Backend> Operation<B> for Compass<B>
where
    Compass<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.mask]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── Morph ─────────────────────────────────────────────────────────────────────

pub struct Morph<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<crate::data::mask2d::Mask2DKind, B>,
    pub morph: OperationMorphology,
}

impl<B: Backend> Operation<B> for Morph<B>
where
    Morph<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.mask]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.morph.into_vips());
    }
}

// ── Conva ─────────────────────────────────────────────────────────────────────

pub struct Conva<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<crate::data::mask2d::Mask2DKind, B>,
    pub layers: Option<i32>,
    pub cluster: Option<i32>,
}

impl<B: Backend> Operation<B> for Conva<B>
where
    Conva<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.mask]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.layers {
            state.write_i32(v);
        }
        if let Some(v) = self.cluster {
            state.write_i32(v);
        }
    }
}

// ── Convf ─────────────────────────────────────────────────────────────────────

pub struct Convf<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<crate::data::mask2d::Mask2DKind, B>,
}

impl<B: Backend> Operation<B> for Convf<B>
where
    Convf<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.mask]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── Convi ─────────────────────────────────────────────────────────────────────

pub struct Convi<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<crate::data::mask2d::Mask2DKind, B>,
}

impl<B: Backend> Operation<B> for Convi<B>
where
    Convi<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.mask]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── Convsep ───────────────────────────────────────────────────────────────────

pub struct Convsep<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<crate::data::mask2d::Mask2DKind, B>,
    pub precision: Option<Precision>,
    pub layers: Option<i32>,
    pub cluster: Option<i32>,
}

impl<B: Backend> Operation<B> for Convsep<B>
where
    Convsep<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.mask]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.precision {
            state.write_i32(v.into_vips());
        }
        if let Some(v) = self.layers {
            state.write_i32(v);
        }
        if let Some(v) = self.cluster {
            state.write_i32(v);
        }
    }
}

// ── Convasep ──────────────────────────────────────────────────────────────────

pub struct Convasep<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<crate::data::mask2d::Mask2DKind, B>,
    pub layers: Option<i32>,
}

impl<B: Backend> Operation<B> for Convasep<B>
where
    Convasep<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.mask]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.layers {
            state.write_i32(v);
        }
    }
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Convolution<B>: crate::operation::Lower<B>,
{
    pub fn convolution(
        &self,
        mask: &crate::data::mask2d::Mask2D<B>,
        precision: Option<Precision>,
        layers: Option<i32>,
        cluster: Option<i32>,
    ) -> Self {
        self.push(Convolution {
            input: self.as_input(),
            mask: mask.as_input(),
            precision,
            layers,
            cluster,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Compass<B>: crate::operation::Lower<B>,
{
    pub fn compass(&self, mask: &crate::data::mask2d::Mask2D<B>) -> Self {
        self.push(Compass {
            input: self.as_input(),
            mask: mask.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Morph<B>: crate::operation::Lower<B>,
{
    pub fn morph(&self, mask: &crate::data::mask2d::Mask2D<B>, morph: OperationMorphology) -> Self {
        self.push(Morph {
            input: self.as_input(),
            mask: mask.as_input(),
            morph,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Conva<B>: crate::operation::Lower<B>,
{
    pub fn conva(
        &self,
        mask: &crate::data::mask2d::Mask2D<B>,
        layers: Option<i32>,
        cluster: Option<i32>,
    ) -> Self {
        self.push(Conva {
            input: self.as_input(),
            mask: mask.as_input(),
            layers,
            cluster,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Convf<B>: crate::operation::Lower<B>,
{
    pub fn convf(&self, mask: &crate::data::mask2d::Mask2D<B>) -> Self {
        self.push(Convf {
            input: self.as_input(),
            mask: mask.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Convi<B>: crate::operation::Lower<B>,
{
    pub fn convi(&self, mask: &crate::data::mask2d::Mask2D<B>) -> Self {
        self.push(Convi {
            input: self.as_input(),
            mask: mask.as_input(),
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Convsep<B>: crate::operation::Lower<B>,
{
    pub fn convsep(
        &self,
        mask: &crate::data::mask2d::Mask2D<B>,
        precision: Option<Precision>,
        layers: Option<i32>,
        cluster: Option<i32>,
    ) -> Self {
        self.push(Convsep {
            input: self.as_input(),
            mask: mask.as_input(),
            precision,
            layers,
            cluster,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Convasep<B>: crate::operation::Lower<B>,
{
    pub fn convasep(&self, mask: &crate::data::mask2d::Mask2D<B>, layers: Option<i32>) -> Self {
        self.push(Convasep {
            input: self.as_input(),
            mask: mask.as_input(),
            layers,
        })
    }
}
