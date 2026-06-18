//! Gamma (TRC exponent) operation.
//!
//! ICC color management is **not** done here. Profile classification happens at
//! load (`FileImageSource::new` → `IccClassification::classify_icc_profile`,
//! which tags the image's `PixelLayout::color_space`), and all color-space
//! conversion is the native `Convert` operation (`operation::color`,
//! `docs/native-color-management.md`) applied as a real Slang/CPU op — never a
//! libvips `icc_import`/`icc_export`/`icc_transform` wrapper. Those vips ICC ops
//! were removed: every backend now obeys the native color pipeline.

use std::hash::Hasher;

use crate::backend::Backend;
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

// ── Gamma ─────────────────────────────────────────────────────────────────────

pub struct Gamma<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub exponent: Option<f64>,
}

impl<B: Backend> Operation<B> for Gamma<B>
where
    Gamma<B>: Lower<B>,
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
        if let Some(v) = self.exponent {
            state.write(&v.to_ne_bytes());
        }
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Gamma<B>: crate::operation::Lower<B>,
{
    pub fn gamma(&self, exponent: Option<f64>) -> Self {
        self.push(Gamma {
            input: self.as_input(),
            exponent,
        })
    }
}
