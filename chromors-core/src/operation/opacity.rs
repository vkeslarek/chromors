use std::hash::Hasher;

use crate::backend::Backend;
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

pub struct Opacity<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub amount: f32,
}

impl<B: Backend> Operation<B> for Opacity<B>
where
    Opacity<B>: Lower<B>,
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
        // If it didn't have alpha, it now does (same codec, +1 band).
        let channels = spec.layout.channel_count();
        if channels == 1 || channels == 3 {
            spec.set_band_count(channels as i32 + 1);
        }
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.amount.to_bits());
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Opacity<B>: crate::operation::Lower<B>,
{
    pub fn opacity(&self, amount: f32) -> Self {
        self.push(Opacity {
            input: self.as_input(),
            amount,
        })
    }
}
