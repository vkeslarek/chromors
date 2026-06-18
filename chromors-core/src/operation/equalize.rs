use std::hash::Hasher;

use crate::backend::Backend;
use crate::data::histogram::{Histogram, HistogramKind};
use crate::data::lut::LutKind;
use crate::operation::{AnyInput, Input, Operation};
use crate::work_unit::{Atomic, Range, WorkUnit};

pub struct EqualizeLut<B: Backend> {
    pub histogram: Input<HistogramKind, B>,
}

impl<B: Backend> Operation<B> for EqualizeLut<B> 
where 
    Self: crate::operation::Lower<B>,
{
    type Output = LutKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.histogram]
    }
    fn demand(&self, _out: &Range) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Atomic)]
    }
    fn output_spec(&self) -> LutKind {
        LutKind::new(self.histogram.spec.bins, self.histogram.spec.bands.max(1))
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct HistogramCumulative<B: Backend> {
    pub histogram: Input<HistogramKind, B>,
}

impl<B: Backend> Operation<B> for HistogramCumulative<B> 
where 
    Self: crate::operation::Lower<B>,
{
    type Output = HistogramKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.histogram]
    }
    fn demand(&self, _out: &Atomic) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Atomic)]
    }
    fn output_spec(&self) -> HistogramKind {
        (*self.histogram.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

pub struct HistogramNormalize<B: Backend> {
    pub histogram: Input<HistogramKind, B>,
}

impl<B: Backend> Operation<B> for HistogramNormalize<B> 
where 
    Self: crate::operation::Lower<B>,
{
    type Output = HistogramKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.histogram]
    }
    fn demand(&self, _out: &Atomic) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Atomic)]
    }
    fn output_spec(&self) -> HistogramKind {
        (*self.histogram.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl<B: Backend> Histogram<B> 
where
    EqualizeLut<B>: crate::operation::Lower<B>,
    HistogramCumulative<B>: crate::operation::Lower<B>,
    HistogramNormalize<B>: crate::operation::Lower<B>,
{
    pub fn equalize_lut(&self) -> crate::data::lut::Lut<B> {
        self.push(EqualizeLut {
            histogram: self.as_input(),
        })
    }

    pub fn cumulative(&self) -> Histogram<B> {
        self.push(HistogramCumulative {
            histogram: self.as_input(),
        })
    }

    pub fn normalize(&self) -> Histogram<B> {
        self.push(HistogramNormalize {
            histogram: self.as_input(),
        })
    }
}

// NOTE: The `Image2D::equalize` method requires `self.histogram(bins, channel)` which comes from `GpuImageExt` or `VipsImageExt` in the old codebase, but `Image2D::histogram` does not exist generically in core.
// Wait! I'll put `equalize` in core but requiring a trait. Or I'll just leave `Image2D::equalize` in the wgpu prelude where `GpuImageExt` is.
