use std::hash::Hasher;
use std::sync::Arc;
use crate::kind::{AnyKind, Kind};
use crate::node::Node;
use crate::work_unit::{WorkUnit, WorkUnitFor};
use crate::backend::Backend;
use crate::backend::vips::IntoVipsEnum;

// -- Core operation traits --

/// Object-safe input edge.
pub trait AnyInput<B: Backend>: Send + Sync + 'static {
    fn src(&self) -> &Arc<Node<B>>;
    fn spec(&self) -> &dyn AnyKind;
}

/// A node edge that owns its typed input specification.
pub struct Input<K: Kind, B: Backend> {
    pub src: Arc<Node<B>>,
    pub spec: Arc<K>,
}

impl<K: Kind, B: Backend> AnyInput<B> for Input<K, B> {
    fn src(&self) -> &Arc<Node<B>> {
        &self.src
    }
    fn spec(&self) -> &dyn AnyKind {
        self.spec.as_ref()
    }
}

pub trait AnyOperation<B: Backend>: Send + Sync + 'static {
    fn inputs(&self) -> Vec<&dyn AnyInput<B>>;
    fn demand_erased(&self, out: &WorkUnit) -> Vec<Option<WorkUnit>>;
    fn output_kind(&self) -> Arc<dyn AnyKind>;
    fn lower(&self, cx: &mut B::Builder);
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

impl<B: Backend, T: Operation<B>> AnyOperation<B> for T {
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        Operation::inputs(self)
    }

    fn demand_erased(&self, out: &WorkUnit) -> Vec<Option<WorkUnit>> {
        let typed = <<T as Operation<B>>::Output as Kind>::WorkUnit::typed(out)
            .expect("work unit shape mismatch for operation output");
        self.demand(&typed)
    }

    fn output_kind(&self) -> Arc<dyn AnyKind> {
        Arc::new(self.output_spec())
    }

    fn lower(&self, cx: &mut B::Builder) {
        Lower::lower(self, cx)
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        Operation::dyn_hash(self, state);
        self.output_spec().dyn_hash(state);
    }
}

pub trait Operation<B: Backend>: Lower<B> + 'static + Send + Sync {
    type Output: Kind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>>;
    fn demand(&self, out: &<Self::Output as Kind>::WorkUnit) -> Vec<Option<WorkUnit>>;
    fn output_spec(&self) -> Self::Output;
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

pub trait Lower<B: Backend> {
    fn lower(&self, cx: &mut B::Builder);
}

// -- Arithmetic operation enums --
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationMath { Sin, Cos, Tan, Asin, Acos, Atan, Log, Log10, Exp, Exp10, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh }
impl IntoVipsEnum for OperationMath { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationBoolean { And, Or, Eor, Lshift, Rshift }
impl IntoVipsEnum for OperationBoolean { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationRelational { Equal, Notequal, Less, Lesseq, More, Moreeq }
impl IntoVipsEnum for OperationRelational { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationRound { Rint, Ceil, Floor }
impl IntoVipsEnum for OperationRound { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationComplex { Polar, Rect, Conj }
impl IntoVipsEnum for OperationComplex { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationComplex2 { CrossPhase }
impl IntoVipsEnum for OperationComplex2 { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationComplexget { Real, Imag }
impl IntoVipsEnum for OperationComplexget { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationMath2 { Pow, Wop, Atan2 }
impl IntoVipsEnum for OperationMath2 { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SdfShape { Circle, Box, RoundedBox, Line }
impl IntoVipsEnum for SdfShape { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextWrap { Word, Char, WordChar, NoWrap }
impl IntoVipsEnum for TextWrap { fn into_vips(self) -> i32 { self as i32 } }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationMorphology { Erode, Dilate }
impl IntoVipsEnum for OperationMorphology { fn into_vips(self) -> i32 { self as i32 } }

// -- Sub-modules --
pub mod arithmetic;
pub mod filters;
pub mod custom_ops;
pub mod edge;
pub mod composite;
pub mod geometry;

pub use arithmetic::*;
pub use filters::*;
pub use custom_ops::*;
pub use edge::*;
pub use composite::*;
pub use geometry::*;
pub mod bands;
pub use bands::*;
pub mod convolution;
pub use convolution::*;
pub mod fft;
pub use fft::*;
pub mod icc;
pub use icc::*;
pub mod misc;
pub use misc::*;
pub mod opacity;
pub use opacity::*;
pub mod mosaicing;
pub use mosaicing::*;
pub mod stats;
pub use stats::*;
