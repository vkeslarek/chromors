use crate::backend::vips::IntoVipsEnum;

// -- Arithmetic operation enums --
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationMath {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Log,
    Log10,
    Exp,
    Exp10,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
}
impl IntoVipsEnum for OperationMath {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationBoolean {
    And,
    Or,
    Eor,
    Lshift,
    Rshift,
}
impl IntoVipsEnum for OperationBoolean {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationRelational {
    Equal,
    Notequal,
    Less,
    Lesseq,
    More,
    Moreeq,
}
impl IntoVipsEnum for OperationRelational {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationRound {
    Rint,
    Ceil,
    Floor,
}
impl IntoVipsEnum for OperationRound {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationComplex {
    Polar,
    Rect,
    Conj,
}
impl IntoVipsEnum for OperationComplex {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationComplex2 {
    CrossPhase,
}
impl IntoVipsEnum for OperationComplex2 {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationComplexget {
    Real,
    Imag,
}
impl IntoVipsEnum for OperationComplexget {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationMath2 {
    Pow,
    Wop,
    Atan2,
}
impl IntoVipsEnum for OperationMath2 {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SdfShape {
    Circle,
    Box,
    RoundedBox,
    Line,
}
impl IntoVipsEnum for SdfShape {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextWrap {
    Word,
    Char,
    WordChar,
    NoWrap,
}
impl IntoVipsEnum for TextWrap {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationMorphology {
    Erode,
    Dilate,
}
impl IntoVipsEnum for OperationMorphology {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

// -- Sub-modules --
pub mod arithmetic;
pub mod bands;
pub mod composite;
pub mod convolution;
pub mod custom_ops;
pub mod edge;
pub mod fft;
pub mod filters;
pub mod geometry;
pub mod icc;
pub mod misc;
pub mod mosaicing;
pub mod opacity;
pub mod stats;

pub mod draw;
pub mod raw;

pub use arithmetic::*;
pub use bands::*;
pub use composite::*;
pub use convolution::*;
pub use custom_ops::{
    Checkerboard, Custom, Histogram, HistogramSink, Invert, Reduce, VECTORSCOPE_GRID,
    VectorscopeSink, vectorscope_from_rgba8,
};
pub use edge::*;
pub use fft::*;
pub use filters::*;
pub use geometry::*;
pub use icc::*;
pub use misc::*;
pub use mosaicing::*;
pub use opacity::*;
pub use stats::*;

pub use crate::backend::vips::operation::VipsOperation;
