//! Concrete datatypes. Each module is a self-contained datatype: its `*Kind`
//! metadata, per-backend lowering capabilities, operations, and the
//! user-facing alias + ergonomic methods. Adding a datatype = adding a module
//! here; nothing central is edited.

pub mod fft2d;
pub mod histogram;
pub mod image;
pub mod lut;
pub mod mask2d;
pub mod vector_graphics;
pub mod vectorscope;

pub use fft2d::{Fft2D, Fft2DKind};
pub use histogram::{Histogram, HistogramKind};
pub use image::{Image2D, ImageKind};
pub use lut::{Lut, LutKind};
pub use mask2d::{Mask2D, Mask2DKind};
pub use vector_graphics::{VectorGraphics, VectorGraphicsKind};
pub use vectorscope::{Vectorscope, VectorscopeKind};
