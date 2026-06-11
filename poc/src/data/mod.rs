//! Concrete datatypes. Each module is a self-contained datatype: its `*Kind`
//! metadata, per-backend lowering capabilities, operations, and the
//! user-facing alias + ergonomic methods. Adding a datatype = adding a module
//! here; nothing central is edited.

pub mod image;
pub mod histogram;
pub mod vectorscope;

pub use image::{Image2D, ImageKind};
pub use histogram::{Histogram, HistogramKind};
pub use vectorscope::{Vectorscope, VectorscopeKind};
