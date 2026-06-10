//! Typed host-facing wrappers and payload buffers — as opposed to
//! [`super::datatype`], which holds the graph-node datatype *tags*
//! (`ImageType`, `HistogramType`, …).

pub mod histogram;
pub mod image;

pub use histogram::*;
