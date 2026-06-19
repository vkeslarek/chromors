pub use chromors_core::color::*;
pub use chromors_core::pixel::*;
pub use chromors_core::*;

#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    warnings,
    clippy::all,
    unnecessary_transmutes,
    unsafe_op_in_unsafe_fn
)]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/ffi.rs"));
}

// Vips backend types (VipsBackend, VipsBuilder, VipsHandle, VipsBand, etc.)
mod backend;
pub use backend::*;

// Vips infrastructure (formerly backend/vips/)
pub mod custom;
pub mod generator;
pub mod generator_rng;
pub mod gobject;
pub mod interpolate;
pub mod region;
pub mod sbuf;
pub mod source;
pub mod target;
pub mod working;

// Vips-specific mappings (formerly color/ and pixel/ subdirs)
pub mod space;
pub mod storage;

// Stage boundary (Source<VipsBackend> for BoundarySource)
pub mod stage;

// Data, operations, prelude
pub mod data;
pub mod operation;
pub mod prelude;
