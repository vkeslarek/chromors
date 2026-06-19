pub mod prelude;
pub use chromors_core::color::*;
pub use chromors_core::pixel::*;
pub use chromors_core::*;

// GPU backend modules (formerly backend/gpu/)
pub mod buffer;
pub mod color_params;
pub mod compile;
pub mod context;
pub mod emit;
pub mod generator;
pub mod gpu;
pub mod pass;
pub mod slang;
pub mod view;

// Re-export GPU backend types at crate root
pub use buffer::*;
pub use context::*;
pub use gpu::*;
pub use view::*;

pub mod data;
pub mod operation;
pub mod stage;
pub mod stage_cache;
pub mod stage_ext;

pub use data::gpu_image::*;
pub use stage_cache::{CacheExt, StageExt};

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
pub mod slang_wrapper_ffi {
    include!(concat!(env!("OUT_DIR"), "/slang_wrapper_ffi.rs"));
}

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
pub mod slang_ffi {
    include!(concat!(env!("OUT_DIR"), "/slang_ffi.rs"));
}
