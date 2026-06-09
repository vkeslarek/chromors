pub mod backend;
pub mod color;
pub mod data;
pub mod error;
mod exif;
pub mod export;
pub mod generator;
pub mod geometry;
pub mod operation;
pub mod pixel;
pub mod target;
#[macro_use]
pub mod utils;
pub mod vector;

pub use backend::gpu::Rect;
pub use backend::gpu::{
    GpuBackend, GpuContext, GpuOperation, GpuSource, GpuTarget, Lod, OutputSpec,
};
pub use backend::vips::data::{ArrayJoinParams, CompositeParams, ThumbnailParams};
pub use backend::vips::{
    Interpolate, InterpolationMethod, Region, Sbuf, Source, Target, VipsBackend,
};
pub use backend::{SourceInput, TargetOutput};
pub use color::space::ColorSpace;
pub use draw::*;
pub use error::Error;
pub use exif::Metadata;
pub use generator::*;
pub use operation::*;
pub use pixel::{AlphaPolicy, PixelFormat, PixelMeta};

pub mod libvips_ffi {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(clippy::all)]
    #![allow(clippy::approx_constant)]
    #![allow(clippy::missing_safety_doc)]
    #![allow(unnecessary_transmutes)]
    #![allow(unsafe_op_in_unsafe_fn)]
    include!(concat!(env!("OUT_DIR"), "/ffi.rs"));
}

pub mod slang_ffi {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(clippy::all)]
    #![allow(unsafe_op_in_unsafe_fn)]
    include!(concat!(env!("OUT_DIR"), "/slang_ffi.rs"));
}

pub(crate) mod slang_wrapper_ffi {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(warnings)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(clippy::all)]
    #![allow(unnecessary_transmutes)]
    #![allow(unsafe_op_in_unsafe_fn)]
    include!(concat!(env!("OUT_DIR"), "/slang_wrapper_ffi.rs"));
}

pub(crate) mod libraw_ffi {
    #![allow(warnings)]
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    #![allow(clippy::all)]
    #![allow(unnecessary_transmutes)]
    #![allow(unsafe_op_in_unsafe_fn)]
    include!(concat!(env!("OUT_DIR"), "/libraw_ffi.rs"));
}

use std::ffi::CString;
use std::sync::OnceLock;

static INIT: OnceLock<()> = OnceLock::new();

pub fn init() {
    INIT.get_or_init(|| {
        let name = CString::new("pixors-engine").unwrap();
        unsafe {
            libvips_ffi::vips_init(name.as_ptr());
        }
    });
}
