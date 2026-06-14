/// Assert two floats are within `eps` (default `1e-6`). Used by `color`/`pixel`
/// unit tests; defined once here so every submodule's `use crate::assert_approx_eq`
/// resolves.
#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
        $crate::assert_approx_eq!($a, $b, 1e-6)
    };
    ($a:expr, $b:expr, $eps:expr) => {{
        let (a, b, eps) = ($a as f64, $b as f64, $eps as f64);
        assert!(
            (a - b).abs() <= eps,
            "assert_approx_eq failed: `{}` vs `{}` (|Δ| = {} > {})",
            a,
            b,
            (a - b).abs(),
            eps
        );
    }};
}

pub mod backend;
pub mod buffer;
pub mod cache;
pub mod data;
pub mod error;
pub mod io;
pub mod kind;
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

pub mod color;
pub mod export;
pub mod node;
pub mod operation;
pub mod pixel;
pub mod work_unit;

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/ffi.rs"));
}

pub use backend::gpu::*;
pub use backend::*;
pub use buffer::*;
pub use error::*;
pub use io::*;
pub use kind::*;
pub use node::*;
pub use operation::*;
pub use work_unit::*;
