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
pub mod color;
pub mod data;
pub mod error;
pub mod generator;
pub mod io;
pub mod kind;
pub mod node;
pub mod operation;
pub mod pixel;
pub mod stage;
pub mod stage_cache;
pub mod work_unit;

pub use backend::*;
pub use buffer::*;
pub use color::*;
pub use data::*;
pub use error::*;
pub use generator::*;
pub use io::*;
pub use kind::*;
pub use node::*;
pub use operation::*;
pub use stage::*;
pub use stage_cache::{
    CacheExt, CacheKey, CacheStats, Cached, DEFAULT_BUDGET, RegionCache, StageExt,
};
pub use work_unit::*;
