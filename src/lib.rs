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

pub use chromors_core::pixel::*;
pub use chromors_core::*;

// Compatibility re-export modules so existing code using `chromors::backend::gpu::*`
// or `chromors::backend::vips::*` continues to compile after the backend crates were flattened.
pub mod backend {
    pub use chromors_core::backend::*;
    pub mod gpu {
        pub use chromors_backend_wgpu::*;
        pub mod context {
            pub use chromors_backend_wgpu::context::*;
        }
        pub mod pass {
            pub use chromors_backend_wgpu::pass::*;
        }
    }
    pub mod vips {
        pub use chromors_backend_vips::*;
    }
}

pub use chromors_backend_vips::data::vips_image::VipsImageExt;
pub use chromors_backend_vips::data::vips_lut::VipsLutExt;
pub use chromors_backend_vips::data::vips_mask2d::VipsMask2DExt;
pub use chromors_backend_wgpu::CacheExt;
pub use chromors_backend_wgpu::StageExt;
pub use chromors_backend_wgpu::data::gpu_lut::GpuLutExt;
pub use chromors_backend_wgpu::data::gpu_mask2d::GpuMask2DExt;
pub use chromors_backend_wgpu::data::histogram::GpuImageExt;

pub mod export;

pub mod data {
    pub mod histogram;
    pub mod image;
    pub mod mask2d {
        pub use chromors_core::data::mask2d::*;
    }
    pub mod lut {
        pub use chromors_core::data::lut::*;
    }
}
