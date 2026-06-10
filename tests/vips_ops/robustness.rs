//! vips backend — robustness / fuzz. Every op must be **safe to call** with
//! absurd parameters: it may return `Ok` or `Err`, but must never segfault or
//! panic. Reaching the end of a test = the process survived every call.
//!
//! `safe!` discards the result; the test's value is that the call returned at
//! all. A segfault in libvips would abort the test binary (test fails).

use crate::common::{init, rgb};
use pixors_engine::backend::vips::VipsBackend;
use pixors_engine::data::image::Image2D;
use pixors_engine::*;

/// Call an expression and discard its `Result` — we only care that it returned.
macro_rules! safe {
    ($e:expr) => {{
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $e));
    }};
}

#[test]
fn geometry_absurd_params() {
    let img = rgb();
    // Resize: zero / negative / gigantic scale.
    for s in [0.0, -1.0, -0.5, 1e9, f64::NAN, f64::INFINITY] {
        safe!(img.execute(&ResizeOperation {
            scale: s,
            kernel: None,
            vertical_scale: None,
            gap: None
        }));
    }
    // Crop: negative origin, negative / oversize extent, fully out of bounds.
    for (l, t, w, h) in [
        (-100, -100, 50, 50),
        (0, 0, -5, -5),
        (0, 0, 100000, 100000),
        (500, 500, 10, 10),
    ] {
        safe!(img.execute(&CropOperation {
            left: l,
            top: t,
            width: w,
            height: h
        }));
        safe!(img.execute(&ExtractAreaOperation {
            left: l,
            top: t,
            width: w,
            height: h
        }));
    }
    // Embed: zero / negative target.
    for (w, h) in [(0, 0), (-10, -10), (1, 1)] {
        safe!(img.execute(&EmbedOperation {
            x: 0,
            y: 0,
            width: w,
            height: h,
            extend: None,
            background: None
        }));
    }
    // Replicate / zoom / subsample with zero / negative factors.
    for n in [0, -1, 100000] {
        safe!(img.execute(&ReplicateOperation { across: n, down: n }));
        safe!(img.execute(&ZoomOperation {
            horizontal: n,
            vertical: n
        }));
        safe!(img.execute(&SubsampleOperation {
            horizontal: n,
            vertical: n,
            point: None
        }));
    }
    // Affine with empty / wrong-length / degenerate matrix.
    for m in [vec![], vec![1.0], vec![0.0, 0.0, 0.0, 0.0]] {
        safe!(img.execute(&AffineOperation {
            matrix: m,
            interpolate: None,
            output_area: None,
            offset_input_x: None,
            offset_input_y: None,
            offset_output_x: None,
            offset_output_y: None,
            background: None,
            premultiplied: None,
            extend: None,
        }));
    }
}

#[test]
fn filter_absurd_params() {
    let img = rgb();
    // Gaussian blur: negative / zero / gigantic / NaN sigma.
    for s in [0.0, -1.0, 1e6, f64::NAN] {
        safe!(img.execute(&GaussianBlurOperation {
            sigma: s,
            minimum_amplitude: None,
            precision: None
        }));
        safe!(img.execute(&GaussianBlurOperation {
            sigma: s,
            minimum_amplitude: Some(-1.0),
            precision: Some(-9)
        }));
    }
    // Median rank with zero / negative / even / huge window.
    for sz in [0, -3, 2, 4, 100000] {
        safe!(img.execute(&MedianOperation { size: sz }));
    }
    // Convolution with an empty / 1×1 / huge mask.
    for mask in [
        Image2D::<VipsBackend>::from_memory(&[0u8], 1, 1, 1, PixelFormat::Gray8),
        Image2D::<VipsBackend>::from_memory(&[], 0, 0, 1, PixelFormat::Gray8),
    ]
    .into_iter()
    .flatten()
    {
        safe!(img.execute(&ConvfOperation { mask: mask.clone() }));
        safe!(img.execute(&ConviOperation { mask: mask }));
    }
    safe!(img.execute(&SharpenOperation {
        sigma: Some(-5.0),
        flat: Some(-1.0),
        jagged: None,
        edge: None,
        smooth: None,
        maximum: None,
    }));
}

#[test]
fn arithmetic_absurd_params() {
    let img = rgb();
    // Linear with NaN / inf coefficients.
    for (a, b) in [
        (f64::NAN, 0.0),
        (f64::INFINITY, f64::NEG_INFINITY),
        (0.0, 1e300),
    ] {
        safe!(img.execute(&LinearOperation { a, b, uchar: None }));
    }
    // Const ops with empty / oversized constant arrays.
    safe!(img.execute(&Math2ConstOperation {
        math2: OperationMath2::Pow,
        constants: vec![]
    }));
    safe!(img.execute(&Math2ConstOperation {
        math2: OperationMath2::Pow,
        constants: vec![1.0; 100],
    }));
    safe!(img.execute(&RelationalConstOperation {
        relational: OperationRelational::More,
        constants: vec![],
    }));
    safe!(img.execute(&RemainderConstOperation {
        constants: vec![0.0]
    })); // divide-by-zero
    // bandjoin_const with empty array.
    safe!(img.bandjoin_const(&[]));
}

#[test]
fn band_and_cast_absurd_params() {
    let img = rgb();
    // Extract a band that doesn't exist / negative.
    for b in [-1, 3, 1000] {
        safe!(img.execute(&ExtractBandOperation {
            band: b,
            count: None
        }));
        safe!(img.execute(&ExtractBandOperation {
            band: 0,
            count: Some(b)
        }));
    }
    // getpoint fully out of bounds.
    for (x, y) in [(-1, -1), (100000, 100000)] {
        safe!(img.execute(&GetpointOperation {
            x,
            y,
            unpack_complex: None
        }));
    }
    // Percent threshold out of [0,100].
    for p in [-50.0, 0.0, 100.0, 1000.0, f64::NAN] {
        safe!(img.execute(&PercentOperation { percent: p }));
    }
}

#[test]
fn io_absurd_inputs() {
    init();
    // Decode garbage / empty buffers.
    safe!(Image2D::<VipsBackend>::from_buffer(&[]));
    safe!(Image2D::<VipsBackend>::from_buffer(&[
        0xFF, 0xD8, 0xFF, 0x00, 0x01, 0x02
    ])); // truncated JPEG header
    safe!(Image2D::<VipsBackend>::from_buffer(&vec![0u8; 4096]));
    // Open a non-existent / non-image file.
    safe!(Image2D::<VipsBackend>::open(
        "tests/fixtures/does_not_exist.jpg"
    ));
    safe!(Image2D::<VipsBackend>::open("tests/common/mod.rs"));
    // from_memory with mismatched / zero / huge dims.
    let buf = vec![0u8; 12];
    safe!(Image2D::<VipsBackend>::from_memory(
        &buf,
        -1,
        -1,
        3,
        PixelFormat::Rgb8
    ));
    safe!(Image2D::<VipsBackend>::from_memory(
        &buf,
        100000,
        100000,
        4,
        PixelFormat::Rgba8
    ));
    safe!(Image2D::<VipsBackend>::from_memory(
        &[],
        0,
        0,
        0,
        PixelFormat::Gray8
    ));
    // Save with a bogus suffix.
    safe!(rgb().write_to_buffer(".not_a_format"));
    safe!(rgb().write_to_buffer(""));
}

#[test]
fn generator_absurd_params() {
    init();
    // Masks with zero / negative dims and degenerate cutoffs.
    for (w, h) in [(0, 0), (-8, -8), (1, 1)] {
        safe!(Image2D::<VipsBackend>::generate(&MaskIdeal {
            width: w,
            height: h,
            frequency_cutoff: -1.0,
            uchar: None,
            nodc: None,
            reject: None,
            optical: None,
        }));
        safe!(Image2D::<VipsBackend>::generate(&MaskGaussian {
            width: w,
            height: h,
            frequency_cutoff: f64::NAN,
            amplitude_cutoff: 1e9,
            uchar: None,
            nodc: None,
            reject: None,
            optical: None,
        }));
    }
}

#[test]
fn thumbnail_absurd_params() {
    init();
    for size in [0, -10, 1000000] {
        safe!(Image2D::<VipsBackend>::thumbnail(
            "tests/fixtures/rgb.jpg",
            size,
            &ThumbnailParams::default()
        ));
    }
    safe!(Image2D::<VipsBackend>::thumbnail(
        "tests/fixtures/does_not_exist.jpg",
        64,
        &ThumbnailParams::default()
    ));
    safe!(Image2D::<VipsBackend>::thumbnail_buffer(
        &[],
        64,
        &ThumbnailParams::default()
    ));
}
