import lib.pixel;

/// What an image's channels represent — mirrors the Rust `ColorModel`
/// (`src/color/model.rs`) and `GpuModelId::gpu_model` (`src/backend/gpu/mod.rs`,
/// CLAUDE.md §3.6). `Multiband` never reaches `color_convert` (rejected by
/// `convert_matrices`) but is included for parity with the Rust enum.
enum ColorModel : uint {
    Rgb      = 0,
    Gray     = 1,
    Cmyk     = 2,
    YCbCr    = 3,
    Lab      = 4,
    Xyz      = 5,
    Yxy      = 6,
    Lch      = 7,
    Hsv      = 8,
    Oklab    = 9,
    Oklch    = 10,
    ScRgb    = 11,
    Multiband = 12,
}
