//! Regression: fusing two ops that push a same-named `param_block` field.
//!
//! `Exposure` and `Brightness` share `exposure_kernel` and both push
//! `gain`/`preserve` via `cx.param_block`. Before `param_block` step-namespaced
//! its fields, the two `gain`s deduped to one declaration in `ChainParams`
//! while their distinct bytes desynced the struct layout — `domain`/`region_out`
//! then read garbage and the dispatch wrote nothing (fully black/transparent
//! output). This pins the fix (CLAUDE.md §5.2.4).
use super::common;
use poc::color::model::ColorModel;
use poc::color::space::ColorSpace;
use poc::data::image::RamImageTarget;
use poc::pixel::{AlphaState, PixelLayout, Storage};
use poc::work_unit::{Lod, Region};

fn disp() -> PixelLayout {
    PixelLayout {
        storage: Storage::U8,
        model: ColorModel::Rgb,
        alpha: AlphaState::Straight,
        color_space: ColorSpace::SRGB,
    }
}

#[test]
fn fused_same_named_param_blocks_dont_collide() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips =
        poc::data::image::Image2D::<poc::backend::vips::VipsBackend>::open("tests/fixtures/rgb.jpg")
            .unwrap();
    let g = common::vips_to_gpu(&vips, &ctx);

    // Both ops push `gain`/`preserve` through `exposure_kernel`; exposure +
    // brightness then exposure(-stops) ≈ a mild net change, never all-zero.
    let out = g
        .exposure(0.5, 0.0)
        .brightness(0.04)
        .exposure(-0.5, 0.0)
        .convert(disp());

    let reg = Region { x: 0, y: 0, w: 32, h: 32, lod: Lod(0) };
    let bytes = out.pull(&RamImageTarget, reg).unwrap();

    // Every pixel opaque, and not a black hole: the fused pass produced real
    // pixels (the desync bug zeroed everything, alpha included).
    assert!(
        bytes.iter().skip(3).step_by(4).all(|&a| a == 255),
        "alpha not opaque"
    );
    let rgb_nonzero = (0..bytes.len()).filter(|i| i % 4 != 3 && bytes[*i] != 0).count();
    assert!(rgb_nonzero > 0, "fused param_block ops produced all-zero RGB");
}
