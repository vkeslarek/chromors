//! Regression test for the chromors-viewer mip tile pipeline.
//!
//! The viewer builds `VipsImageSource -> convert(display RGBA8) ->
//! with_lod(mip)` and pulls per-tile sub-regions. The `convert` step is a 1:1
//! producer fused upstream of `with_lod`'s resampling `Shrink`. Before the
//! fusion-barrier fix, the fused pass dispatched at the (downsampled) output
//! domain while `shrink_kernel` read source coordinates `idx * factor` from a
//! domain-sized work buffer — out of bounds into uninitialized VRAM. The result
//! was non-deterministic garbage (the corrupted atlas tiles users saw at mip>0)
//! that nonetheless sometimes matched the reference. These tests lock in:
//!  - determinism: repeated identical pulls agree;
//!  - correctness: GPU tiles match a vips `shrink` reference;
//!  - target parity: `GpuBufferTarget` (viewport path) == `RamImageTarget`.
use super::common;

use std::sync::Arc;
use chromors::color::model::ColorModel;
use chromors::color::space::ColorSpace;
use chromors::data::image::{GpuBufferTarget, RamImageTarget};
use chromors::pixel::{AlphaState, PixelLayout, Storage};
use chromors::work_unit::{Lod, Region};
use chromors::VipsImageExt;

fn display_layout() -> PixelLayout {
    PixelLayout {
        storage: Storage::U8,
        model: ColorModel::Rgb,
        alpha: AlphaState::Straight,
        color_space: ColorSpace::SRGB,
    }
}

fn large_display_mip(
    ctx: &Arc<chromors::backend::gpu::GpuContext>,
) -> chromors::data::image::Image2D<chromors::backend::gpu::GpuBackend> {
    let vips_img = chromors::data::image::Image2D::<chromors::backend::vips::VipsBackend>::open(
        "tests/fixtures/large.jpg",
    )
    .unwrap();
    let gpu_img = common::vips_to_gpu(&vips_img, ctx);
    gpu_img.convert(display_layout()).with_lod(Lod(2)) // factor 4: 2000² -> 500²
}

#[test]
fn convert_then_lod_is_deterministic() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let mip_img = large_display_mip(&ctx);
    let region = Region {
        x: 0,
        y: 0,
        w: 256,
        h: 256,
        lod: Lod(0),
    };

    let a = mip_img.pull(&RamImageTarget, region.clone()).unwrap();
    let b = mip_img.pull(&RamImageTarget, region.clone()).unwrap();
    let c = mip_img.pull(&RamImageTarget, region).unwrap();
    assert_eq!(a, b, "repeated pull diverged (a vs b)");
    assert_eq!(a, c, "repeated pull diverged (a vs c)");
}

#[test]
fn convert_then_lod_tiles_match_vips_shrink() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();

    let vips_img = chromors::data::image::Image2D::<chromors::backend::vips::VipsBackend>::open(
        "tests/fixtures/large.jpg",
    )
    .unwrap();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);
    let mip_img = gpu_img.convert(display_layout()).with_lod(Lod(2));

    let vips_shrunk = vips_img.shrink(4.0, 4.0, None);

    const TILE: i32 = 256;
    for ty in 0..2 {
        for tx in 0..2 {
            let w = TILE.min(500 - tx * TILE);
            let h = TILE.min(500 - ty * TILE);
            let region = Region {
                x: tx * TILE,
                y: ty * TILE,
                w,
                h,
                lod: Lod(0),
            };

            let gpu_buf = mip_img.pull(&GpuBufferTarget, region.clone()).unwrap();
            let gpu_bytes = gpu_buf.read_to_cpu(&ctx).unwrap();
            let ram_bytes = mip_img.pull(&RamImageTarget, region).unwrap();

            assert_eq!(gpu_bytes.len(), (w * h * 4) as usize);
            assert_eq!(
                gpu_bytes, ram_bytes,
                "tile ({tx},{ty}): GpuBufferTarget != RamImageTarget"
            );

            let vips_ref = vips_shrunk.crop(tx * TILE, ty * TILE, w, h);
            let vips_bytes = common::vips_materialize(&vips_ref);

            let mut gpu_rgb = Vec::with_capacity((w * h * 3) as usize);
            for px in 0..(w * h) as usize {
                gpu_rgb.extend_from_slice(&gpu_bytes[px * 4..px * 4 + 3]);
            }
            let rms = common::rms_u8(&vips_bytes, &gpu_rgb);
            assert!(
                rms < 5.0,
                "tile ({tx},{ty}) RMS vs vips shrink too high: {rms}"
            );
        }
    }
}
