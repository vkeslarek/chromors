//! LOD as a pull-demand dimension. `VipsImageSource` honors `Region.lod` by
//! shrinking on load (CPU/streaming in VIPS), so a coarse mip never decodes,
//! uploads, or processes full resolution on the GPU — this is what keeps large
//! images smooth (the chromors-viewer mip path). No GPU `Shrink` op involved.
use super::common;
use chromors::color::model::ColorModel;
use chromors::color::space::ColorSpace;
use chromors::data::image::RamImageTarget;
use chromors::pixel::{AlphaState, PixelLayout, Storage};
use chromors::work_unit::{Lod, Region};
use chromors::VipsImageExt;

fn disp() -> PixelLayout {
    PixelLayout {
        storage: Storage::U8,
        model: ColorModel::Rgb,
        alpha: AlphaState::Straight,
        color_space: ColorSpace::SRGB,
    }
}

/// Pulling the same graph at `Lod(k)` returns the tile downsampled by `2^k`,
/// matching a VIPS `shrink(2^k)` reference. Coordinates are LOD-space.
#[test]
fn lod_demand_matches_vips_shrink() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips = chromors::data::image::Image2D::<chromors::backend::vips::VipsBackend>::open("tests/fixtures/rgb.jpg").unwrap();
    let gpu = common::vips_to_gpu(&vips, &ctx);
    let disp_img = gpu.convert(disp());

    for lod in [0u32, 1, 2] {
        let scale = 1i32 << lod;
        let (lw, lh) = (200 / scale, 200 / scale); // rgb.jpg is 200x200
        let reg = Region {
            x: 0,
            y: 0,
            w: lw,
            h: lh,
            lod: Lod(lod),
        };
        let gpu_bytes = disp_img.pull(&RamImageTarget, reg).unwrap();

        // Every pixel opaque (straight-alpha RGBA8 of an alpha-less JPEG).
        let opaque = gpu_bytes.iter().skip(3).step_by(4).all(|&a| a == 255);
        assert!(opaque, "lod {lod}: found non-opaque alpha");
        assert_eq!(gpu_bytes.len(), (lw * lh * 4) as usize, "lod {lod} size");

        // Reference: VIPS shrink to the same level, RGB compared.
        let vips_ref = if lod == 0 {
            vips.clone()
        } else {
            vips.shrink(scale as f64, scale as f64, None)
        };
        let vips_bytes = common::vips_materialize(&vips_ref);
        let mut gpu_rgb = Vec::with_capacity((lw * lh * 3) as usize);
        for px in 0..(lw * lh) as usize {
            gpu_rgb.extend_from_slice(&gpu_bytes[px * 4..px * 4 + 3]);
        }
        let rms = common::rms_u8(&vips_bytes[..gpu_rgb.len()], &gpu_rgb);
        println!("lod {lod}: {lw}x{lh} RMS vs vips shrink = {rms}");
        assert!(rms < 5.0, "lod {lod}: RMS too high {rms}");
    }
}
