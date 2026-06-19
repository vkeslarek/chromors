use super::*;
use chromors::color::model::ColorModel;
use chromors::color::space::ColorSpace;
use chromors::data::image::Image2D;
use chromors::node::Data;
use chromors::pixel::{AlphaState, PixelLayout, Storage};
use chromors_core::generator::{Constant, GenSource};
use std::sync::Arc;

#[test]
fn constant_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();

    let layout = PixelLayout {
        storage: Storage::U8,
        model: ColorModel::Rgb,
        alpha: AlphaState::Straight,
        color_space: ColorSpace::SRGB,
    };

    let color = [0.2, 0.5, 0.8, 1.0];

    let gpu_src = Arc::new(GenSource(Constant {
        w: 128,
        h: 128,
        layout,
        color,
    }));
    let vips_src = Arc::new(GenSource(Constant {
        w: 128,
        h: 128,
        layout,
        color,
    }));

    let gpu_img: Image2D<GpuBackend> = Data::from_source(gpu_src, ctx.clone());
    let vips_img: Image2D<VipsBackend> = Data::from_source(vips_src, Arc::new(()));

    let vips_bytes = common::vips_materialize(&vips_img);
    let gpu_bytes = common::poc_materialize(&gpu_img);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("constant RMS = {}", rms);
    assert!(rms < 2.0, "constant diff too high: {}", rms);
}

use chromors_core::generator::{GaussNoise, LinearGradient, Xyz};

#[test]
fn linear_gradient_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();

    let layout = PixelLayout {
        storage: Storage::U8,
        model: ColorModel::Rgb,
        alpha: AlphaState::Straight,
        color_space: ColorSpace::SRGB,
    };

    let params = LinearGradient {
        w: 128,
        h: 128,
        layout,
        c0: [1.0, 0.0, 0.0, 1.0],
        c1: [0.0, 0.0, 1.0, 1.0],
        angle: 1.57079632679,
    };

    let gpu_src = Arc::new(GenSource(params.clone()));
    let vips_src = Arc::new(GenSource(params));

    let ctx = common::gpu_ctx();
    let gpu_img: Image2D<GpuBackend> = Data::from_source(gpu_src, ctx.clone());
    let vips_img: Image2D<VipsBackend> = Data::from_source(vips_src, Arc::new(()));

    let vips_bytes = common::vips_materialize(&vips_img);
    let gpu_bytes = common::poc_materialize(&gpu_img);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("linear_gradient RMS = {}", rms);
    let mut diff_count = 0;
    for i in 0..vips_bytes.len() {
        if vips_bytes[i].abs_diff(gpu_bytes[i]) > 10 {
            if diff_count < 10 {
                println!("diff at {}: vips {} gpu {}", i, vips_bytes[i], gpu_bytes[i]);
            }
            diff_count += 1;
        }
    }
    assert!(rms < 2.0, "linear_gradient diff too high: {}", rms);
}

#[test]
fn xyz_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();

    let layout = PixelLayout {
        storage: Storage::F32,
        model: ColorModel::Rgb, // we just need a layout
        alpha: AlphaState::Straight,
        color_space: ColorSpace::SRGB,
    };

    let params = Xyz {
        w: 64,
        h: 64,
        layout,
    };

    let gpu_src = Arc::new(GenSource(params.clone()));
    let vips_src = Arc::new(GenSource(params));

    let gpu_img: Image2D<GpuBackend> = Data::from_source(gpu_src, ctx.clone());
    let vips_img: Image2D<VipsBackend> = Data::from_source(vips_src, Arc::new(()));

    let vips_bytes = common::vips_materialize(&vips_img);
    let gpu_bytes = common::poc_materialize(&gpu_img);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_f32(&vips_bytes, &gpu_bytes);
    println!("xyz RMS = {}", rms);
    assert!(rms < 1.0, "xyz diff too high: {}", rms);
}

#[test]
fn gaussnoise_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();

    let layout = PixelLayout {
        storage: Storage::F32,
        model: ColorModel::Rgb,
        alpha: AlphaState::Straight,
        color_space: ColorSpace::SRGB,
    };

    let params = GaussNoise {
        w: 128,
        h: 128,
        layout,
        mean: 128.0,
        sigma: 20.0,
        seed: 42,
    };

    let gpu_src = Arc::new(GenSource(params.clone()));
    let vips_src = Arc::new(GenSource(params));

    let gpu_img: Image2D<GpuBackend> = Data::from_source(gpu_src, ctx.clone());
    let vips_img: Image2D<VipsBackend> = Data::from_source(vips_src, Arc::new(()));

    let vips_bytes = common::vips_materialize(&vips_img);
    let gpu_bytes = common::poc_materialize(&gpu_img);

    assert_eq!(vips_bytes.len(), gpu_bytes.len());
    let rms = common::rms_f32(&vips_bytes, &gpu_bytes);
    println!("gaussnoise RMS = {}", rms);

    // GaussNoise is inherently random, so exact match isn't expected if the PRNG implementations differ slightly.
    // However, since we re-implemented the exact same logic (pcg2d + Box-Muller) in both, they should match EXACTLY
    // (modulo tiny float discrepancies). Let's use a small tolerance.
    assert!(rms < 1.0, "gaussnoise diff too high: {}", rms);
}
