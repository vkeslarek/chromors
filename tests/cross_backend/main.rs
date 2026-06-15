#[path = "../common/mod.rs"]
pub mod common;
use poc::backend::gpu::GpuBackend;
use poc::backend::vips::VipsBackend;
use poc::data::image::Image2D as GenImage;
use poc::data::mask2d::Mask2D;
use poc::operation::composite::{BlendMode, Composite2};
use poc::operation::geometry::{Angle, Angle45, CompassDirection, Direction, Extend, Interesting};
use poc::operation::{OperationBoolean, OperationMath, OperationMorphology, OperationRound};
use poc::pixel::Storage;

/// Read a vips image whose runtime pixels are FLOAT (e.g. promoted by
/// `linear`) but whose `format()` metadata still reports the pre-promotion
/// u8 format (an `output_spec` staleness, same family of issue as `Shrink`).
/// Pulls the raw bytes and reinterprets them as `f32` directly, ignoring
/// `format()`. Returns normalized+clamped [0,1] values (raw vips `linear`
/// output is `in_u8 * gain`, i.e. in the 0..255*gain range).
fn vips_materialize_linear_f32_norm(img: &GenImage<VipsBackend>) -> Vec<f32> {
    use poc::io::Target;
    use poc::work_unit::{Lod, Region};
    let (w, h) = (img.width(), img.height());
    let bands = img.layout().channel_count() as usize;
    let target = poc::data::image::RamImageTarget;
    let bytes = img
        .pull(
            &target,
            Region {
                x: 0,
                y: 0,
                w: w as i32,
                h: h as i32,
                lod: Lod(0),
            },
        )
        .unwrap();
    let pixel_count = w as usize * h as usize * bands;
    let floats: &[f32] = bytemuck::cast_slice(&bytes);
    floats
        .iter()
        .take(pixel_count)
        .map(|v| (v / 255.0).clamp(0.0, 1.0))
        .collect()
}

mod arithmetic;
mod bands;
mod chain_bug;
mod color;
mod composite;
mod edge;
mod filters;
mod geometry;
mod icc;
mod lod_demand;
mod misc;
mod mosaicing;
mod opacity;
mod pixel;
mod stats;
mod viewer_repro;
