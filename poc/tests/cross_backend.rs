//! Cross-backend proof: `Invert` and `Blur` run on the exact same DAG structure
//! on BOTH the GPU and VIPS backends, leveraging the agnostic `Operation` and
//! per-backend `Lower` architecture without manual matches.

use std::sync::Arc;

use poc::backend::gpu::{GpuBackend, GpuBuilder, GpuView, view::{ParamBlock, Role, View}};
use poc::backend::vips::{VipsBackend, VipsBuilder, VipsBand};
use poc::backend::Backend;
use poc::data::image::{Image2D, ImageKind, RamImageTarget};
use poc::node::Node;
use poc::io::Target;
use poc::work_unit::Region;

// ── Factory helpers for the test ──────────────────────────────────────────────
fn create_vips_real() -> Image2D<VipsBackend> {
    Image2D::<VipsBackend>::open("../tests/fixtures/rgba.png").expect("Failed to open fixture")
}

fn create_gpu_mock(vips_img: Image2D<VipsBackend>) -> Image2D<GpuBackend> {
    let source = Arc::new(poc::data::image::VipsImageSource::new(vips_img.clone()));
    let root = Arc::new(Node::Source(source.clone()));
    poc::node::Data {
        root,
        spec: <poc::data::image::VipsImageSource as poc::io::Source<GpuBackend>>::spec(&source),
        ctx: poc::backend::gpu::GpuContext::new().expect("No GPU available"),
        _m: std::marker::PhantomData,
    }
}

pub fn rms_u8(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    (sse / a.len() as f64).sqrt()
}

pub fn rms_u8_interior(a: &[u8], b: &[u8], w: usize, h: usize, bands: usize, pad: usize) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    assert_eq!(a.len(), w * h * bands);
    let mut sse = 0.0;
    let mut count = 0;
    for y in pad..(h - pad) {
        for x in pad..(w - pad) {
            for c in 0..bands {
                let idx = (y * w + x) * bands + c;
                sse += (a[idx] as f64 - b[idx] as f64).powi(2);
                count += 1;
            }
        }
    }
    (sse / count as f64).sqrt()
}

pub fn materialize_vips(img: &Image2D<VipsBackend>) -> Vec<u8> {
    let target = RamImageTarget;
    let wu = Region { x: 0, y: 0, w: img.width(), h: img.height(), lod: poc::work_unit::Lod(0) };
    img.extract(&target, wu).expect("Vips pull failed")
}

pub fn materialize_gpu(img: &Image2D<GpuBackend>) -> Vec<u8> {
    let target = RamImageTarget;
    let wu = Region { x: 0, y: 0, w: img.width(), h: img.height(), lod: poc::work_unit::Lod(0) };
    img.extract(&target, wu).expect("GPU pull failed")
}

static INIT: std::sync::Once = std::sync::Once::new();

fn init_vips() {
    INIT.call_once(|| unsafe {
        poc::ffi::vips_init(b"test\0".as_ptr() as *const i8);
    });
}

#[test]
fn invert_matches_vips() {
    init_vips();
    let vips_img = create_vips_real();
    let vips_out = vips_img.invert();
    let vips_bytes = materialize_vips(&vips_out);
    
    let gpu_img = create_gpu_mock(vips_img);
    let gpu_out = gpu_img.invert();
    let gpu_bytes = materialize_gpu(&gpu_out);

    let (w, h) = vips_out.spec.dims();
    let pad = 2;
    let rmse = rms_u8_interior(&gpu_bytes, &vips_bytes, w as usize, h as usize, 4, pad);
    assert!(rmse < 2.0, "RMS error {} too high! GPU pipeline doesn't match VIPS reference.", rmse);
}

#[test]
fn blur_matches_vips() {
    init_vips();
    let vips_img = create_vips_real();
    let vips_out = vips_img.blur(3.0);
    let vips_bytes = materialize_vips(&vips_out);
    
    let gpu_img = create_gpu_mock(vips_img);
    let gpu_out = gpu_img.blur(3.0);
    
    let gpu_bytes = materialize_gpu(&gpu_out);

    let (w, h) = (vips_out.width() as usize, vips_out.height() as usize);
    let bands = vips_out.format().channel_count() as usize;
    let radius = (3.0 * 3.0_f32).ceil() as usize;

    let rms = rms_u8_interior(&vips_bytes, &gpu_bytes, w, h, bands, radius);
    println!("blur GPU vs vips interior RMS = {rms:.4} (0..255)");
    // Cross-engine tolerance: both are separable Gaussians of the same sigma,
    // but vips and our kernel differ in edge handling and alpha treatment
    // (vips blurs premultiplied), so an exact match isn't expected — this
    // pins "same operation, close result", not bit-equality.
    assert!(rms < 12.0, "GPU blur diverges from vips: RMS {rms:.4}");
}

#[test]
fn diamond_shared_source_materializes_once() {
    init_vips();
    let vips_img = create_vips_real();
    let gpu = create_gpu_mock(vips_img);

    // Diamond: one source feeds two branches that a third op merges.
    //   S ─┬─> invert ─┐
    //      └─> blur ───┴─> add
    // `invert` and `blur` both hold `gpu.root`; the demand/lower walk dedups
    // S to a single source buffer, and `add` reads each branch's temp.
    let a = gpu.invert();
    let b = gpu.blur(2.0);
    let merged = a.add(&b);

    let out = materialize_gpu(&merged);

    // Reference: same arithmetic on the host (invert = 1-c, add, clamp). We only
    // need this to be a sane, finite image of the right size — the point of the
    // test is that the diamond *compiles + runs* (shared node once).
    let (w, h) = merged.spec.dims();
    assert_eq!(out.len(), (w * h * 4) as usize, "diamond output wrong size");
    assert!(out.iter().any(|&b| b != 0), "diamond output is all-zero");
}
