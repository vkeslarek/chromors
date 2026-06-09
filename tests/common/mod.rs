#![allow(dead_code)]

use std::sync::{Arc, Mutex, Once};

use bytemuck;
use pixors_engine::backend::gpu::{GpuBackend, Rect, context::GpuContext};
use pixors_engine::backend::vips::VipsBackend;
use pixors_engine::data::image::Image as GenImage;
use pixors_engine::data::image::Image;

static INIT: Once = Once::new();

pub static VIPS_SERIAL: Mutex<()> = Mutex::new(());

pub fn vips_serial() -> std::sync::MutexGuard<'static, ()> {
    VIPS_SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn init() {
    INIT.call_once(pixors_engine::init);
}

pub fn rgb() -> Image<VipsBackend> {
    init();
    Image::<VipsBackend>::open("tests/fixtures/rgb.jpg").unwrap()
}

pub fn gray() -> Image<VipsBackend> {
    init();
    Image::<VipsBackend>::open("tests/fixtures/gray.jpg").unwrap()
}

pub fn rgba() -> Image<VipsBackend> {
    init();
    Image::<VipsBackend>::open("tests/fixtures/rgba.png").unwrap()
}

pub fn gpu_ctx() -> Arc<GpuContext> {
    GpuContext::new().expect("GPU adapter required for GPU tests")
}

/// Upload a vips image to the POC GpuBackend.
pub fn vips_to_gpu(img: &Image<VipsBackend>, ctx: &Arc<GpuContext>) -> GenImage<GpuBackend> {
    let src = pixors_engine::backend::gpu::source::GpuSource::new_vips(img.clone(), ctx.clone());
    GenImage::<GpuBackend>::new_from_source(&src).unwrap()
}

/// Materialize a POC GPU image and read back the raw f32 bytes.
pub fn poc_materialize(img: &GenImage<GpuBackend>) -> Vec<u8> {
    let (w, h) = (img.handle.width as i32, img.handle.height as i32);
    let rect = Rect::new(0, 0, w, h);
    let target = pixors_engine::target::ImageTarget::new(img.clone());
    let mat = target.pull(rect, 0).unwrap();
    mat.buffer.read_to_cpu(&img.handle.node.ctx).unwrap()
}

/// Read vips bytes as f32 in [0, 1] range.
pub fn vips_to_f32(img: &Image<VipsBackend>) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let target = pixors_engine::target::ImageTarget::new(img.clone());
    let mat = target.pull(Rect::new(0, 0, w as i32, h as i32), 0).unwrap();
    mat.buffer.iter().map(|b| *b as f32 / 255.0).collect()
}

pub fn vips_materialize(img: &Image<VipsBackend>) -> Vec<u8> {
    let (w, h) = (img.width(), img.height());
    let target = pixors_engine::target::ImageTarget::new(img.clone());
    let mat = target.pull(Rect::new(0, 0, w as i32, h as i32), 0).unwrap();
    mat.buffer
}

/// Convert POC f32 output (4 channels, [0,1]) to u8 `bands`-channel slice
/// for comparison with vips u8 output.
pub fn poc_f32_to_u8(bytes: &[u8], w: usize, h: usize, bands: usize) -> Vec<u8> {
    let pixel_count = w * h;
    if bytes.len() == pixel_count * 4 {
        let mut out = Vec::with_capacity(pixel_count * bands);
        for p in 0..pixel_count {
            for c in 0..bands {
                out.push(bytes[p * 4 + c]);
            }
        }
        return out;
    }

    let pixels: &[f32] = bytemuck::cast_slice(bytes);
    let mut out = Vec::with_capacity(pixel_count * bands);
    for p in 0..pixel_count {
        for c in 0..bands {
            let val = (pixels[p * 4 + c] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            out.push(val);
        }
    }
    out
}

/// RMS error between two `u8` byte slices (0..255 scale).
pub fn rms_u8(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    (sse / a.len() as f64).sqrt()
}

pub fn rms_f32(a: &[u8], b: &[u8]) -> f64 {
    let af: &[f32] = bytemuck::cast_slice(a);
    let bf: &[f32] = bytemuck::cast_slice(b);
    assert_eq!(af.len(), bf.len(), "length mismatch");
    let sse: f64 = af
        .iter()
        .zip(bf)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum();
    (sse / af.len() as f64).sqrt()
}

/// Convert any Vips materialized image bytes to normalized f32 [0, 1].
/// Handles u8, u16, and f32 formats.
pub fn vips_materialize_f32(img: &Image<VipsBackend>) -> Vec<f32> {
    let (w, h) = (img.width(), img.height());
    let bands = img.bands() as usize;
    let target = pixors_engine::target::ImageTarget::new(img.clone());
    let mat = target.pull(Rect::new(0, 0, w as i32, h as i32), 0).unwrap();
    let bps = mat.meta.format.bytes_per_pixel() as usize / mat.meta.format.channel_count() as usize;
    let pixel_count = w as usize * h as usize * bands;
    match bps {
        1 => mat
            .buffer
            .iter()
            .take(pixel_count)
            .map(|&b| b as f32 / 255.0)
            .collect(),
        2 => {
            let pixels: &[u16] = bytemuck::cast_slice(&mat.buffer);
            pixels
                .iter()
                .take(pixel_count)
                .map(|&p| p as f32 / 65535.0)
                .collect()
        }
        _ => {
            let pixels: &[f32] = bytemuck::cast_slice(&mat.buffer);
            pixels.iter().take(pixel_count).copied().collect()
        }
    }
}

/// Convert f32 bytes to u8 bytes for RMS comparison, preserving exact float encoding.
pub fn f32_to_bytes_u8(f32_values: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice::<f32, u8>(f32_values).to_vec()
}

pub fn rms_u8_interior(a: &[u8], b: &[u8], w: usize, h: usize, bands: usize, border: usize) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    let (mut sse, mut n) = (0.0f64, 0u64);
    for y in border..(h - border) {
        for x in border..(w - border) {
            for c in 0..bands {
                let i = (y * w + x) * bands + c;
                let d = (a[i] as f64) - (b[i] as f64);
                sse += d * d;
                n += 1;
            }
        }
    }
    (sse / n as f64).sqrt()
}

pub fn rms_f32_interior(
    a: &[u8],
    b: &[u8],
    w: usize,
    h: usize,
    bands: usize,
    border: usize,
) -> f64 {
    let af: &[f32] = bytemuck::cast_slice(a);
    let bf: &[f32] = bytemuck::cast_slice(b);
    let (mut sse, mut n) = (0.0f64, 0u64);
    for y in border..(h - border) {
        for x in border..(w - border) {
            for c in 0..bands {
                let i = (y * w + x) * bands + c;
                let d = (af[i] - bf[i]) as f64;
                sse += d * d;
                n += 1;
            }
        }
    }
    (sse / n as f64).sqrt()
}
