use pixors_engine::backend::gpu::GpuBackend;
use pixors_engine::backend::gpu::{GpuContext, GpuSource};
use pixors_engine::backend::vips::VipsBackend;
use pixors_engine::color::space::ColorSpace;
use pixors_engine::data::image::Image;
use pixors_engine::geometry::Rect;
use pixors_engine::pixel::{AlphaPolicy, PixelFormat, PixelMeta};
use std::sync::Arc;

#[test]
fn vips_source_stride_subrect() {
    pixors_engine::init();

    let device = pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        (device, queue)
    });

    let ctx = GpuContext::from_device(Arc::new(device.0), Arc::new(device.1));

    let meta = PixelMeta::new(PixelFormat::Rgba8, ColorSpace::SRGB, AlphaPolicy::Straight);
    let mut pixels = vec![0u8; 100 * 100 * 4];
    for i in 0..(100 * 100) {
        pixels[i * 4] = 255;
        pixels[i * 4 + 1] = 0;
        pixels[i * 4 + 2] = 0;
        pixels[i * 4 + 3] = 255;
    }

    let vips_image = Image::<VipsBackend>::from_memory(&pixels, 100, 100, 4, meta.format).unwrap();
    let gpu_image =
        Image::<GpuBackend>::new_from_source(&GpuSource::new_vips(vips_image, ctx.clone()))
            .unwrap();

    use pixors_engine::operation::misc::ExposureOperation;
    let gpu_image = gpu_image
        .execute(&ExposureOperation {
            stops: 0.0,
            preserve: 0.0,
        })
        .unwrap();

    let rect = Rect::new(10, 10, 20, 20);
    let region = pixors_engine::backend::gpu::region::GpuRegion::new(
        gpu_image.handle.node.graph.clone(),
        gpu_image.handle.node.ctx.cache.clone(),
        gpu_image.handle.node.root_id,
        ctx.clone(),
        pixors_engine::Lod::FULL,
    );
    region.prepare(rect);
    let mat = region.materialize().unwrap();

    let buf = match &*mat {
        pixors_engine::backend::gpu::GraphValue::Image { buffer, .. } => buffer,
        _ => panic!("Expected Image buffer"),
    };
    assert_eq!(buf.width, 20);
    assert_eq!(buf.height, 20);

    let bytes = buf
        .read_subrect_to_cpu(Rect::new(0, 0, 20, 20), &ctx.device, &ctx.queue)
        .unwrap();

    let f32_data: &[f32] = bytemuck::cast_slice(&bytes);
    let mut all_red = true;
    let mut first_err = None;
    for i in 0..(20 * 20) {
        let r = f32_data[i * 4];
        let g = f32_data[i * 4 + 1];
        let b = f32_data[i * 4 + 2];
        let a = f32_data[i * 4 + 3];
        if r < 0.9 || g > 0.1 || b > 0.1 || a < 0.9 {
            all_red = false;
            first_err = Some((i, r, g, b, a));
            break;
        }
    }
    assert!(
        all_red,
        "Expected solid red at ({}, {}), got {:?}",
        first_err.unwrap().0 % 20,
        first_err.unwrap().0 / 20,
        (
            first_err.unwrap().1,
            first_err.unwrap().2,
            first_err.unwrap().3,
            first_err.unwrap().4
        )
    );
}
