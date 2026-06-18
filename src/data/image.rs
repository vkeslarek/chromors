use std::sync::Arc;
use std::hash::Hasher;

pub use chromors_core::data::image::{Image2D, ImageKind, GpuBufferTarget, RamImageTarget};
pub use chromors_backend_vips::data::vips_image::VipsImageExt;
use chromors_core::buffer::Buffer;
use chromors_core::io::Source;
use chromors_core::work_unit::{Region, WorkUnit};
use chromors_backend_wgpu::view::RegionParams;
use chromors_backend_wgpu::{GpuBackend, GpuBuilder, GpuView, GpuContext, GpuBuffer};
use chromors_backend_vips::VipsBackend;

pub struct VipsImageSource {
    pub vips_img: Image2D<VipsBackend>,
}

impl VipsImageSource {
    pub fn new(vips_img: Image2D<VipsBackend>) -> Self {
        Self { vips_img }
    }
}

fn lod_dims(w: i32, h: i32, lod: chromors_core::work_unit::Lod) -> (i32, i32) {
    let s = lod.scale_factor() as i32;
    ((w / s).max(1), (h / s).max(1))
}

fn clamp_region(region: &Region, w: i32, h: i32) -> Region {
    let x0 = region.x.clamp(0, w);
    let y0 = region.y.clamp(0, h);
    let x1 = (region.x + region.w).clamp(0, w);
    let y1 = (region.y + region.h).clamp(0, h);
    Region {
        x: x0,
        y: y0,
        w: (x1 - x0).max(1),
        h: (y1 - y0).max(1),
        lod: region.lod,
    }
}

impl Source<GpuBackend> for VipsImageSource {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        self.vips_img.spec.clone()
    }

    fn fetch(
        &self,
        ctx: &GpuContext,
        wu: &Region,
    ) -> Result<Buffer<GpuBackend>, chromors_core::error::Error> {
        use wgpu::util::DeviceExt;

        let scale = wu.lod.scale_factor();
        let (lod_w, lod_h) = lod_dims(self.spec().width, self.spec().height, wu.lod);
        let src = if scale > 1 {
            self.vips_img.shrink(scale as f64, scale as f64, None)
        } else {
            self.vips_img.clone()
        };

        let vips_buffer = src.materialize(Region::full((lod_w, lod_h), chromors_core::work_unit::Lod(0)))?;

        let region = clamp_region(wu, lod_w, lod_h);
        let mut crop = chromors_backend_vips::gobject::VipsGObject::new(b"extract_area\0")
            .map_err(|e| chromors_core::error::Error::Vips(format!("{e:?}")))?;
        crop.set_image("input", vips_buffer.payload.ptr);
        crop.set_int("left", region.x);
        crop.set_int("top", region.y);
        crop.set_int("width", region.w);
        crop.set_int("height", region.h);
        let cropped: chromors_backend_vips::VipsHandle = crop.run()
            .map_err(|e| chromors_core::error::Error::Vips(format!("{e:?}")))?;

        let mut size: usize = 0;
        let ptr = unsafe { chromors_backend_vips::ffi::vips_image_write_to_memory(cropped.ptr, &mut size as *mut usize) };
        if ptr.is_null() {
            return Err(chromors_core::error::Error::Vips(chromors_backend_vips::vips_error()));
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };

        let wgpu_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_vips_source"),
            contents: slice,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        unsafe { chromors_backend_vips::ffi::g_free(ptr as *mut std::ffi::c_void) };

        Ok(Buffer {
            payload: GpuBuffer::from_raw(Arc::new(wgpu_buffer), size as u64),
            spec: self.spec(),
        })
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let WorkUnit::Region(ref region) = wu else {
            cx.fail(chromors_core::error::Error::InvalidWorkUnit("image source expects a Region".into()));
            return;
        };
        match self.fetch(cx.ctx().as_ref(), region) {
            Ok(buf) => {
                let (lod_w, lod_h) = lod_dims(self.spec().width, self.spec().height, region.lod);
                let clamped = clamp_region(region, lod_w, lod_h);
                let pad_x = clamped.x - region.x;
                let pad_y = clamped.y - region.y;
                let geom = RegionParams::padded(
                    clamped.w as u32,
                    0,
                    0,
                    clamped.w as u32,
                    clamped.h as u32,
                    pad_x,
                    pad_y,
                );
                cx.input(self.spec().input(), geom.into_block("region_in_{slot}"), buf.payload);
            }
            Err(e) => cx.fail(e),
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_usize(chromors_core::node::NodeId::of(&self.vips_img.root).0);
    }
}
