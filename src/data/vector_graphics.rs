//! The vector graphics datatype — used to hold Vello scenes or SVG data.

use std::any::Any;
use std::hash::Hasher;

use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::work_unit::{Atomic, WorkUnit};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// VectorGraphics metadata. Usually resolution-independent, but may have a nominal bounding box.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorGraphicsKind {
    pub width: f32,
    pub height: f32,
}

impl VectorGraphicsKind {
    pub fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }
}

impl AnyKind for VectorGraphicsKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, _wu: &WorkUnit) -> u64 {
        0
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.width.to_bits());
        state.write_u32(self.height.to_bits());
    }
}

impl Kind for VectorGraphicsKind {
    type WorkUnit = Atomic;
}

/// What the user holds. Aliased over the generic core; per-backend.
pub type VectorGraphics<B> = Data<VectorGraphicsKind, B>;

// ── Interop: Vello -> Image2D ──────────────────────────────────────────────────

use crate::backend::gpu::GpuBackend;
use crate::backend::gpu::GpuView;
use crate::backend::vello::{VelloBackend, VelloHandle};
use crate::backend::vips::VipsBackend;
use crate::buffer::Buffer;
use crate::color::space::ColorSpace;
use crate::data::image::ImageKind;
use crate::error::Error;
use crate::io::{Source, Target};
use crate::pixel::format::PixelFormat;
use crate::work_unit::Region;
use std::sync::Arc;

pub struct VelloTarget;

impl Target<VectorGraphicsKind, VelloBackend> for VelloTarget {
    type Out = Arc<VelloHandle>;

    fn extract(
        &self,
        buf: &Buffer<VelloBackend>,
        _wu: &Atomic,
        _ctx: &(),
    ) -> Result<Self::Out, Error> {
        Ok(buf.payload.clone())
    }
}

pub struct VectorGraphicsImageSource {
    pub vector_graphics: VectorGraphics<VelloBackend>,
}

impl Source<GpuBackend> for VectorGraphicsImageSource {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        Arc::new(ImageKind {
            color_space: ColorSpace::SRGB,
            format: PixelFormat::Rgba8,
            width: self.vector_graphics.spec.width.max(1.0) as i32,
            height: self.vector_graphics.spec.height.max(1.0) as i32,
        })
    }

    fn fetch(
        &self,
        ctx: &crate::backend::gpu::GpuContext,
        _wu: &Region,
    ) -> Result<Buffer<GpuBackend>, Error> {
        use wgpu::util::DeviceExt;

        let vello_target = VelloTarget;
        let _vello_handle = self.vector_graphics.pull(&vello_target, Atomic)?;

        let spec: Arc<ImageKind> = <VectorGraphicsImageSource as Source<GpuBackend>>::spec(self);
        let pixel_count = (spec.width * spec.height) as usize;
        let bytes_per_pixel = spec.format.bytes_per_pixel() as usize;
        let size = pixel_count * bytes_per_pixel;

        let vec = vec![0u8; size];

        let wgpu_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vello_gpu_source"),
                contents: &vec,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Buffer {
            payload: crate::backend::gpu::GpuBuffer::from_raw(Arc::new(wgpu_buffer), size as u64),
            spec,
        })
    }

    fn lower(&self, cx: &mut crate::backend::gpu::GpuBuilder) {
        let wu = cx.wu().clone();
        let WorkUnit::Region(region) = &wu else {
            cx.fail(Error::InvalidWorkUnit(
                "image source expects a Region".into(),
            ));
            return;
        };
        match <VectorGraphicsImageSource as Source<GpuBackend>>::fetch(
            self,
            cx.ctx().as_ref(),
            region,
        ) {
            Ok(buf) => {
                let spec = <VectorGraphicsImageSource as Source<GpuBackend>>::spec(self);
                let geom = crate::backend::gpu::view::RegionParams::tight(spec.width, spec.height);
                cx.input(
                    spec.input(),
                    geom.into_block("region_in_{slot}"),
                    buf.payload,
                );
                cx.output(spec.output(&wu));
            }
            Err(e) => cx.fail(e),
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_usize(crate::node::NodeId::of(&self.vector_graphics.root).0);
    }
}

impl Source<VipsBackend> for VectorGraphicsImageSource {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        Arc::new(ImageKind {
            color_space: ColorSpace::SRGB,
            format: PixelFormat::Rgba8,
            width: self.vector_graphics.spec.width.max(1.0) as i32,
            height: self.vector_graphics.spec.height.max(1.0) as i32,
        })
    }

    fn fetch(
        &self,
        _ctx: &<VipsBackend as crate::backend::Backend>::Ctx,
        _wu: &Region,
    ) -> Result<Buffer<VipsBackend>, Error> {
        let vello_target = VelloTarget;
        let _vello_handle = self.vector_graphics.pull(&vello_target, Atomic)?;

        let spec: Arc<ImageKind> = <VectorGraphicsImageSource as Source<VipsBackend>>::spec(self);
        let pixel_count = (spec.width * spec.height) as usize;
        let bytes_per_pixel = spec.format.bytes_per_pixel() as usize;
        let size = pixel_count * bytes_per_pixel;
        let mut vec = vec![0u8; size];

        let ptr = unsafe {
            crate::ffi::vips_image_new_from_memory(
                vec.as_mut_ptr() as *const std::ffi::c_void,
                size,
                spec.width,
                spec.height,
                4, // bands
                crate::ffi::VipsBandFormat_VIPS_FORMAT_UCHAR,
            )
        };

        if ptr.is_null() {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }

        Ok(Buffer {
            payload: Arc::new(crate::backend::vips::VipsHandle { ptr }),
            spec,
        })
    }

    fn lower(&self, cx: &mut crate::backend::vips::VipsBuilder) {
        let spec = <VectorGraphicsImageSource as Source<VipsBackend>>::spec(self);
        let region = Region::full(
            (spec.width as i32, spec.height as i32),
            crate::work_unit::Lod(0),
        );
        let buf =
            <VectorGraphicsImageSource as Source<VipsBackend>>::fetch(self, &(), &region).unwrap();
        cx.emit((*buf.payload).clone());
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_usize(crate::node::NodeId::of(&self.vector_graphics.root).0);
    }
}

impl VectorGraphics<VelloBackend> {
    pub fn rasterize_vips(&self) -> crate::data::image::Image2D<VipsBackend> {
        let src = Arc::new(VectorGraphicsImageSource {
            vector_graphics: self.clone(),
        });
        crate::node::Data::from_source(src, Arc::new(()))
    }

    pub fn rasterize_gpu(
        &self,
        ctx: Arc<crate::backend::gpu::GpuContext>,
    ) -> crate::data::image::Image2D<GpuBackend> {
        let src = Arc::new(VectorGraphicsImageSource {
            vector_graphics: self.clone(),
        });
        crate::node::Data::from_source(src, ctx)
    }
}
