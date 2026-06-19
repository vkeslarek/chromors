use crate::{GpuBackend, GpuBuffer, GpuBuilder, GpuContext, GpuView, RegionParams};
use chromors_core::data::mask2d::{Mask2D, Mask2DKind, RamMaskTarget};
use chromors_core::*;
use std::hash::Hasher;
use std::sync::Arc;

pub trait GpuMask2DExt {
    fn identity(ctx: Arc<GpuContext>, n: i32) -> Mask2D<GpuBackend>;
    fn from_values(
        ctx: Arc<GpuContext>,
        width: i32,
        height: i32,
        values: &[f32],
    ) -> Mask2D<GpuBackend>;
    fn extract_target(&self) -> GpuBufferTarget;
}

impl GpuMask2DExt for Mask2D<GpuBackend> {
    fn from_values(ctx: Arc<GpuContext>, width: i32, height: i32, values: &[f32]) -> Self {
        let spec = Arc::new(Mask2DKind::new(width, height));
        let src = GpuConstantMaskSource {
            spec: spec.clone(),
            data: values.to_vec(),
        };
        chromors_core::node::Data::from_source(Arc::new(src), ctx)
    }

    fn identity(ctx: Arc<GpuContext>, n: i32) -> Self {
        let dim = n.max(0) as usize;
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self::from_values(ctx, n, n, &data)
    }

    fn extract_target(&self) -> GpuBufferTarget {
        GpuBufferTarget
    }
}

pub struct GpuConstantMaskSource {
    pub spec: Arc<Mask2DKind>,
    pub data: Vec<f32>,
}

impl Source<GpuBackend> for GpuConstantMaskSource {
    type Kind = Mask2DKind;

    fn spec(&self) -> Arc<Mask2DKind> {
        self.spec.clone()
    }

    fn fetch(
        &self,
        ctx: &GpuContext,
        _wu: &chromors_core::work_unit::Region,
    ) -> Result<Buffer<GpuBackend>, Error> {
        use wgpu::util::DeviceExt;
        let bytes = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4)
        };
        let wgpu_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_constant_mask_source"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        Ok(Buffer {
            payload: GpuBuffer::from_raw(Arc::new(wgpu_buffer), bytes.len() as u64),
            spec: self.spec.clone(),
        })
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let chromors_core::work_unit::WorkUnit::Region(region) = &wu else {
            cx.fail(Error::InvalidWorkUnit(
                "mask source expects a Region".into(),
            ));
            return;
        };
        match self.fetch(cx.ctx().as_ref(), region) {
            Ok(buf) => {
                let geom = RegionParams::tight(self.spec.width, self.spec.height);
                cx.input(
                    crate::view::View::new(
                        "float",
                        "MaskRegion",
                        "{ {buf}, {params}[0].region_in_{slot} }",
                    ),
                    geom.into_block("region_in_{slot}"),
                    buf.payload,
                );
                let r = chromors_core::work_unit::Region::typed(&wu).unwrap();
                cx.output(crate::view::OutputWrap {
                    arg: crate::view::View::new(
                        "float",
                        "RWMaskRegion",
                        "{ {buf}, {params}[0].region_out }",
                    ),
                    dest: crate::view::OutBuffer::Target,
                    encode: None,
                    params: crate::view::RegionParams::tight(r.w as i32, r.h as i32)
                        .into_block("region_out"),
                });
            }
            Err(e) => cx.fail(e),
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        for &v in &self.data {
            state.write_u32(v.to_bits());
        }
    }
}

pub struct GpuBufferTarget;

impl Target<Mask2DKind, GpuBackend> for GpuBufferTarget {
    type Out = Arc<GpuBuffer>;

    fn extract(
        &self,
        buf: &Buffer<GpuBackend>,
        _wu: &chromors_core::work_unit::Region,
        _ctx: &GpuContext,
    ) -> Result<Self::Out, Error> {
        Ok(buf.payload.clone())
    }
}

impl Target<Mask2DKind, GpuBackend> for RamMaskTarget {
    type Out = Vec<f32>;

    fn extract(
        &self,
        buf: &Buffer<GpuBackend>,
        _wu: &chromors_core::work_unit::Region,
        ctx: &<GpuBackend as Backend>::Ctx,
    ) -> Result<Self::Out, Error> {
        let bytes = buf.payload.read_to_cpu(ctx)?;
        // GPU masks are f32
        let count = bytes.len() / 4;
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(values)
    }
}
