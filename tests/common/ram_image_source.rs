use chromors_backend_wgpu::view::RegionParams;
use chromors_backend_wgpu::{GpuBackend, GpuBuffer, GpuBuilder, GpuContext, GpuView};
use chromors_core::buffer::Buffer;
use chromors_core::data::image::ImageKind;
use chromors_core::error::Error;
use chromors_core::io::Source;
use chromors_core::work_unit::Region;
use std::hash::Hasher;
use std::sync::Arc;

pub struct RamImageSource {
    pub spec: Arc<ImageKind>,
    pub data: Vec<u8>,
}

impl Source<GpuBackend> for RamImageSource {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        self.spec.clone()
    }

    fn fetch(&self, ctx: &GpuContext, _wu: &Region) -> Result<Buffer<GpuBackend>, Error> {
        use wgpu::util::DeviceExt;
        let buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RamImageSource"),
                contents: &self.data,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let gpu_buf = GpuBuffer::from_raw(Arc::new(buf), self.data.len() as u64);
        Ok(Buffer {
            payload: gpu_buf,
            spec: self.spec.clone(),
        })
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let chromors_core::work_unit::WorkUnit::Region(region) = &wu else {
            cx.fail(Error::InvalidWorkUnit(
                "RamImageSource expects a Region".into(),
            ));
            return;
        };
        match self.fetch(cx.ctx().as_ref(), region) {
            Ok(buf) => {
                let geom = RegionParams::tight(self.spec.width, self.spec.height);
                cx.input(
                    self.spec.input(),
                    geom.into_block("region_in_{slot}"),
                    buf.payload,
                );
                cx.output(self.spec.output(&wu));
            }
            Err(e) => cx.fail(e),
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_usize(self.data.len());
    }
}
