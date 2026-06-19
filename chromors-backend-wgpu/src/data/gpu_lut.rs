use crate::{
    GpuBackend, GpuBuffer, GpuBuilder, GpuContext, GpuView, OutBuffer, OutputWrap, RegionParams,
    View,
};
use chromors_core::lut::RawLutTarget;
use chromors_core::*;
use std::hash::Hasher;
use std::sync::Arc;

impl GpuView for LutKind {
    fn input(&self) -> View {
        View::new(
            "float4",
            "Region",
            "{ {buf}, {params}[0].region_in_{slot} }",
        )
    }

    fn output(&self, wu: &WorkUnit) -> OutputWrap {
        let r = match wu {
            WorkUnit::Region(r) => r.clone(),
            WorkUnit::Range(range) => Region {
                x: range.start,
                y: 0,
                w: range.end - range.start,
                h: 1,
                lod: chromors_core::Lod(0),
            },
            _ => Region {
                x: 0,
                y: 0,
                w: self.entries as i32,
                h: 1,
                lod: chromors_core::Lod(0),
            },
        };
        OutputWrap {
            arg: View::new("float4", "RWRegion", "{ {buf}, {region} }"),
            dest: OutBuffer::Target,
            encode: None,
            params: RegionParams::tight(r.w, r.h).into_block("region_out"),
        }
    }
}

pub struct GpuConstantLutSource {
    pub spec: Arc<LutKind>,
    pub data: Vec<f32>,
}

impl Source<GpuBackend> for GpuConstantLutSource {
    type Kind = LutKind;

    fn spec(&self) -> Arc<LutKind> {
        self.spec.clone()
    }

    fn fetch(&self, ctx: &GpuContext, _wu: &Range) -> Result<Buffer<GpuBackend>, Error> {
        use wgpu::util::DeviceExt;
        let bytes = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4)
        };
        let wgpu_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_constant_lut_source"),
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
        let WorkUnit::Range(range) = &wu else {
            cx.fail(Error::InvalidWorkUnit("lut source expects a Range".into()));
            return;
        };
        let region = Region {
            x: range.start,
            y: 0,
            w: range.end - range.start,
            h: 1,
            lod: chromors_core::Lod(0),
        };
        match self.fetch(cx.ctx().as_ref(), range) {
            Ok(buf) => {
                let geom = RegionParams::tight(self.spec.entries as i32, 1);
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
        for &v in &self.data {
            state.write_u32(v.to_bits());
        }
    }
}

pub trait GpuLutExt {
    fn from_values(ctx: Arc<GpuContext>, entries: u32, bands: u32, values: &[f32]) -> Self;
}

impl GpuLutExt for Lut<GpuBackend> {
    fn from_values(ctx: Arc<GpuContext>, entries: u32, bands: u32, values: &[f32]) -> Self {
        let spec = Arc::new(LutKind::new(entries, bands));
        let bands = bands.min(4) as usize;
        let mut data = vec![0.0f32; entries as usize * 4];
        for i in 0..entries as usize {
            for b in 0..bands {
                data[i * 4 + b] = values[i * bands + b];
            }
        }
        let src = GpuConstantLutSource { spec, data };
        Data::from_source(Arc::new(src), ctx)
    }
}

impl Target<LutKind, GpuBackend> for RawLutTarget {
    type Out = Vec<u8>;

    fn extract(
        &self,
        buf: &Buffer<GpuBackend>,
        _wu: &Range,
        ctx: &GpuContext,
    ) -> Result<Self::Out, Error> {
        buf.payload.read_to_cpu(ctx)
    }
}
