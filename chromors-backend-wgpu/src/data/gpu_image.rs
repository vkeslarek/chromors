use crate::prelude::*;

impl GpuView for ImageKind {
    fn input(&self) -> View {
        View::new(
            "uint",
            format!(
                "CodecRegion<{}, {}>",
                self.layout.storage.gpu_codec(),
                self.layout.channel_count()
            ),
            "{ {buf}, {params}[0].region_in_{slot} }",
        )
    }
    fn output(&self, wu: &WorkUnit) -> OutputWrap {
        let r = Region::typed(wu).expect("ImageKind::output: Region-shaped WorkUnit");
        OutputWrap {
            arg: View::new("uint", "RWRegion", "{ {buf}, {region} }"),
            dest: OutBuffer::Scratch,
            encode: Some(View::new(
                "Atomic<uint>",
                format!(
                    "RWCodecRegion<{}, {}>",
                    self.layout.storage.gpu_codec(),
                    self.layout.channel_count()
                ),
                "{ {buf}, {region} }",
            )),
            params: RegionParams::tight(r.w, r.h).into_block("region_out"),
        }
    }
}

impl Target<ImageKind, GpuBackend> for GpuBufferTarget {
    type Out = Arc<GpuBuffer>;
    fn extract(
        &self,
        buf: &Buffer<GpuBackend>,
        _wu: &Region,
        _ctx: &GpuContext,
    ) -> Result<Self::Out, Error> {
        Ok(buf.payload.clone())
    }
}

impl Target<ImageKind, GpuBackend> for RamImageTarget {
    type Out = Vec<u8>;

    fn extract(
        &self,
        buf: &Buffer<GpuBackend>,
        _wu: &Region,
        ctx: &<GpuBackend as Backend>::Ctx,
    ) -> Result<Self::Out, Error> {
        buf.payload.read_to_cpu(ctx)
    }
}
