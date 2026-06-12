//! The 2-D mask datatype — small raw-f32 weight grids: convolution masks,
//! morphology elements, band-recombination matrices. NOT colorimetric: no
//! `PixelFormat`, no `ColorSpace`, no codec sandwich. Weights are bound and
//! read as plain `f32`, broadcast to `float4(v, v, v, 1)` for `IRegion`
//! consumers (same trick as the Gray codecs).

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use crate::backend::gpu::view::{OutBuffer, OutputWrap, RegionParams, View};
use crate::backend::gpu::{GpuBackend, GpuBuffer, GpuBuilder, GpuContext, GpuView};
use crate::backend::vips::VipsBand;
use crate::buffer::Buffer;
use crate::error::Error;
use crate::io::Source;
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::work_unit::{Region, WorkUnit, WorkUnitFor};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// Mask metadata: just an extent. A raw `f32` grid, no pixel format, no color
/// space — there is nothing left to lie about.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask2DKind {
    pub width: i32,
    pub height: i32,
}

impl Mask2DKind {
    pub fn new(width: i32, height: i32) -> Self {
        Self { width, height }
    }
    pub fn dims(&self) -> (i32, i32) {
        (self.width, self.height)
    }
}

impl AnyKind for Mask2DKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        match wu {
            WorkUnit::Region(r) => (r.w.max(0) as u64) * (r.h.max(0) as u64) * 4,
            _ => 0,
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height);
    }
}

impl Kind for Mask2DKind {
    type WorkUnit = Region;
}

impl GpuView for Mask2DKind {
    /// Raw read, no codec: `MaskRegion` reads `f32` and broadcasts `(v,v,v,1)`
    /// like the existing Gray wrappers — kernels keep their `IRegion` inputs.
    fn input(&self) -> View {
        View::new("float", "MaskRegion", "{ {buf}, {params}[0].region_in_{slot} }")
    }

    /// Raw `f32` write, no encode step — the kernel writes the target buffer
    /// directly through `RWMaskRegion`.
    fn output(&self, wu: &WorkUnit) -> OutputWrap {
        let r = Region::typed(wu).expect("Mask2DKind::output: Region-shaped WorkUnit");
        OutputWrap {
            arg: View::new("float", "RWMaskRegion", "{ {buf}, {region} }"),
            dest: OutBuffer::Target,
            encode: None,
            params: RegionParams::tight(r.w, r.h).into_block("region_out"),
        }
    }
}

impl VipsBand for Mask2DKind {
    fn band_format(&self) -> i32 {
        // vips "matrix images" are double-precision.
        crate::ffi::VipsBandFormat_VIPS_FORMAT_DOUBLE
    }
}

/// What the user holds. Aliased over the generic core; per-backend.
pub type Mask2D<B> = Data<Mask2DKind, B>;

// ── GPU constant source ──────────────────────────────────────────────────────

/// A GPU leaf holding a constant `f32` grid — replaces
/// `Image2D::from_constant_f32` for non-colorimetric weight data.
pub struct GpuConstantMaskSource {
    pub spec: Arc<Mask2DKind>,
    pub data: Vec<f32>,
}

impl Source<GpuBackend> for GpuConstantMaskSource {
    type Kind = Mask2DKind;

    fn spec(&self) -> Arc<Mask2DKind> {
        self.spec.clone()
    }

    fn fetch(&self, ctx: &GpuContext, _wu: &Region) -> Result<Buffer<GpuBackend>, Error> {
        use wgpu::util::DeviceExt;
        let bytes = unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4) };
        let wgpu_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
        let WorkUnit::Region(region) = &wu else {
            cx.fail(Error::InvalidWorkUnit("mask source expects a Region".into()));
            return;
        };
        match self.fetch(cx.ctx().as_ref(), region) {
            Ok(buf) => {
                let geom = RegionParams::tight(self.spec.width, self.spec.height);
                cx.input(self.spec.input(), geom.into_block("region_in_{slot}"), buf.payload);
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

impl Mask2D<GpuBackend> {
    /// A mask holding `values` (row-major, `width * height` entries).
    pub fn from_values(ctx: Arc<GpuContext>, width: i32, height: i32, values: &[f32]) -> Self {
        let spec = Arc::new(Mask2DKind::new(width, height));
        let src = GpuConstantMaskSource { spec: spec.clone(), data: values.to_vec() };
        Data::from_source(Arc::new(src), ctx)
    }

    /// The `n x n` identity matrix.
    pub fn identity(ctx: Arc<GpuContext>, n: i32) -> Self {
        let dim = n.max(0) as usize;
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self::from_values(ctx, n, n, &data)
    }
}

impl<B: crate::backend::Backend> Mask2D<B> {
    pub fn width(&self) -> i32 {
        self.spec.width
    }
    pub fn height(&self) -> i32 {
        self.spec.height
    }
}
