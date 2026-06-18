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
use crate::backend::vips::{VipsBackend, VipsBand, VipsBuilder, VipsHandle};
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
        View::new(
            "float",
            "MaskRegion",
            "{ {buf}, {params}[0].region_in_{slot} }",
        )
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
        let WorkUnit::Region(region) = &wu else {
            cx.fail(Error::InvalidWorkUnit(
                "mask source expects a Region".into(),
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
        for &v in &self.data {
            state.write_u32(v.to_bits());
        }
    }
}

impl Mask2D<GpuBackend> {
    /// A mask holding `values` (row-major, `width * height` entries).
    pub fn from_values(ctx: Arc<GpuContext>, width: i32, height: i32, values: &[f32]) -> Self {
        let spec = Arc::new(Mask2DKind::new(width, height));
        let src = GpuConstantMaskSource {
            spec: spec.clone(),
            data: values.to_vec(),
        };
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

// ── Vips constant source ────────────────────────────────────────────────────

/// A Vips leaf holding a constant `f64` weight grid — `conv`/`morph`-family
/// ops read mask images in vips' native `VIPS_FORMAT_DOUBLE`.
pub struct VipsConstantMaskSource {
    pub spec: Arc<Mask2DKind>,
    pub data: Vec<f64>,
    pub scale: f64,
    pub offset: f64,
}

impl Source<VipsBackend> for VipsConstantMaskSource {
    type Kind = Mask2DKind;

    fn spec(&self) -> Arc<Mask2DKind> {
        self.spec.clone()
    }

    fn fetch(&self, _ctx: &(), _wu: &Region) -> Result<Buffer<VipsBackend>, Error> {
        let ptr = unsafe {
            crate::ffi::vips_image_new_from_memory_copy(
                self.data.as_ptr() as *const std::ffi::c_void,
                self.data.len() * 8,
                self.spec.width,
                self.spec.height,
                1,
                crate::ffi::VipsBandFormat_VIPS_FORMAT_DOUBLE,
            )
        };
        if ptr.is_null() {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        // vips_conv/conva/compass/morph read the mask's "scale"/"offset"
        // double properties (defaulting to 0 if unset, which divides the
        // convolution result by zero).
        unsafe {
            let scale = std::ffi::CString::new("scale").unwrap();
            let offset = std::ffi::CString::new("offset").unwrap();
            let xoffset = std::ffi::CString::new("xoffset").unwrap();
            let yoffset = std::ffi::CString::new("yoffset").unwrap();
            crate::ffi::vips_image_set_double(ptr, scale.as_ptr(), self.scale);
            crate::ffi::vips_image_set_double(ptr, offset.as_ptr(), self.offset);
            crate::ffi::vips_image_set_int(ptr, xoffset.as_ptr(), self.spec.width / 2);
            crate::ffi::vips_image_set_int(ptr, yoffset.as_ptr(), self.spec.height / 2);
        }
        Ok(Buffer {
            payload: Arc::new(VipsHandle { ptr }),
            spec: self.spec.clone(),
        })
    }

    fn lower(&self, cx: &mut VipsBuilder) {
        let region = Region::full(
            (self.spec.width, self.spec.height),
            crate::work_unit::Lod(0),
        );
        let buf = self.fetch(&(), &region).unwrap();
        cx.emit((*buf.payload).clone());
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        for &v in &self.data {
            state.write_u64(v.to_bits());
        }
    }
}

impl Mask2D<VipsBackend> {
    /// A mask holding `values` (row-major, `width * height` entries), with
    /// vips' default `scale=1`/`offset=0` (unscaled weights).
    pub fn from_values(width: i32, height: i32, values: &[f32]) -> Self {
        Self::from_values_scaled(width, height, values, 1.0, 0.0)
    }

    /// A mask holding `values`, with explicit `scale`/`offset` set on the
    /// vips mask image (read by `vips_conv`/`conva`/`compass`/`morph` to
    /// normalise the convolution result: `result = sum(src*weight) / scale +
    /// offset`).
    pub fn from_values_scaled(
        width: i32,
        height: i32,
        values: &[f32],
        scale: f64,
        offset: f64,
    ) -> Self {
        let spec = Arc::new(Mask2DKind::new(width, height));
        let data: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        let src = VipsConstantMaskSource {
            spec: spec.clone(),
            data,
            scale,
            offset,
        };
        Data::from_source(Arc::new(src), Arc::new(()))
    }

    /// The `n x n` identity matrix.
    pub fn identity(n: i32) -> Self {
        let dim = n.max(0) as usize;
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            data[i * dim + i] = 1.0;
        }
        Self::from_values(n, n, &data)
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

// ── Targets ──────────────────────────────────────────────────────────────────

use crate::io::Target;
use crate::backend::Backend;

/// Extracts the mask's raw `f32` weight grid into host RAM.
///
/// This is the sanctioned exit for mask data — analogous to `RamImageTarget`
/// for images.
pub struct RamMaskTarget;

impl Target<Mask2DKind, VipsBackend> for RamMaskTarget {
    type Out = Vec<f32>;

    fn extract(
        &self,
        buf: &Buffer<VipsBackend>,
        _wu: &Region,
        _ctx: &<VipsBackend as Backend>::Ctx,
    ) -> Result<Self::Out, Error> {
        let mut size: usize = 0;
        let ptr = unsafe {
            crate::ffi::vips_image_write_to_memory(buf.payload.ptr(), &mut size as *mut usize)
        };
        if ptr.is_null() {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        // Vips mask images are VIPS_FORMAT_DOUBLE (f64)
        let count = size / 8;
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const f64, count) };
        let values: Vec<f32> = slice.iter().map(|&v| v as f32).collect();
        unsafe { crate::ffi::g_free(ptr as *mut std::ffi::c_void) };
        Ok(values)
    }
}

impl Target<Mask2DKind, GpuBackend> for RamMaskTarget {
    type Out = Vec<f32>;

    fn extract(
        &self,
        buf: &Buffer<GpuBackend>,
        _wu: &Region,
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
