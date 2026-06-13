//! The lookup-table datatype — a 1-D raw-f32 table, `entries` samples x
//! `bands` channels. The first `Range`-shaped Kind: a LUT is genuinely
//! 1-D, unlike `Mask2DKind`'s small 2-D grids.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use crate::backend::gpu::view::{OutBuffer, OutputWrap, RegionParams, View};
use crate::backend::gpu::{GpuBackend, GpuBuffer, GpuBuilder, GpuContext, GpuView};
use crate::backend::vips::{VipsBackend, VipsBand, VipsBuilder, VipsHandle};
use crate::buffer::Buffer;
use crate::error::Error;
use crate::io::{Source, Target};
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::work_unit::{Range, Region, WorkUnit};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// LUT metadata: `entries` samples x `bands` channels, raw `f32`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LutKind {
    pub entries: u32,
    pub bands: u32,
}

impl LutKind {
    pub fn new(entries: u32, bands: u32) -> Self {
        Self { entries, bands }
    }
}

impl AnyKind for LutKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        let entries = match wu {
            WorkUnit::Range(r) => (r.end - r.start).max(0) as u64,
            _ => self.entries as u64,
        };
        (entries * self.bands.max(1) as u64 * 4).max(16)
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.entries);
        state.write_u32(self.bands);
    }
}

impl Kind for LutKind {
    type WorkUnit = Range;
}

impl GpuView for LutKind {
    /// A LUT is stored as `entries` packed `float4`s (`bands` <= 4 channels,
    /// zero-padded) — the existing raw `Region` wrapper (`StructuredBuffer<float4>`)
    /// reads it directly, indexed `(entry, 0)` by the data-driven kernels.
    fn input(&self) -> View {
        View::new(
            "float4",
            "Region",
            "{ {buf}, {params}[0].region_in_{slot} }",
        )
    }

    fn output(&self, wu: &WorkUnit) -> OutputWrap {
        // LUT entries are 1-D (`Kind::WorkUnit = Range`), but the GPU buffer
        // is addressed as a `width x 1` `Region` — accept a `Range` here too,
        // mapping it to `Region { x: start, y: 0, w: end - start, h: 1 }`.
        let r = match wu {
            WorkUnit::Region(r) => r.clone(),
            WorkUnit::Range(range) => Region {
                x: range.start,
                y: 0,
                w: range.end - range.start,
                h: 1,
                lod: crate::work_unit::Lod(0),
            },
            WorkUnit::Atomic => panic!("LutKind::output: Atomic WorkUnit not supported"),
        };
        OutputWrap {
            arg: View::new("float4", "RWRegion", "{ {buf}, {region} }"),
            dest: OutBuffer::Target,
            encode: None,
            params: RegionParams::tight(r.w, r.h).into_block("region_out"),
        }
    }
}

/// What the user holds. Aliased over the generic core; per-backend.
pub type Lut<B> = Data<LutKind, B>;

// ── GPU constant source ────────────────────────────────────────────────────

/// A GPU leaf holding a constant LUT — `entries` packed `float4`s
/// (row-major, `bands` <= 4 channels, zero-padded).
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
        match self.fetch(
            cx.ctx().as_ref(),
            &Range {
                start: 0,
                end: self.spec.entries as i32,
            },
        ) {
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

impl Lut<GpuBackend> {
    /// A LUT holding `values` (row-major, `entries * bands` floats, `bands <= 4`,
    /// packed into `float4`s with zero padding).
    pub fn from_values(ctx: Arc<GpuContext>, entries: u32, bands: u32, values: &[f32]) -> Self {
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

impl VipsBand for LutKind {
    fn band_format(&self) -> i32 {
        // vips "matrix" images (e.g. invertlut's input) are double-precision,
        // 1-band, `Xsize x Ysize` = `bands x entries`.
        crate::ffi::VipsBandFormat_VIPS_FORMAT_DOUBLE
    }
}

// ── Vips constant source ────────────────────────────────────────────────────

/// A Vips leaf holding a constant `f64` LUT/matrix — `entries` rows x
/// `bands` columns, row-major, as a 1-band `VIPS_FORMAT_DOUBLE` image of
/// `Xsize = bands`, `Ysize = entries` (vips' "matrix image" convention).
pub struct VipsConstantLutSource {
    pub spec: Arc<LutKind>,
    pub data: Vec<f64>,
}

impl Source<VipsBackend> for VipsConstantLutSource {
    type Kind = LutKind;

    fn spec(&self) -> Arc<LutKind> {
        self.spec.clone()
    }

    fn fetch(&self, _ctx: &(), _wu: &Range) -> Result<Buffer<VipsBackend>, Error> {
        let ptr = unsafe {
            crate::ffi::vips_image_new_from_memory_copy(
                self.data.as_ptr() as *const std::ffi::c_void,
                self.data.len() * 8,
                self.spec.bands as i32,
                self.spec.entries as i32,
                1,
                crate::ffi::VipsBandFormat_VIPS_FORMAT_DOUBLE,
            )
        };
        if ptr.is_null() {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(Buffer {
            payload: Arc::new(VipsHandle { ptr }),
            spec: self.spec.clone(),
        })
    }

    fn lower(&self, cx: &mut VipsBuilder) {
        let buf = self
            .fetch(
                &(),
                &Range {
                    start: 0,
                    end: self.spec.entries as i32,
                },
            )
            .unwrap();
        cx.emit((*buf.payload).clone());
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        for &v in &self.data {
            state.write_u64(v.to_bits());
        }
    }
}

impl Lut<VipsBackend> {
    /// A LUT/matrix holding `values` (row-major, `entries * bands` doubles).
    pub fn from_values(entries: u32, bands: u32, values: &[f32]) -> Self {
        let spec = Arc::new(LutKind::new(entries, bands));
        let data: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        let src = VipsConstantLutSource {
            spec: spec.clone(),
            data,
        };
        Data::from_source(Arc::new(src), Arc::new(()))
    }
}

// ── Raw targets ─────────────────────────────────────────────────────────────

/// Reads a LUT buffer back to host RAM as raw bytes — GPU side gives packed
/// `float4` (`entries * 16` bytes), Vips side gives row-major `f64` (`entries
/// * bands * 8` bytes, 1-band `Xsize x Ysize` matrix image).
pub struct RawLutTarget;

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

impl Target<LutKind, VipsBackend> for RawLutTarget {
    type Out = Vec<u8>;

    fn extract(
        &self,
        buf: &Buffer<VipsBackend>,
        _wu: &Range,
        _ctx: &(),
    ) -> Result<Self::Out, Error> {
        let mut size: usize = 0;
        let ptr = unsafe {
            crate::ffi::vips_image_write_to_memory(buf.payload.ptr, &mut size as *mut usize)
        };
        if ptr.is_null() {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        let vec = slice.to_vec();
        unsafe { crate::ffi::g_free(ptr as *mut std::ffi::c_void) };
        Ok(vec)
    }
}
