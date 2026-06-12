//! The image datatype — the whole thing in one place.
//!
//! `ImageKind` is the agnostic metadata (format + color space + extent).
//! `Image2D<B>` is what the user holds. Per-backend lowering capabilities
//! (`GpuView`, `VipsBand`) and a representative set of operations
//! (`Invert`, `Blur`) live here too. Everything is additive — no central enum.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use crate::backend::Backend;
use crate::backend::gpu::view::{OutBuffer, OutputWrap, RegionParams, View};
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use crate::backend::vips::{VipsBackend, VipsBand, VipsBuilder};
use crate::color::space::ColorSpace;
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::pixel::format::PixelFormat;
use crate::work_unit::{Region, Shape, WorkUnit};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// Image metadata: pixel encoding (`format` + `color_space`) and extent
/// (`width`/`height`). Backend-agnostic — the same value tags an image whether
/// it ends up on the GPU or on libvips.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageKind {
    pub format: PixelFormat,
    pub color_space: ColorSpace,
    pub width: i32,
    pub height: i32,
}

impl ImageKind {
    pub fn new(format: PixelFormat, color_space: ColorSpace, width: i32, height: i32) -> Self {
        Self {
            format,
            color_space,
            width,
            height,
        }
    }
    pub fn dims(&self) -> (i32, i32) {
        (self.width, self.height)
    }
}

impl AnyKind for ImageKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn shape(&self) -> Shape {
        Shape::Region
    }
    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        let bpp = self.format.bytes_per_pixel() as u64;
        match wu {
            WorkUnit::Region(r) => (r.w.max(0) as u64) * (r.h.max(0) as u64) * bpp,
            _ => 0,
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        // PixelFormat / ColorSpace are foreign types without `Hash`; a compact
        // Debug proxy is fine for this datatype's identity in the POC.
        state.write(format!("{:?}/{:?}", self.format, self.color_space).as_bytes());
        state.write_i32(self.width);
        state.write_i32(self.height);
    }
}

impl Kind for ImageKind {
    type WorkUnit = Region;
}

impl GpuView for ImageKind {
    /// Decode wrapper: kernels read the image through a `CodecRegion` that
    /// unpacks the pixel format to working `float4` on `read`.
    fn input(&self) -> View {
        View::new(
            "uint",
            format!("CodecRegion<{}, {}>", self.codec(), self.layout()),
            "{ {buf}, {region} }",
        )
    }

    /// The image codec sandwich: the kernel writes working `float4` to an
    /// `RWRegion` scratch; afterwards an `RWCodecRegion` encodes it back into
    /// the pixel-format target. Both this encode and the `input` decode are the
    /// image's own concern — the emitter and ops know nothing of codecs.
    fn output(&self) -> OutputWrap {
        OutputWrap {
            arg_type: "RWRegion".into(),
            arg_ctor: "{ {buf}, {region} }".into(),
            arg_buffer: OutBuffer::Scratch,
            buffer_type: "uint".into(),
            encode: Some(View::new(
                "Atomic<uint>",
                format!("RWCodecRegion<{}, {}>", self.codec(), self.layout()),
                "{ {buf}, {region} }",
            )),
        }
    }
}

impl ImageKind {
    fn codec(&self) -> &'static str {
        match self.format {
            PixelFormat::RgbaF32 | PixelFormat::RgbF32 | PixelFormat::GrayF32 | PixelFormat::GrayAF32 => "F32Codec",
            PixelFormat::Rgba16 | PixelFormat::Rgb16 | PixelFormat::Gray16 | PixelFormat::GrayA16 => "U16Codec",
            _ => "U8Codec",
        }
    }
    /// `CH` = the `ChannelLayout` enum value (uint) the codec switches on.
    fn layout(&self) -> u32 {
        match self.format {
            PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaF32 => 0,
            PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::RgbF32 => 1,
            PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayF32 => 2,
            PixelFormat::GrayA8 | PixelFormat::GrayA16 | PixelFormat::GrayAF32 => 3,
            _ => 0,
        }
    }

    /// The format with `count` bands, same sample type as `self` — used by
    /// `ExtractBand`/`Bandjoin`'s GPU output spec. Family-aware (byte size
    /// alone can't disambiguate U16 vs F16, both 2 bytes/sample); only
    /// U8/U16/F32 codecs exist, so F16-family inputs fall back to F32. Band
    /// counts outside 1..=4 keep `self.format` (no codec/layout for them).
    pub fn with_band_count(&self, count: i32) -> PixelFormat {
        match (self.codec(), count) {
            ("F32Codec", 1) => PixelFormat::GrayF32,
            ("F32Codec", 2) => PixelFormat::GrayAF32,
            ("F32Codec", 3) => PixelFormat::RgbF32,
            ("F32Codec", 4) => PixelFormat::RgbaF32,
            ("U16Codec", 1) => PixelFormat::Gray16,
            ("U16Codec", 2) => PixelFormat::GrayA16,
            ("U16Codec", 3) => PixelFormat::Rgb16,
            ("U16Codec", 4) => PixelFormat::Rgba16,
            (_, 1) => PixelFormat::Gray8,
            (_, 2) => PixelFormat::GrayA8,
            (_, 3) => PixelFormat::Rgb8,
            (_, 4) => PixelFormat::Rgba8,
            _ => self.format,
        }
    }
}

impl VipsBand for ImageKind {
    fn band_format(&self) -> i32 {
        // Map the pixel format to a VipsBandFormat enum value. Real mapping
        // lives in the FFI layer; a coarse byte-width split suffices here.
        match self.format.bytes_per_pixel() / self.format.channel_count().max(1) {
            1 => 0,  // VIPS_FORMAT_UCHAR
            2 => 2,  // VIPS_FORMAT_USHORT
            _ => 10, // VIPS_FORMAT_FLOAT
        }
    }
}

/// What the user holds. Aliased over the generic core; per-backend.
pub type Image2D<B> = Data<ImageKind, B>;

// ── Spatial ergonomics (Region-shaped only) ──────────────────────────────────

impl Image2D<VipsBackend> {
    pub fn open(path: &str) -> Result<Self, crate::error::Error> {
        let source = Arc::new(FileImageSource::new(path)?);
        let root = Arc::new(crate::node::Node::Source(source.clone()));
        Ok(crate::node::Data {
            root,
            spec: <FileImageSource as Source<VipsBackend>>::spec(&source),
            ctx: Arc::new(()),
            _m: std::marker::PhantomData,
        })
    }
}

impl<B: Backend> Image2D<B> {
    pub fn width(&self) -> i32 {
        self.spec.width
    }
    pub fn height(&self) -> i32 {
        self.spec.height
    }
    pub fn format(&self) -> PixelFormat {
        self.spec.format
    }
    pub fn color_space(&self) -> ColorSpace {
        self.spec.color_space
    }
}

// ── File to VIPS Bridge ───────────────────────────────────────────────────────

pub struct FileImageSource {
    spec: Arc<ImageKind>,
    pub filename: String,
}

impl FileImageSource {
    pub fn new(filename: &str) -> Result<Self, crate::error::Error> {
        let c = std::ffi::CString::new(filename)
            .map_err(|_| crate::error::Error::Vips("invalid filename".into()))?;
        let ptr = unsafe {
            crate::ffi::vips_image_new_from_file(
                c.as_ptr(),
                std::ptr::null_mut::<std::ffi::c_void>(),
            )
        };
        if ptr.is_null() {
            return Err(crate::error::Error::Vips(crate::backend::vips::vips_error()));
        }

        let width = unsafe { crate::ffi::vips_image_get_width(ptr) };
        let height = unsafe { crate::ffi::vips_image_get_height(ptr) };
        let bands = unsafe { crate::ffi::vips_image_get_bands(ptr) };
        let format_raw = unsafe { crate::ffi::vips_image_get_format(ptr) };

        unsafe { crate::ffi::g_object_unref(ptr as *mut std::ffi::c_void) };

        let format =
            <PixelFormat as crate::backend::vips::FromVipsBandFormat>::from_vips_band_format(
                format_raw, bands,
            );
        let spec = Arc::new(ImageKind::new(format, ColorSpace::SRGB, width, height));

        Ok(Self {
            spec,
            filename: filename.to_string(),
        })
    }
}

impl Source<VipsBackend> for FileImageSource {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        self.spec.clone()
    }

    fn fetch(
        &self,
        _ctx: &<VipsBackend as Backend>::Ctx,
        _wu: &Region,
    ) -> Result<Buffer<VipsBackend>, crate::error::Error> {
        let c = std::ffi::CString::new(self.filename.as_str()).unwrap();
        let ptr = unsafe {
            crate::ffi::vips_image_new_from_file(
                c.as_ptr(),
                std::ptr::null_mut::<std::ffi::c_void>(),
            )
        };
        if ptr.is_null() {
            return Err(crate::error::Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(Buffer {
            payload: Arc::new(crate::backend::vips::VipsHandle { ptr }),
            spec: self.spec.clone(),
        })
    }

    fn lower(&self, cx: &mut VipsBuilder) {
        let c = std::ffi::CString::new(self.filename.as_str()).unwrap();
        let ptr = unsafe {
            crate::ffi::vips_image_new_from_file(
                c.as_ptr(),
                std::ptr::null_mut::<std::ffi::c_void>(),
            )
        };
        cx.emit(crate::backend::vips::VipsHandle { ptr });
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        state.write(self.filename.as_bytes());
    }
}

// ── File to RAW Bridge ────────────────────────────────────────────────────────

pub struct RawFileImageSource {
    spec: Arc<ImageKind>,
    pub handle: std::sync::Mutex<crate::backend::raw::handle::RawHandle>,
}

impl RawFileImageSource {
    pub fn new(
        path: &str,
        params: crate::backend::raw::RawDecodeParams,
    ) -> Result<Self, crate::error::Error> {
        let handle = crate::backend::raw::handle::RawHandle::open_with(path, params)?;

        let format = match handle.params().output_bps {
            16 => PixelFormat::Rgba16,
            _ => PixelFormat::Rgba8,
        };

        let spec = Arc::new(ImageKind::new(
            format,
            ColorSpace::SRGB,
            handle.raw_width() as i32,
            handle.raw_height() as i32,
        ));

        Ok(Self {
            spec,
            handle: std::sync::Mutex::new(handle),
        })
    }
}

impl Source<crate::backend::raw::RawBackend> for RawFileImageSource {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        self.spec.clone()
    }

    fn fetch(
        &self,
        _ctx: &<crate::backend::raw::RawBackend as Backend>::Ctx,
        _wu: &Region,
    ) -> Result<Buffer<crate::backend::raw::RawBackend>, crate::error::Error> {
        let handle = self.handle.lock().unwrap().clone();
        Ok(Buffer {
            payload: Arc::new(handle),
            spec: self.spec.clone(),
        })
    }

    fn lower(&self, cx: &mut <crate::backend::raw::RawBackend as Backend>::Builder) {
        let handle = self.handle.lock().unwrap().clone();
        cx.emit(Arc::new(handle));
    }

    fn dyn_hash(&self, _state: &mut dyn std::hash::Hasher) {
        // Source identity
    }
}

impl Image2D<crate::backend::raw::RawBackend> {
    pub fn open_raw(
        path: &str,
        params: crate::backend::raw::RawDecodeParams,
    ) -> Result<Self, crate::error::Error> {
        let source = Arc::new(RawFileImageSource::new(path, params)?);
        let root = Arc::new(crate::node::Node::Source(source.clone()));
        Ok(crate::node::Data {
            root,
            spec: <RawFileImageSource as Source<crate::backend::raw::RawBackend>>::spec(&source),
            ctx: Arc::new(()),
            _m: std::marker::PhantomData,
        })
    }
}

// ── VIPS to GPU Bridge ────────────────────────────────────────────────────────

use crate::buffer::Buffer;
use crate::io::Source;

/// A GPU Source that reads from a Vips pipeline.
/// This enforces the boundary invariant: data enters the GPU ONLY through a Source.
pub struct VipsImageSource {
    pub vips_img: Image2D<VipsBackend>,
}

impl VipsImageSource {
    pub fn new(vips_img: Image2D<VipsBackend>) -> Self {
        Self { vips_img }
    }
}

impl Source<GpuBackend> for VipsImageSource {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        self.vips_img.spec.clone()
    }

    fn fetch(
        &self,
        ctx: &crate::backend::gpu::GpuContext,
        wu: &Region,
    ) -> Result<Buffer<GpuBackend>, crate::error::Error> {
        use wgpu::util::DeviceExt;

        // 1. Materialize the VIPS graph up to this node
        let vips_buffer = self.vips_img.materialize(wu.clone())?;

        // 2. Pull the raw bytes from the materialized VipsImage
        let mut size: usize = 0;
        let ptr = unsafe {
            crate::ffi::vips_image_write_to_memory(vips_buffer.payload.ptr, &mut size as *mut usize)
        };
        if ptr.is_null() {
            return Err(crate::error::Error::Vips(crate::backend::vips::vips_error()));
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };

        // 3. Upload bytes to a WGPU buffer
        let wgpu_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_vips_source"),
                contents: slice,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        unsafe { crate::ffi::g_free(ptr as *mut std::ffi::c_void) };

        Ok(Buffer {
            payload: crate::backend::gpu::GpuBuffer::from_raw(
                std::sync::Arc::new(wgpu_buffer),
                size as u64,
            ),
            spec: self.spec(),
        })
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        // Fetch + upload our own buffer here (symmetric to a vips source's
        // `emit`), so the materializer needs no `Node::Source` branch.
        let WorkUnit::Region(region) = &wu else {
            cx.fail(crate::error::Error::InvalidWorkUnit(
                "image source expects a Region".into(),
            ));
            return;
        };
        match self.fetch(cx.ctx().as_ref(), region) {
            Ok(buf) => {
                // The fetched buffer is the full image, tightly packed.
                let geom = RegionParams::tight(self.spec().width, self.spec().height);
                cx.input(self.spec().input(), geom, buf.payload);
            }
            Err(e) => cx.fail(e),
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        // Identity is based on the Vips pipeline root.
        let k = Arc::as_ptr(&self.vips_img.root) as *const () as usize;
        state.write_usize(k);
    }
}

pub struct GpuConstantSource {
    pub spec: Arc<ImageKind>,
    pub data: Vec<f32>,
}

impl Source<GpuBackend> for GpuConstantSource {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        self.spec.clone()
    }

    fn fetch(
        &self,
        ctx: &crate::backend::gpu::GpuContext,
        _wu: &Region,
    ) -> Result<Buffer<GpuBackend>, crate::error::Error> {
        use wgpu::util::DeviceExt;
        let bytes = unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4) };
        let wgpu_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_constant_source"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        Ok(Buffer {
            payload: crate::backend::gpu::GpuBuffer::from_raw(
                std::sync::Arc::new(wgpu_buffer),
                bytes.len() as u64,
            ),
            spec: self.spec.clone(),
        })
    }

    fn lower(&self, cx: &mut GpuBuilder) {
        let wu = cx.wu().clone();
        let WorkUnit::Region(region) = &wu else {
            cx.fail(crate::error::Error::InvalidWorkUnit(
                "constant source expects a Region".into(),
            ));
            return;
        };
        match self.fetch(cx.ctx().as_ref(), region) {
            Ok(buf) => {
                let geom = crate::backend::gpu::view::RegionParams::tight(self.spec.width, self.spec.height);
                cx.input(self.spec.input(), geom, buf.payload);
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

impl Image2D<GpuBackend> {
    pub fn from_constant_f32(
        ctx: Arc<crate::backend::gpu::GpuContext>,
        width: i32,
        height: i32,
        data: &[f32],
    ) -> Self {
        let spec = Arc::new(ImageKind {
            width,
            height,
            format: crate::pixel::format::PixelFormat::GrayF32,
            color_space: crate::color::space::ColorSpace::SRGB,
        });
        let src = GpuConstantSource {
            spec: spec.clone(),
            data: data.to_vec(),
        };
        Self {
            root: Arc::new(crate::node::Node::Source(Arc::new(src))),
            ctx,
            spec,
            _m: std::marker::PhantomData,
        }
    }
}

// ── Targets ───────────────────────────────────────────────────────────────────
use crate::io::Target;

/// A simple target that reads the image bytes into host RAM.
pub struct RamImageTarget;

impl Target<ImageKind, VipsBackend> for RamImageTarget {
    type Out = Vec<u8>;

    fn extract(
        &self,
        buf: &Buffer<VipsBackend>,
        _wu: &Region,
        _ctx: &<VipsBackend as Backend>::Ctx,
    ) -> Result<Self::Out, crate::error::Error> {
        let mut size: usize = 0;
        let ptr = unsafe {
            crate::ffi::vips_image_write_to_memory(buf.payload.ptr, &mut size as *mut usize)
        };
        if ptr.is_null() {
            return Err(crate::error::Error::Vips(crate::backend::vips::vips_error()));
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        let vec = slice.to_vec();
        unsafe { crate::ffi::g_free(ptr as *mut std::ffi::c_void) };
        Ok(vec)
    }
}

pub struct RawImageSource {
    pub raw_img: Image2D<crate::backend::raw::RawBackend>,
}

impl RawImageSource {
    pub fn new(raw_img: Image2D<crate::backend::raw::RawBackend>) -> Self {
        Self { raw_img }
    }
}

impl Source<VipsBackend> for RawImageSource {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        self.raw_img.spec.clone()
    }

    fn fetch(
        &self,
        _ctx: &<VipsBackend as Backend>::Ctx,
        wu: &Region,
    ) -> Result<Buffer<VipsBackend>, crate::error::Error> {
        let buf = self.raw_img.materialize(wu.clone())?;
        let mut handle = (*buf.payload).clone();
        let frame = handle.materialize()?;

        let bands = frame.colors as i32;
        let vips_format = match frame.bits {
            16 => crate::ffi::VipsBandFormat_VIPS_FORMAT_USHORT,
            _ => crate::ffi::VipsBandFormat_VIPS_FORMAT_UCHAR,
        };

        let ptr = unsafe {
            crate::ffi::vips_image_new_from_memory_copy(
                frame.pixel_data().as_ptr() as *const std::ffi::c_void,
                frame.pixel_data().len(),
                frame.width as i32,
                frame.height as i32,
                bands,
                vips_format,
            )
        };

        if ptr.is_null() {
            return Err(crate::error::Error::Vips(crate::backend::vips::vips_error()));
        }

        Ok(Buffer {
            payload: Arc::new(crate::backend::vips::VipsHandle { ptr }),
            spec: self.spec(),
        })
    }

    fn lower(&self, cx: &mut VipsBuilder) {
        let region = Region::full(
            (self.raw_img.width() as i32, self.raw_img.height() as i32),
            crate::work_unit::Lod(0),
        );
        let buf = self.fetch(&(), &region).unwrap();
        cx.emit((*buf.payload).clone());
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        let k = Arc::as_ptr(&self.raw_img.root) as *const () as usize;
        state.write_usize(k);
    }
}

impl Target<ImageKind, GpuBackend> for RamImageTarget {
    type Out = Vec<u8>;

    fn extract(
        &self,
        buf: &Buffer<GpuBackend>,
        _wu: &Region,
        ctx: &<GpuBackend as Backend>::Ctx,
    ) -> Result<Self::Out, crate::error::Error> {
        buf.payload.read_to_cpu(ctx)
    }
}

impl Target<ImageKind, crate::backend::raw::RawBackend> for RamImageTarget {
    type Out = Vec<u8>;

    fn extract(
        &self,
        buf: &Buffer<crate::backend::raw::RawBackend>,
        _wu: &Region,
        _ctx: &(),
    ) -> Result<Self::Out, crate::error::Error> {
        let mut handle = (*buf.payload).clone();
        let frame = handle.materialize()?;
        Ok(frame.pixel_data().to_vec())
    }
}
