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
use crate::backend::gpu::{GpuBackend, GpuBuilder, GpuStorageCodec, GpuView};
use crate::backend::vips::{FromVipsBandFormat, IntoVipsBandFormat, VipsBackend, VipsBand, VipsBuilder};
use crate::color::model::ColorModel;
use crate::color::space::ColorSpace;
use crate::kind::{AnyKind, Kind};
use crate::node::Data;
use crate::pixel::{AlphaState, PixelLayout, Storage, layout_with_bands};
use crate::work_unit::{Region, WorkUnit, WorkUnitFor};

// ── Kind ──────────────────────────────────────────────────────────────────────

/// Image metadata: pixel layout (storage/model/alpha/color space) and extent
/// (`width`/`height`). Backend-agnostic — the same value tags an image whether
/// it ends up on the GPU or on libvips.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageKind {
    pub layout: PixelLayout,
    pub width: i32,
    pub height: i32,
}

impl ImageKind {
    pub fn new(layout: PixelLayout, width: i32, height: i32) -> Self {
        Self {
            layout,
            width,
            height,
        }
    }
    pub fn dims(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    pub fn color_space(&self) -> ColorSpace {
        self.layout.color_space
    }

    /// Returns a copy of this `ImageKind` with `layout` swapped in (extent
    /// preserved) — `Convert`'s `output_spec` (§6.1).
    pub fn with_layout(&self, layout: PixelLayout) -> Self {
        Self {
            layout,
            width: self.width,
            height: self.height,
        }
    }

    /// Replaces `layout` in place with the `count`-band derivation of the
    /// current layout (`docs/native-color-management.md` §6.3).
    pub fn set_band_count(&mut self, count: i32) {
        self.layout = layout_with_bands(self.layout, count as usize);
    }
}

impl AnyKind for ImageKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        let bpp = self.layout.bytes_per_pixel() as u64;
        match wu {
            WorkUnit::Region(r) => (r.w.max(0) as u64) * (r.h.max(0) as u64) * bpp,
            _ => 0,
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        // `PixelLayout` is `Hash`, but `dyn Hasher` doesn't implement the
        // `Hasher` bound `Hash::hash` requires; a compact Debug proxy is fine
        // for this datatype's identity in the POC.
        state.write(format!("{:?}", self.layout).as_bytes());
        state.write_i32(self.width);
        state.write_i32(self.height);
    }
}

impl Kind for ImageKind {
    type WorkUnit = Region;
}

impl GpuView for ImageKind {
    /// Decode wrapper: kernels read the image through a `CodecRegion` that
    /// unpacks the raw storage to working `float4` on `read`. Storage-only —
    /// the codec knows only `(storage, channel_count)`, never the color model.
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

    /// The image codec sandwich: the kernel writes working `float4` to an
    /// `RWRegion` scratch; afterwards an `RWCodecRegion` encodes it back into
    /// the raw storage target. Both this encode and the `input` decode are the
    /// image's own concern — the emitter and ops know nothing of codecs.
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

impl VipsBand for ImageKind {
    fn band_format(&self) -> i32 {
        self.layout.storage.into_vips_band_format()
    }
}

/// What the user holds. Aliased over the generic core; per-backend.
pub type Image2D<B> = Data<ImageKind, B>;

// ── Spatial ergonomics (Region-shaped only) ──────────────────────────────────

impl Image2D<VipsBackend> {
    pub fn open(path: &str) -> Result<Self, crate::error::Error> {
        let source = Arc::new(FileImageSource::new(path)?);
        Ok(crate::node::Data::from_source(source, Arc::new(())))
    }
}

impl<B: Backend> Image2D<B> {
    pub fn width(&self) -> i32 {
        self.spec.width
    }
    pub fn height(&self) -> i32 {
        self.spec.height
    }
    pub fn layout(&self) -> PixelLayout {
        self.spec.layout
    }
    pub fn color_space(&self) -> ColorSpace {
        self.spec.color_space()
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
        let interp = unsafe { crate::ffi::vips_image_get_interpretation(ptr) };

        let storage = Storage::from_vips_band_format(format_raw, bands);
        let (model, alpha, default_cs) =
            crate::color::space::from_vips_interpretation(interp, bands);

        // For RGB-family models, refine the default color space via an
        // embedded ICC profile (matrix/TRC profiles only — `docs/native-
        // color-management.md` §7). Gray/Lab/Xyz/Cmyk/etc keep their
        // interpretation-derived default.
        let color_space = if matches!(model, ColorModel::Rgb | ColorModel::ScRgb) {
            let icc_name = c"icc-profile-data";
            let mut data: *const std::ffi::c_void = std::ptr::null();
            let mut len: usize = 0;
            let has_icc = unsafe {
                crate::ffi::vips_image_get_blob(ptr, icc_name.as_ptr(), &mut data, &mut len) == 0
            };
            if has_icc && !data.is_null() && len > 0 {
                let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, len) };
                let classification = crate::color::detect::IccClassification::classify_icc_profile(bytes);
                classification.color_space.unwrap_or(default_cs)
            } else {
                default_cs
            }
        } else {
            default_cs
        };

        unsafe { crate::ffi::g_object_unref(ptr as *mut std::ffi::c_void) };

        let layout = PixelLayout {
            storage,
            model,
            alpha,
            color_space,
        };
        let spec = Arc::new(ImageKind::new(layout, width, height));

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

        let storage = match handle.params().output_bps {
            16 => Storage::U16,
            _ => Storage::U8,
        };

        let spec = Arc::new(ImageKind::new(
            PixelLayout {
                storage,
                model: ColorModel::Rgb,
                alpha: AlphaState::Straight,
                color_space: ColorSpace::SRGB,
            },
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
        Ok(crate::node::Data::from_source(source, Arc::new(())))
    }
}

// ── VIPS to GPU Bridge ────────────────────────────────────────────────────────

use crate::buffer::Buffer;
use crate::io::Source;

/// A GPU Source that reads from a Vips pipeline.
/// This enforces the boundary invariant: data enters the GPU ONLY through a Source.
pub struct VipsImageSource {
    /// The Vips pipeline root to read from. Every GPU frame begins here.
    pub vips_img: Image2D<VipsBackend>,
}

impl VipsImageSource {
    /// Creates a GPU source that reads from a Vips pipeline.
    pub fn new(vips_img: Image2D<VipsBackend>) -> Self {
        Self { vips_img }
    }
}

/// LOD-space dimensions of a full-res `w x h` image at `lod` (floor, matching
/// `vips shrink`). `Lod(0)` returns the full size.
fn lod_dims(w: i32, h: i32, lod: crate::work_unit::Lod) -> (i32, i32) {
    let s = lod.scale_factor() as i32;
    ((w / s).max(1), (h / s).max(1))
}

/// Clamps a requested region to `[0,0]..[w,h]` — edge tiles legitimately
/// overshoot the image bounds, but `vips_extract_area` rejects out-of-bounds
/// rects, and the GPU buffer can only ever hold the clamped extent.
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
        ctx: &crate::backend::gpu::GpuContext,
        wu: &Region,
    ) -> Result<Buffer<GpuBackend>, crate::error::Error> {
        use wgpu::util::DeviceExt;

        // LOD-aware fetch: when the demand is at `lod > 0`, downsample in VIPS
        // (shrink-on-load / streaming on CPU) so the GPU never decodes, uploads,
        // or processes full resolution for a coarse mip. The demand's region
        // coordinates are already in LOD space. This is the source honoring the
        // LOD demand dimension — no GPU `Shrink` op, no full-res slab.
        let scale = wu.lod.scale_factor();
        let (lod_w, lod_h) = lod_dims(self.spec().width, self.spec().height, wu.lod);
        let src = if scale > 1 {
            self.vips_img.shrink(scale as f64, scale as f64, None)
        } else {
            self.vips_img.clone()
        };

        // 1. Materialize the (downsampled) VIPS pipeline. VIPS streams the
        // full-res decode internally; only the reduced image is realized.
        let vips_buffer =
            src.materialize(Region::full((lod_w, lod_h), crate::work_unit::Lod(0)))?;

        // 2. Crop to the demanded tile (LOD-space coords) before uploading.
        let region = clamp_region(wu, lod_w, lod_h);
        let mut crop = crate::backend::vips::gobject::VipsGObject::new(b"extract_area\0")?;
        crop.set_image("input", vips_buffer.payload.ptr);
        crop.set_int("left", region.x);
        crop.set_int("top", region.y);
        crop.set_int("width", region.w);
        crop.set_int("height", region.h);
        let cropped: crate::backend::vips::VipsHandle = crop.run()?;

        // 3. Pull the raw bytes from the cropped tile
        let mut size: usize = 0;
        let ptr =
            unsafe { crate::ffi::vips_image_write_to_memory(cropped.ptr, &mut size as *mut usize) };
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
                // The fetched buffer is the demanded tile (clamped to the
                // LOD-space image bounds), tightly packed.
                let (lod_w, lod_h) = lod_dims(self.spec().width, self.spec().height, region.lod);
                let clamped = clamp_region(region, lod_w, lod_h);
                let pad_x = clamped.x - region.x;
                let pad_y = clamped.y - region.y;
                println!("VipsImageSource: clamped={:?}, region={:?}, pad_x={}, pad_y={}", clamped, region, pad_x, pad_y);
                let geom = RegionParams::padded(clamped.w as u32, 0, 0, clamped.w as u32, clamped.h as u32, pad_x, pad_y);
                cx.input(
                    self.spec().input(),
                    geom.into_block("region_in_{slot}"),
                    buf.payload,
                );
            }
            Err(e) => cx.fail(e),
        }
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        // Identity is based on the Vips pipeline root.
        state.write_usize(crate::node::NodeId::of(&self.vips_img.root).0);
    }
}

/// A GPU source backed by a constant f32 array (used for test images, kernels, etc.).
pub struct GpuConstantSource {
    pub spec: Arc<ImageKind>,
    /// Raw float data, tightly packed: `[r,g,b, ...]` or `[g, ...]` for grayscale.
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
        let bytes = unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const u8, self.data.len() * 4)
        };
        let wgpu_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
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
                let geom = crate::backend::gpu::view::RegionParams::tight(
                    self.spec.width,
                    self.spec.height,
                );
                cx.input(
                    self.spec.input(),
                    geom.into_block("region_in_{slot}"),
                    buf.payload,
                );
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
        let spec = Arc::new(ImageKind::new(
            PixelLayout {
                storage: Storage::F32,
                model: ColorModel::Gray,
                alpha: AlphaState::None,
                color_space: ColorSpace::SRGB,
            },
            width,
            height,
        ));
        let src = GpuConstantSource {
            spec: spec.clone(),
            data: data.to_vec(),
        };
        crate::node::Data::from_source(Arc::new(src), ctx)
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

/// A Vips source that reads from a RAW pipeline — bridges the RawBackend
/// into the VipsBackend so RAW images can be processed with Vips operations.
pub struct RawImageSource {
    pub raw_img: Image2D<crate::backend::raw::RawBackend>,
}

impl RawImageSource {
    /// Creates a new Vips source backed by a RAW image.
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
        state.write_usize(crate::node::NodeId::of(&self.raw_img.root).0);
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

/// The viewport exit: extracts the materialized region as a still
/// GPU-resident `Arc<GpuBuffer>` (clones the Arc, no download). Callers
/// (e.g. the tile fetcher) `copy_buffer_to_texture` it directly on the
/// shared device.
pub struct GpuBufferTarget;

impl Target<ImageKind, GpuBackend> for GpuBufferTarget {
    type Out = Arc<crate::backend::gpu::GpuBuffer>;

    fn extract(
        &self,
        buf: &Buffer<GpuBackend>,
        _wu: &Region,
        _ctx: &<GpuBackend as Backend>::Ctx,
    ) -> Result<Self::Out, crate::error::Error> {
        Ok(buf.payload.clone())
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
