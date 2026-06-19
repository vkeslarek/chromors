use crate::prelude::*;
use chromors_core::color::detect::IccClassification;

impl VipsBand for ImageKind {
    fn band_format(&self) -> i32 {
        self.layout.storage.into_vips_band_format()
    }
}

// ── FileImageSource ──────────────────────────────────────────────────────────

pub struct FileImageSource {
    spec: Arc<ImageKind>,
    pub filename: String,
}

impl FileImageSource {
    pub fn new(filename: &str) -> Result<Self, Error> {
        ensure_init();
        let c =
            std::ffi::CString::new(filename).map_err(|_| Error::Vips("invalid filename".into()))?;
        let ptr = unsafe {
            ffi::vips_image_new_from_file(c.as_ptr(), std::ptr::null_mut::<std::ffi::c_void>())
        };
        if ptr.is_null() {
            return Err(Error::Vips(vips_error()));
        }
        let width = unsafe { ffi::vips_image_get_width(ptr) };
        let height = unsafe { ffi::vips_image_get_height(ptr) };
        let bands = unsafe { ffi::vips_image_get_bands(ptr) };
        let format_raw = unsafe { ffi::vips_image_get_format(ptr) };
        let interp = unsafe { ffi::vips_image_get_interpretation(ptr) };
        let storage = Storage::from_vips_band_format(format_raw, bands);
        let (model, alpha, default_cs) = crate::space::from_vips_interpretation(interp, bands);
        let color_space = if matches!(model, ColorModel::Rgb | ColorModel::ScRgb) {
            let icc_name = c"icc-profile-data";
            let mut data: *const std::ffi::c_void = std::ptr::null();
            let mut len: usize = 0;
            let has_icc = unsafe {
                ffi::vips_image_get_blob(ptr, icc_name.as_ptr(), &mut data, &mut len) == 0
            };
            if has_icc && !data.is_null() && len > 0 {
                let bytes = unsafe { std::slice::from_raw_parts(data as *const u8, len) };
                IccClassification::classify_icc_profile(bytes)
                    .color_space
                    .unwrap_or(default_cs)
            } else {
                default_cs
            }
        } else {
            default_cs
        };
        unsafe { ffi::g_object_unref(ptr as *mut std::ffi::c_void) };
        let spec = Arc::new(ImageKind::new(
            PixelLayout {
                storage,
                model,
                alpha,
                color_space,
            },
            width,
            height,
        ));
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
    fn fetch(&self, _ctx: &(), _wu: &Region) -> Result<Buffer<VipsBackend>, Error> {
        let c = std::ffi::CString::new(self.filename.as_str()).unwrap();
        let ptr = unsafe {
            ffi::vips_image_new_from_file(c.as_ptr(), std::ptr::null_mut::<std::ffi::c_void>())
        };
        if ptr.is_null() {
            return Err(Error::Vips(vips_error()));
        }
        Ok(Buffer {
            payload: Arc::new(VipsHandle { ptr }),
            spec: self.spec.clone(),
        })
    }
    fn lower(&self, cx: &mut VipsBuilder) {
        let c = std::ffi::CString::new(self.filename.as_str()).unwrap();
        let ptr = unsafe {
            ffi::vips_image_new_from_file(c.as_ptr(), std::ptr::null_mut::<std::ffi::c_void>())
        };
        cx.emit(VipsHandle { ptr });
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write(self.filename.as_bytes());
    }
}

// ── RamImageSource ───────────────────────────────────────────────────────────

pub struct RamImageSource {
    pub spec: Arc<ImageKind>,
    pub data: Vec<u8>,
}

impl Source<VipsBackend> for RamImageSource {
    type Kind = ImageKind;
    fn spec(&self) -> Arc<ImageKind> {
        self.spec.clone()
    }
    fn fetch(&self, _ctx: &(), _wu: &Region) -> Result<Buffer<VipsBackend>, Error> {
        let vips_format = self.spec.layout.storage.into_vips_band_format();
        let bands = self.spec.layout.channel_count() as i32;
        let ptr = unsafe {
            ffi::vips_image_new_from_memory_copy(
                self.data.as_ptr() as *const std::ffi::c_void,
                self.data.len(),
                self.spec.width,
                self.spec.height,
                bands,
                vips_format,
            )
        };
        if ptr.is_null() {
            return Err(Error::Vips(vips_error()));
        }
        Ok(Buffer {
            payload: Arc::new(VipsHandle { ptr }),
            spec: self.spec.clone(),
        })
    }
    fn lower(&self, cx: &mut VipsBuilder) {
        let region = Region::full((self.spec.width, self.spec.height), Lod(0));
        let buf = self.fetch(&(), &region).unwrap();
        cx.emit((*buf.payload).clone());
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_usize(self.data.len());
        if self.data.len() >= 16 {
            state.write(&self.data[..16]);
        }
    }
}

// ── Target ───────────────────────────────────────────────────────────────────

impl Target<ImageKind, VipsBackend> for RamImageTarget {
    type Out = Vec<u8>;
    fn extract(
        &self,
        buf: &Buffer<VipsBackend>,
        _wu: &Region,
        _ctx: &(),
    ) -> Result<Self::Out, Error> {
        let mut size: usize = 0;
        let ptr =
            unsafe { ffi::vips_image_write_to_memory(buf.payload.ptr, &mut size as *mut usize) };
        if ptr.is_null() {
            return Err(Error::Vips(vips_error()));
        }
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, size) };
        let vec = slice.to_vec();
        unsafe { ffi::g_free(ptr as *mut std::ffi::c_void) };
        Ok(vec)
    }
}

pub trait VipsImageExt {
    fn open(path: &str) -> Result<Image2D<VipsBackend>, Error>;
    fn from_bytes(
        data: Vec<u8>,
        width: i32,
        height: i32,
        layout: PixelLayout,
    ) -> Image2D<VipsBackend>;
}

impl VipsImageExt for Image2D<VipsBackend> {
    fn open(path: &str) -> Result<Image2D<VipsBackend>, Error> {
        let source = Arc::new(FileImageSource::new(path)?);
        Ok(Data::from_source(source, Arc::new(())))
    }
    fn from_bytes(
        data: Vec<u8>,
        width: i32,
        height: i32,
        layout: PixelLayout,
    ) -> Image2D<VipsBackend> {
        let spec = Arc::new(ImageKind::new(layout, width, height));
        let src = RamImageSource {
            spec: spec.clone(),
            data,
        };
        Data::from_source(Arc::new(src), Arc::new(()))
    }
}
