use std::ffi::{CStr, CString};

use crate::backend::vips::Source;
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::backend::vips::{FromVipsBandFormat, IntoVipsBandFormat, IntoVipsEnum, null};
use crate::backend::vips::{VipsBackend, VipsHandle};
use crate::data::image::Image;
use crate::error::Error;
use crate::generator::GenerateOperation;
use crate::libvips_ffi as ffi;

// ---- VipsBackend internal helpers -------------------------------------------------

impl Image<VipsBackend> {
    /// Raw libvips pointer. For vips-internal code only — never expose through
    /// the public API.
    pub(crate) fn vips_ptr(&self) -> *mut ffi::VipsImage {
        self.handle.ptr
    }

    pub(crate) fn from_vips_ptr(ptr: *mut ffi::VipsImage) -> Self {
        Image::from_handle(VipsHandle { ptr })
    }

    pub fn n_pages(&self) -> i32 {
        unsafe { ffi::vips_image_get_n_pages(self.handle.ptr) }
    }

    pub fn open_page(path: &str, page: i32) -> Result<Self, Error> {
        let ext = std::path::Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let loader: &[u8] = match ext.as_str() {
            "tif" | "tiff" => b"tiffload\0",
            "pdf" => b"pdfload\0",
            "webp" => b"webpload\0",
            "gif" => b"gifload\0",
            "avif" | "heif" | "heic" => b"heifload\0",
            "jxl" => b"jxlload\0",
            "jp2" | "j2k" | "jpc" | "j2c" => b"jp2kload\0",
            "exr" => b"openexrload\0",
            _ => b"VipsForeignLoad\0",
        };
        let c_path = CString::new(path).map_err(|_| Error::Vips("invalid path".into()))?;
        let mut op = VipsGObject::new(loader)?;
        op.set_string("filename", c_path.to_str().unwrap());
        op.set_int("page", page);
        op.run()
    }

    pub fn apply(op: VipsGObject) -> Result<Self, Error> {
        op.run()
    }

    /// Force the lazy vips pipeline to evaluate into a flat, owned RAM image.
    ///
    /// A composite/op chain kept lazy re-evaluates every time it is consumed
    /// (e.g. shrunk for an LOD). Flattening once turns subsequent reads into
    /// cheap resamples instead of full-pipeline re-evaluations.
    pub fn copy_to_memory(&self) -> Result<Self, Error> {
        unsafe {
            let ptr = ffi::vips_image_copy_memory(self.handle.ptr);
            if ptr.is_null() {
                return Err(Error::Vips(crate::backend::vips::vips_error()));
            }
            Ok(Image::from_vips_ptr(ptr))
        }
    }
}

// ---- VipsBackend I/O --------------------------------------------------------------

impl Image<VipsBackend> {
    /// Wraps a raw pixel buffer. `bands` is independent of the format's own
    /// channel count; validates `width*height*bands*sample_size` before copying.
    pub fn from_memory(
        buf: &[u8],
        width: i32,
        height: i32,
        bands: i32,
        format: crate::pixel::PixelFormat,
    ) -> Result<Self, Error> {
        if width <= 0 || height <= 0 || bands <= 0 {
            return Err(Error::Vips(
                "from_memory: width, height and bands must be positive".into(),
            ));
        }
        let sample_size = format.bytes_per_pixel() / format.channel_count();
        let needed = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(bands as usize))
            .and_then(|n| n.checked_mul(sample_size));
        match needed {
            Some(n) if buf.len() >= n => {}
            Some(n) => {
                return Err(Error::Vips(format!(
                    "from_memory: buffer too small ({} bytes, need {n})",
                    buf.len()
                )));
            }
            None => return Err(Error::Vips("from_memory: dimensions overflow".into())),
        }
        let ptr = unsafe {
            ffi::vips_image_new_from_memory_copy(
                buf.as_ptr() as *const _,
                buf.len(),
                width,
                height,
                bands,
                format.into_vips_band_format(),
            )
        };
        if ptr.is_null() {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(Image::from_vips_ptr(ptr))
    }

    /// Build a Vips image from raw row-major memory and tag it with
    /// `color_space`, setting the libvips interpretation accordingly (linear →
    /// scRGB, gamma → sRGB) so downstream GPU/color code reads the correct
    /// space. This is the tagging the former `FileCache::as_vips_image`
    /// performed; use it when wrapping an already-decoded pixel buffer.
    pub fn from_memory_with_cs(
        buf: &[u8],
        width: i32,
        height: i32,
        bands: i32,
        format: crate::pixel::PixelFormat,
        color_space: crate::color::space::ColorSpace,
    ) -> Result<Self, Error> {
        let img = Self::from_memory(buf, width, height, bands, format)?;
        // Correct interpretation: linear data → scRGB (28), gamma → sRGB (22).
        let vips_interp: i32 = if color_space.is_linear() { 28 } else { 22 };
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"copy\0")?;
        op.set_image("in", img.vips_ptr());
        op.set_int("interpretation", vips_interp);
        let tagged = op.run()?;
        let flat = tagged.copy_to_memory()?;
        flat.set_pixors_cs(color_space);
        Ok(flat)
    }

    pub fn save(&self, filename: &str) -> Result<(), Error> {
        let c = CString::new(filename).map_err(|_| Error::Vips("invalid filename".into()))?;
        if unsafe { ffi::vips_image_write_to_file(self.vips_ptr(), c.as_ptr(), null()) } != 0 {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(())
    }

    pub fn save_with_options(&self, filename: &str, options: &str) -> Result<(), Error> {
        let full = format!("{filename}{options}");
        let c = CString::new(full.as_str()).map_err(|_| Error::Vips("invalid filename".into()))?;
        if unsafe { ffi::vips_image_write_to_file(self.vips_ptr(), c.as_ptr(), null()) } != 0 {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(())
    }

    pub fn write_to_buffer(&self, suffix: &str) -> Result<Vec<u8>, Error> {
        let c = CString::new(suffix).map_err(|_| Error::Vips("invalid suffix".into()))?;
        let mut buf: *mut std::ffi::c_void = std::ptr::null_mut();
        let mut len: usize = 0;
        unsafe {
            if ffi::vips_image_write_to_buffer(
                self.vips_ptr(),
                c.as_ptr(),
                &mut buf,
                &mut len,
                null(),
            ) != 0
            {
                return Err(Error::Vips(crate::backend::vips::vips_error()));
            }
        }
        if buf.is_null() || len == 0 {
            return Err(Error::NullPtr);
        }
        let out = unsafe { std::slice::from_raw_parts(buf as *const u8, len).to_vec() };
        unsafe { ffi::g_free(buf) };
        Ok(out)
    }
}

// ---- VipsBackend properties -------------------------------------------------------

impl Image<VipsBackend> {
    pub fn width(&self) -> i32 {
        unsafe { ffi::vips_image_get_width(self.vips_ptr()) }
    }
    pub fn height(&self) -> i32 {
        unsafe { ffi::vips_image_get_height(self.vips_ptr()) }
    }
    pub fn bands(&self) -> i32 {
        unsafe { ffi::vips_image_get_bands(self.vips_ptr()) }
    }
    pub fn has_alpha(&self) -> bool {
        unsafe { ffi::vips_image_hasalpha(self.vips_ptr()) != 0 }
    }
    pub(crate) fn raw_format(&self) -> i32 {
        unsafe { ffi::vips_image_get_format(self.vips_ptr()) }
    }
    pub(crate) fn raw_interpretation(&self) -> i32 {
        unsafe { ffi::vips_image_get_interpretation(self.vips_ptr()) }
    }

    pub fn pixel_format(&self) -> crate::pixel::PixelFormat {
        crate::pixel::PixelFormat::from_vips_band_format(self.raw_format(), self.bands())
    }
}

// ---- VipsBackend color ------------------------------------------------------------

impl Image<VipsBackend> {
    /// Read the `pixors-cs` metadata integer (set for non-Vips-native color spaces).
    pub(crate) fn get_pixors_cs(&self) -> Option<crate::color::space::ColorSpace> {
        // Check for the field before reading to avoid Vips printing an error
        // message when the metadata is absent.
        let key = c"pixors-cs";
        let has_field = unsafe { ffi::vips_image_get_typeof(self.vips_ptr(), key.as_ptr()) != 0 };
        if !has_field {
            return None;
        }
        let mut out = 0i32;
        let ok = unsafe { ffi::vips_image_get_int(self.vips_ptr(), key.as_ptr(), &mut out) };
        if ok != 0 {
            return None;
        }
        crate::color::space::ColorSpace::from_pixors_id(out)
    }

    /// Write the `pixors-cs` metadata integer so downstream callers know the
    /// actual color space even when the Vips interpretation is only approximate
    /// (e.g., scRGB for ACES AP0 data).
    pub(crate) fn set_pixors_cs(&self, cs: crate::color::space::ColorSpace) {
        let id = cs.to_pixors_id();
        if id == 0 {
            return;
        }
        let key = c"pixors-cs";
        unsafe {
            ffi::vips_image_set_int(self.vips_ptr(), key.as_ptr(), id);
        }
    }
}

// ---- VipsBackend metadata ---------------------------------------------------------

impl Image<VipsBackend> {
    pub fn get_fields(&self) -> Vec<String> {
        unsafe {
            let ptrs = ffi::vips_image_get_fields(self.vips_ptr());
            if ptrs.is_null() {
                return vec![];
            }
            let mut result = Vec::new();
            for i in 0.. {
                let p = *ptrs.add(i);
                if p.is_null() {
                    break;
                }
                if let Ok(s) = CStr::from_ptr(p).to_str() {
                    result.push(s.to_string());
                }
            }
            ffi::g_strfreev(ptrs);
            result
        }
    }

    pub fn get_metadata(&self, name: &str) -> Result<String, Error> {
        let c_name = CString::new(name).map_err(|_| Error::Vips("bad field name".into()))?;
        unsafe {
            let mut out: *mut std::ffi::c_char = std::ptr::null_mut();
            if ffi::vips_image_get_as_string(self.vips_ptr(), c_name.as_ptr(), &mut out) != 0 {
                return Err(Error::Vips(crate::backend::vips::vips_error()));
            }
            if out.is_null() {
                return Err(Error::NullPtr);
            }
            let s = CStr::from_ptr(out).to_string_lossy().into_owned();
            ffi::g_free(out as *mut std::ffi::c_void);
            Ok(s)
        }
    }

    pub fn has_metadata(&self, name: &str) -> bool {
        let c_name = CString::new(name).unwrap();
        unsafe { ffi::vips_image_get_typeof(self.vips_ptr(), c_name.as_ptr()) != 0 }
    }

    pub fn set_metadata(&self, name: &str, value: &str) -> Result<(), Error> {
        let c_name = CString::new(name).map_err(|_| Error::Vips("bad field name".into()))?;
        let c_value = CString::new(value).map_err(|_| Error::Vips("bad field value".into()))?;
        unsafe {
            let mut val = std::mem::MaybeUninit::<ffi::GValue>::zeroed();
            ffi::g_value_init(val.as_mut_ptr(), (16 << 2) as ffi::GType);
            ffi::g_value_set_string(val.as_mut_ptr(), c_value.as_ptr());
            ffi::vips_image_set(self.vips_ptr(), c_name.as_ptr(), val.as_mut_ptr());
            ffi::g_value_unset(val.as_mut_ptr());
        }
        Ok(())
    }

    pub fn remove_metadata(&self, name: &str) -> bool {
        let c_name = CString::new(name).unwrap();
        unsafe { ffi::vips_image_remove(self.vips_ptr(), c_name.as_ptr()) != 0 }
    }

    pub fn extract_metadata(&self) -> Vec<crate::exif::Metadata> {
        crate::exif::extract(self.vips_ptr())
    }
}

// ---- VipsBackend draw / generate / misc -------------------------------------------

impl Image<VipsBackend> {
    pub fn draw<D: VipsOperation<Output = ()>>(&self, params: &D) -> Result<(), Error> {
        self.execute(params)
    }

    pub fn generate<G: GenerateOperation>(params: &G) -> Result<Self, Error> {
        let mut op = VipsGObject::new(G::op_name())?;
        params.build(&mut op);
        op.run_generator()
    }

    pub fn buildlut(&self) -> Result<Self, Error> {
        let mut op = VipsGObject::new(b"buildlut\0")?;
        op.set_image("in", self.vips_ptr());
        op.run()
    }

    pub fn bandjoin(&self, other: &Self) -> Result<Self, Error> {
        let mut images = [self.vips_ptr(), other.vips_ptr()];
        let mut out = std::ptr::null_mut();
        unsafe {
            if ffi::vips_bandjoin(images.as_mut_ptr(), &mut out, 2, null()) != 0 {
                return Err(Error::Vips(crate::backend::vips::vips_error()));
            }
        }
        Ok(Image::from_vips_ptr(out))
    }

    pub fn bandjoin_const(&self, constants: &[f64]) -> Result<Self, Error> {
        if constants.is_empty() {
            return Err(Error::Vips("bandjoin_const: empty constant array".into()));
        }
        let mut op = VipsGObject::new(b"bandjoin_const\0")?;
        op.set_image("in", self.vips_ptr());
        op.set_array_double("c", constants);
        op.run()
    }

    pub fn array_join(images: &[&Self], params: &ArrayJoinParams) -> Result<Self, Error> {
        if images.is_empty() {
            return Err(Error::Vips("array_join: empty image list".into()));
        }
        let mut op = VipsGObject::new(b"arrayjoin\0")?;
        let ptrs: Vec<*mut ffi::VipsImage> = images.iter().map(|i| i.vips_ptr()).collect();
        op.set_array_image("in", &ptrs);
        if let Some(v) = params.across {
            op.set_int("across", v);
        }
        if let Some(v) = params.shim {
            op.set_int("shim", v);
        }
        if let Some(v) = &params.background {
            op.set_array_double("background", v);
        }
        if let Some(v) = params.halign {
            op.set_int("halign", v.into_vips());
        }
        if let Some(v) = params.valign {
            op.set_int("valign", v.into_vips());
        }
        if let Some(v) = params.hspacing {
            op.set_int("hspacing", v);
        }
        if let Some(v) = params.vspacing {
            op.set_int("vspacing", v);
        }
        op.run()
    }

    pub fn switch(tests: &[&Self]) -> Result<Self, Error> {
        if tests.is_empty() {
            return Err(Error::Vips("switch: empty test list".into()));
        }
        let mut op = VipsGObject::new(b"switch\0")?;
        let ptrs: Vec<*mut ffi::VipsImage> = tests.iter().map(|i| i.vips_ptr()).collect();
        op.set_array_image("tests", &ptrs);
        op.run()
    }

    pub fn sum(images: &[&Self]) -> Result<Self, Error> {
        if images.is_empty() {
            return Err(Error::Vips("sum: empty image list".into()));
        }
        let mut op = VipsGObject::new(b"sum\0")?;
        let ptrs: Vec<*mut ffi::VipsImage> = images.iter().map(|i| i.vips_ptr()).collect();
        op.set_array_image("in", &ptrs);
        op.run()
    }

    pub fn band_rank(images: &[&Self], index: i32) -> Result<Self, Error> {
        if images.is_empty() {
            return Err(Error::Vips("band_rank: empty image list".into()));
        }
        let mut op = VipsGObject::new(b"bandrank\0")?;
        let ptrs: Vec<*mut ffi::VipsImage> = images.iter().map(|i| i.vips_ptr()).collect();
        op.set_array_image("in", &ptrs);
        op.set_int("index", index);
        op.run()
    }

    pub fn composite(
        images: &[&Self],
        modes: &[crate::operation::BlendMode],
        params: &CompositeParams,
    ) -> Result<Self, Error> {
        if images.is_empty() {
            return Err(Error::Vips("composite: empty image list".into()));
        }
        let mut op = VipsGObject::new(b"composite\0")?;
        let ptrs: Vec<*mut ffi::VipsImage> = images.iter().map(|i| i.vips_ptr()).collect();
        op.set_array_image("in", &ptrs);
        let mode_ints: Vec<i32> = modes.iter().map(|m| m.into_vips()).collect();
        op.set_array_int("mode", &mode_ints);
        if let Some(v) = &params.x {
            op.set_array_int("x", v);
        }
        if let Some(v) = &params.y {
            op.set_array_int("y", v);
        }
        if let Some(v) = params.compositing_space {
            op.set_int("compositing_space", v);
        }
        if let Some(v) = params.premultiplied {
            op.set_bool("premultiplied", v);
        }
        op.run()
    }

    pub fn thumbnail(filename: &str, width: i32, params: &ThumbnailParams) -> Result<Self, Error> {
        let mut op = VipsGObject::new(b"thumbnail\0")?;
        op.set_string("filename", filename);
        params.apply(&mut op, width);
        op.run()
    }

    pub fn thumbnail_buffer(
        buf: &[u8],
        width: i32,
        params: &ThumbnailParams,
    ) -> Result<Self, Error> {
        let mut op = VipsGObject::new(b"thumbnail_buffer\0")?;
        op.set_blob("buffer", buf);
        params.apply(&mut op, width);
        op.run()
    }

    pub fn thumbnail_source(
        source: &Source,
        width: i32,
        params: &ThumbnailParams,
    ) -> Result<Self, Error> {
        let mut op = VipsGObject::new(b"thumbnail_source\0")?;
        op.set_object("source", source.ptr as ffi::gpointer, unsafe {
            ffi::vips_source_get_type()
        });
        params.apply(&mut op, width);
        op.run()
    }
}

// ---- Parameter structs ------------------------------------------------------------

/// Optional parameters for [`Image::array_join`].
#[derive(Default)]
pub struct ArrayJoinParams {
    pub across: Option<i32>,
    pub shim: Option<i32>,
    pub background: Option<Vec<f64>>,
    pub halign: Option<crate::operation::Align>,
    pub valign: Option<crate::operation::Align>,
    pub hspacing: Option<i32>,
    pub vspacing: Option<i32>,
}

/// Optional parameters for [`Image::composite`].
#[derive(Default)]
pub struct CompositeParams {
    pub x: Option<Vec<i32>>,
    pub y: Option<Vec<i32>>,
    pub compositing_space: Option<i32>,
    pub premultiplied: Option<bool>,
}

/// Optional parameters shared by the `thumbnail*` loaders.
#[derive(Default)]
pub struct ThumbnailParams {
    pub height: Option<i32>,
    pub size: Option<crate::operation::Size>,
    pub no_rotate: Option<bool>,
    pub crop: Option<crate::operation::Interesting>,
    pub linear: Option<bool>,
    pub import_profile: Option<String>,
    pub export_profile: Option<String>,
    pub intent: Option<i32>,
    pub fail_on: Option<i32>,
}

impl ThumbnailParams {
    fn apply(&self, op: &mut VipsGObject, width: i32) {
        op.set_int("width", width);
        if let Some(v) = self.height {
            op.set_int("height", v);
        }
        if let Some(v) = self.size {
            op.set_int("size", v.into_vips());
        }
        if let Some(v) = self.no_rotate {
            op.set_bool("no_rotate", v);
        }
        if let Some(v) = self.crop {
            op.set_int("crop", v.into_vips());
        }
        if let Some(v) = self.linear {
            op.set_bool("linear", v);
        }
        if let Some(v) = &self.import_profile {
            op.set_string("import_profile", v);
        }
        if let Some(v) = &self.export_profile {
            op.set_string("export_profile", v);
        }
        if let Some(v) = self.intent {
            op.set_int("intent", v);
        }
        if let Some(v) = self.fail_on {
            op.set_int("fail_on", v);
        }
    }
}
