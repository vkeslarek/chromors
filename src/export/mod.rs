//! Export configuration types for all supported image formats.
//!
//! Each format has its own config struct (e.g. `PngExportConfig`) with
//! format-specific options. `ExportConfig` is the unified enum that wraps
//! all of them and converts to vips option strings via `to_vips_options()`.

pub mod avif;
pub mod bmp;
pub mod gif;
pub mod jpeg;
pub mod png;
pub mod tiff;
pub mod webp;

use crate::error::Error;

/// Unified export configuration for all supported output formats.
///
/// Each variant wraps the format-specific config. Use `to_vips_options()`
/// to serialize into a libvips-compatible option string.
#[derive(Debug, Clone)]
pub enum ExportConfig {
    /// PNG export options.
    Png(png::PngExportConfig),
    /// TIFF export options.
    Tiff(tiff::TiffExportConfig),
    /// JPEG export options.
    Jpeg(jpeg::JpegExportConfig),
    /// WebP export options.
    WebP(webp::WebPExportConfig),
    /// AVIF export options.
    Avif(avif::AvifExportConfig),
    /// GIF export options.
    Gif(gif::GifExportConfig),
    /// BMP export options.
    Bmp(bmp::BmpExportConfig),
}

impl ExportConfig {
    /// Serializes the config to a libvips option string (e.g. `"[Q=90,interlace=true]"`).
    pub fn to_vips_options(&self) -> String {
        match self {
            ExportConfig::Png(c) => c.to_vips_options(),
            ExportConfig::Tiff(c) => c.to_vips_options(),
            ExportConfig::Jpeg(c) => c.to_vips_options(),
            ExportConfig::WebP(c) => c.to_vips_options(),
            ExportConfig::Avif(c) => c.to_vips_options(),
            ExportConfig::Gif(c) => c.to_vips_options(),
            ExportConfig::Bmp(c) => c.to_vips_options(),
        }
    }
}

pub trait ExportExt {
    fn save_with_config(&self, filename: &str, config: &ExportConfig) -> Result<(), crate::error::Error>;
}

impl ExportExt for chromors_core::Image2D<chromors_backend_vips::VipsBackend> {
    fn save_with_config(
        &self,
        filename: &str,
        config: &ExportConfig,
    ) -> Result<(), crate::error::Error> {
        let (w, h) = (self.width() as i32, self.height() as i32);
        let wu = crate::work_unit::Region {
            x: 0,
            y: 0,
            w,
            h,
            lod: crate::work_unit::Lod(0),
        };
        let mat = self.materialize(wu)?;
        let options = config.to_vips_options();
        let full = format!("{filename}{options}");
        let c = std::ffi::CString::new(full.as_str())
            .map_err(|_| Error::Vips("invalid filename".into()))?;
        if unsafe {
            chromors_backend_vips::ffi::vips_image_write_to_file(
                mat.payload.ptr,
                c.as_ptr(),
                std::ptr::null::<std::ffi::c_void>(),
            )
        } != 0
        {
            return Err(Error::Vips(chromors_backend_vips::vips_error()));
        }
        Ok(())
    }
}
