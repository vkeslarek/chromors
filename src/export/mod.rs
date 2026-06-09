//! Export configuration types for all supported image formats.

pub mod avif;
pub mod bmp;
pub mod gif;
pub mod jpeg;
pub mod png;
pub mod tiff;
pub mod webp;

use crate::error::Error;

#[derive(Debug, Clone)]
pub enum ExportConfig {
    Png(png::PngExportConfig),
    Tiff(tiff::TiffExportConfig),
    Jpeg(jpeg::JpegExportConfig),
    WebP(webp::WebPExportConfig),
    Avif(avif::AvifExportConfig),
    Gif(gif::GifExportConfig),
    Bmp(bmp::BmpExportConfig),
}

impl ExportConfig {
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

impl crate::data::image::Image<crate::backend::vips::VipsBackend> {
    pub fn save_with_config(
        &self,
        filename: &str,
        config: &ExportConfig,
    ) -> Result<(), crate::error::Error> {
        let options = config.to_vips_options();
        let full = format!("{filename}{options}");
        let c = std::ffi::CString::new(full.as_str())
            .map_err(|_| Error::Vips("invalid filename".into()))?;
        if unsafe {
            crate::libvips_ffi::vips_image_write_to_file(
                self.vips_ptr(),
                c.as_ptr(),
                std::ptr::null::<std::ffi::c_void>(),
            )
        } != 0
        {
            return Err(Error::Vips(crate::backend::vips::vips_error()));
        }
        Ok(())
    }
}
