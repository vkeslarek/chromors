//! BMP export configuration.

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BmpExportConfig {
    pub bitdepth: u8,
}

impl BmpExportConfig {
    pub fn to_vips_options(&self) -> String {
        if self.bitdepth > 0 {
            format!("[bitdepth={}]", self.bitdepth)
        } else {
            String::new()
        }
    }
}
