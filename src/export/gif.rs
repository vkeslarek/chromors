//! GIF export configuration.

#[derive(Debug, Clone, PartialEq)]
pub struct GifExportConfig {
    pub dither: f32,
    pub effort: u8,
    pub bitdepth: u8,
    pub interlace: bool,
}

impl Default for GifExportConfig {
    fn default() -> Self {
        Self {
            dither: 1.0,
            effort: 7,
            bitdepth: 8,
            interlace: false,
        }
    }
}

impl GifExportConfig {
    pub fn to_vips_options(&self) -> String {
        let mut opts = Vec::new();
        if (self.dither - 1.0).abs() > f32::EPSILON {
            opts.push(format!("dither={}", self.dither));
        }
        if self.effort != 7 {
            opts.push(format!("effort={}", self.effort));
        }
        if self.bitdepth != 8 {
            opts.push(format!("bitdepth={}", self.bitdepth));
        }
        if self.interlace {
            opts.push("interlace".into());
        }
        if opts.is_empty() {
            String::new()
        } else {
            format!("[{}]", opts.join(","))
        }
    }
}
