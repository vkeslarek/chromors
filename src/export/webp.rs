//! WebP export configuration.

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum WebPPreset {
    #[default]
    Default,
    Picture,
    Photo,
    Drawing,
    Icon,
    Text,
}

impl std::fmt::Display for WebPPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
impl WebPPreset {
    pub fn vips_name(&self) -> &'static str {
        match self {
            WebPPreset::Default => "default",
            WebPPreset::Picture => "picture",
            WebPPreset::Photo => "photo",
            WebPPreset::Drawing => "drawing",
            WebPPreset::Icon => "icon",
            WebPPreset::Text => "text",
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            WebPPreset::Default => "Default",
            WebPPreset::Picture => "Picture",
            WebPPreset::Photo => "Photo",
            WebPPreset::Drawing => "Drawing",
            WebPPreset::Icon => "Icon",
            WebPPreset::Text => "Text",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WebPExportConfig {
    pub lossless: bool,
    pub quality: f32,
    pub preset: WebPPreset,
    pub effort: u8,
    pub alpha_q: u8,
    pub smart_subsample: bool,
    pub near_lossless: bool,
}

impl Default for WebPExportConfig {
    fn default() -> Self {
        Self {
            lossless: false,
            quality: 75.0,
            preset: WebPPreset::default(),
            effort: 4,
            alpha_q: 100,
            smart_subsample: false,
            near_lossless: false,
        }
    }
}

impl WebPExportConfig {
    pub fn to_vips_options(&self) -> String {
        let mut opts = Vec::new();
        if self.lossless {
            opts.push("lossless".into());
        } else if (self.quality - 75.0).abs() > f32::EPSILON {
            opts.push(format!("Q={}", self.quality as u8));
        }
        if self.preset != WebPPreset::Default {
            opts.push(format!("preset={}", self.preset.vips_name()));
        }
        if self.effort != 4 {
            opts.push(format!("effort={}", self.effort));
        }
        if self.alpha_q != 100 {
            opts.push(format!("alpha_q={}", self.alpha_q));
        }
        if self.smart_subsample {
            opts.push("smart_subsample".into());
        }
        if self.near_lossless {
            opts.push("near_lossless".into());
        }
        if opts.is_empty() {
            String::new()
        } else {
            format!("[{}]", opts.join(","))
        }
    }
}
