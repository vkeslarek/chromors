//! AVIF/HEIF export configuration.

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AvifCompression {
    #[default]
    Av1,
    Hevc,
    Avc,
    Jpeg,
}

impl std::fmt::Display for AvifCompression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
impl AvifCompression {
    pub fn vips_name(&self) -> &'static str {
        match self {
            AvifCompression::Av1 => "av1",
            AvifCompression::Hevc => "hevc",
            AvifCompression::Avc => "avc",
            AvifCompression::Jpeg => "jpeg",
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            AvifCompression::Av1 => "AV1",
            AvifCompression::Hevc => "HEVC",
            AvifCompression::Avc => "AVC",
            AvifCompression::Jpeg => "JPEG",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum AvifBitDepth {
    Eight,
    Ten,
    #[default]
    Twelve,
}

impl std::fmt::Display for AvifBitDepth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
impl AvifBitDepth {
    pub fn vips_val(&self) -> u8 {
        match self {
            AvifBitDepth::Eight => 8,
            AvifBitDepth::Ten => 10,
            AvifBitDepth::Twelve => 12,
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            AvifBitDepth::Eight => "8-bit",
            AvifBitDepth::Ten => "10-bit",
            AvifBitDepth::Twelve => "12-bit",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AvifExportConfig {
    pub quality: u8,
    pub lossless: bool,
    pub compression: AvifCompression,
    pub effort: u8,
    pub subsample_mode: super::jpeg::JpegSubsample,
    pub bitdepth: AvifBitDepth,
}

impl Default for AvifExportConfig {
    fn default() -> Self {
        Self {
            quality: 50,
            lossless: false,
            compression: AvifCompression::default(),
            effort: 4,
            subsample_mode: super::jpeg::JpegSubsample::default(),
            bitdepth: AvifBitDepth::default(),
        }
    }
}

impl AvifExportConfig {
    pub fn to_vips_options(&self) -> String {
        let mut opts = Vec::new();
        if self.lossless {
            opts.push("lossless".into());
        } else if self.quality != 50 {
            opts.push(format!("Q={}", self.quality));
        }
        if self.compression != AvifCompression::Av1 {
            opts.push(format!("compression={}", self.compression.vips_name()));
        }
        if self.effort != 4 {
            opts.push(format!("effort={}", self.effort));
        }
        match self.subsample_mode {
            super::jpeg::JpegSubsample::Auto => {}
            super::jpeg::JpegSubsample::On => opts.push("subsample_mode=on".into()),
            super::jpeg::JpegSubsample::Off => opts.push("subsample_mode=off".into()),
        }
        if self.bitdepth != AvifBitDepth::Twelve {
            opts.push(format!("bitdepth={}", self.bitdepth.vips_val()));
        }
        if opts.is_empty() {
            String::new()
        } else {
            format!("[{}]", opts.join(","))
        }
    }
}
