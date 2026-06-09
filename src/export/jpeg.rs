//! JPEG-family export configuration (JPEG, JPEG-XL, JPEG 2000, Ultra HDR).

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum JpegCodec {
    #[default]
    Jpeg,
    Jxl,
    Jp2k,
    UltraHdr,
}

impl std::fmt::Display for JpegCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
impl JpegCodec {
    pub fn file_extension(&self) -> &'static str {
        match self {
            JpegCodec::Jpeg | JpegCodec::UltraHdr => "jpg",
            JpegCodec::Jxl => "jxl",
            JpegCodec::Jp2k => "jp2",
        }
    }
    pub fn label(&self) -> &'static str {
        match self {
            JpegCodec::Jpeg => "JPEG",
            JpegCodec::Jxl => "JPEG-XL",
            JpegCodec::Jp2k => "JPEG 2000",
            JpegCodec::UltraHdr => "Ultra HDR",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum JpegSubsample {
    #[default]
    Auto,
    On,
    Off,
}

impl std::fmt::Display for JpegSubsample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
impl JpegSubsample {
    pub fn label(&self) -> &'static str {
        match self {
            JpegSubsample::Auto => "Auto",
            JpegSubsample::On => "4:2:0",
            JpegSubsample::Off => "4:4:4",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct JpegExportConfig {
    pub codec: JpegCodec,
    pub quality: u8,
    pub progressive: bool,
    pub optimize_coding: bool,
    pub subsample_mode: JpegSubsample,
    // JPEG-XL
    pub jxl_lossless: bool,
    pub jxl_effort: u8,
    pub jxl_distance: f32,
    pub jxl_tier: u8,
    pub jxl_interlace: bool,
    // JPEG 2000
    pub jp2k_lossless: bool,
    pub jp2k_tile_width: u32,
    pub jp2k_tile_height: u32,
}

impl Default for JpegExportConfig {
    fn default() -> Self {
        Self {
            codec: JpegCodec::default(),
            quality: 90,
            progressive: false,
            optimize_coding: false,
            subsample_mode: JpegSubsample::default(),
            jxl_lossless: false,
            jxl_effort: 7,
            jxl_distance: 1.0,
            jxl_tier: 0,
            jxl_interlace: false,
            jp2k_lossless: false,
            jp2k_tile_width: 512,
            jp2k_tile_height: 512,
        }
    }
}

impl JpegExportConfig {
    pub fn to_vips_options(&self) -> String {
        match self.codec {
            JpegCodec::Jpeg | JpegCodec::UltraHdr => {
                let mut opts = Vec::new();
                if self.quality != 75 {
                    opts.push(format!("Q={}", self.quality));
                }
                if self.progressive {
                    opts.push("interlace".into());
                }
                if self.optimize_coding {
                    opts.push("optimize_coding".into());
                }
                match self.subsample_mode {
                    JpegSubsample::Auto => {}
                    JpegSubsample::On => opts.push("subsample_mode=on".into()),
                    JpegSubsample::Off => opts.push("subsample_mode=off".into()),
                }
                if opts.is_empty() {
                    String::new()
                } else {
                    format!("[{}]", opts.join(","))
                }
            }
            JpegCodec::Jxl => {
                let mut opts = Vec::new();
                if self.jxl_lossless {
                    opts.push("lossless".into());
                } else if self.quality != 75 {
                    opts.push(format!("Q={}", self.quality));
                }
                if self.jxl_effort != 7 {
                    opts.push(format!("effort={}", self.jxl_effort));
                }
                if (self.jxl_distance - 1.0).abs() > f32::EPSILON {
                    opts.push(format!("distance={}", self.jxl_distance));
                }
                if self.jxl_tier > 0 {
                    opts.push(format!("tier={}", self.jxl_tier));
                }
                if self.jxl_interlace {
                    opts.push("interlace".into());
                }
                if opts.is_empty() {
                    String::new()
                } else {
                    format!("[{}]", opts.join(","))
                }
            }
            JpegCodec::Jp2k => {
                let mut opts = Vec::new();
                if self.jp2k_lossless {
                    opts.push("lossless".into());
                } else if self.quality != 48 {
                    opts.push(format!("Q={}", self.quality));
                }
                match self.subsample_mode {
                    JpegSubsample::Auto => {}
                    JpegSubsample::On => opts.push("subsample_mode=on".into()),
                    JpegSubsample::Off => opts.push("subsample_mode=off".into()),
                }
                opts.push(format!(
                    "tile_width={},tile_height={}",
                    self.jp2k_tile_width, self.jp2k_tile_height
                ));
                format!("[{}]", opts.join(","))
            }
        }
    }

    pub fn file_extension(&self) -> &'static str {
        self.codec.file_extension()
    }
}
