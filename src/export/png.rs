//! PNG export configuration.

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PngColorType {
    #[default]
    Rgba,
    Rgb,
    Gray,
    GrayAlpha,
}

impl std::fmt::Display for PngColorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}
impl PngColorType {
    pub fn label(&self) -> &'static str {
        match self {
            PngColorType::Rgba => "RGBA",
            PngColorType::Rgb => "RGB",
            PngColorType::Gray => "Grayscale",
            PngColorType::GrayAlpha => "Grayscale + Alpha",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PngBitDepth {
    Eight,
    #[default]
    Sixteen,
}

impl std::fmt::Display for PngBitDepth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}
impl PngBitDepth {
    pub fn label(&self) -> &'static str {
        match self {
            PngBitDepth::Eight => "8-bit",
            PngBitDepth::Sixteen => "16-bit",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PngCompression {
    None,
    Fast,
    #[default]
    Default,
    Best,
    Level(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PngFilter {
    #[default]
    Adaptive,
    None,
    Sub,
    Up,
    Average,
    Paeth,
}

impl std::fmt::Display for PngFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}
impl PngFilter {
    pub fn label(&self) -> &'static str {
        match self {
            PngFilter::Adaptive => "Adaptive",
            PngFilter::None => "None",
            PngFilter::Sub => "Sub",
            PngFilter::Up => "Up",
            PngFilter::Average => "Average",
            PngFilter::Paeth => "Paeth",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PngInterlace {
    #[default]
    None,
    Adam7,
}

impl std::fmt::Display for PngInterlace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}
impl PngInterlace {
    pub fn label(&self) -> &'static str {
        match self {
            PngInterlace::None => "None",
            PngInterlace::Adam7 => "Adam7",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PngExportConfig {
    pub color_type: PngColorType,
    pub bit_depth: PngBitDepth,
    pub compression: PngCompression,
    pub filter: PngFilter,
    pub interlace: PngInterlace,
    pub palette: bool,
    pub palette_quality: u8,
    pub dither: f32,
    pub effort: u8,
    pub embed_dpi: bool,
    pub embed_icc: bool,
}

impl PngExportConfig {
    pub fn to_vips_options(&self) -> String {
        let mut opts = Vec::new();
        let level = match self.compression {
            PngCompression::None => 0,
            PngCompression::Fast => 2,
            PngCompression::Default => 6,
            PngCompression::Best => 9,
            PngCompression::Level(l) => l.min(9),
        };
        if level != 6 {
            opts.push(format!("compression={level}"));
        }
        if self.interlace == PngInterlace::Adam7 {
            opts.push("interlace".into());
        }
        let filter_val: u32 = match self.filter {
            PngFilter::None => 0x08,
            PngFilter::Sub => 0x10,
            PngFilter::Up => 0x20,
            PngFilter::Average => 0x40,
            PngFilter::Paeth => 0x80,
            PngFilter::Adaptive => 0xF8,
        };
        opts.push(format!("filter={filter_val}"));
        if self.palette {
            opts.push("palette".into());
            if self.palette_quality != 100 {
                opts.push(format!("Q={}", self.palette_quality));
            }
            if (self.dither - 1.0).abs() > f32::EPSILON {
                opts.push(format!("dither={}", self.dither));
            }
            if self.effort > 0 {
                opts.push(format!("effort={}", self.effort));
            }
        }
        format!("[{}]", opts.join(","))
    }
}
