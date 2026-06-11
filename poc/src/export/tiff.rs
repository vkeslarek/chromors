//! TIFF export configuration.

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TiffColorType {
    #[default]
    Rgb,
    Rgba,
    Gray,
    GrayscaleAlpha,
    Cmyk,
    CmykAlpha,
    CieLab,
}
impl std::fmt::Display for TiffColorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
impl TiffColorType {
    pub fn label(&self) -> &'static str {
        match self {
            TiffColorType::Rgb => "RGB",
            TiffColorType::Rgba => "RGBA",
            TiffColorType::Gray => "Grayscale",
            TiffColorType::GrayscaleAlpha => "Grayscale + Alpha",
            TiffColorType::Cmyk => "CMYK",
            TiffColorType::CmykAlpha => "CMYK + Alpha",
            TiffColorType::CieLab => "CIE Lab",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TiffBitDepth {
    Eight,
    #[default]
    Sixteen,
    ThirtyTwo,
}
impl std::fmt::Display for TiffBitDepth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
impl TiffBitDepth {
    pub fn label(&self) -> &'static str {
        match self {
            TiffBitDepth::Eight => "8-bit",
            TiffBitDepth::Sixteen => "16-bit",
            TiffBitDepth::ThirtyTwo => "32-bit (Float)",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TiffCompression {
    None,
    Deflate { level: u8, predictor: TiffPredictor },
    Lzw { predictor: TiffPredictor },
    PackBits,
    Jpeg,
}
impl Default for TiffCompression {
    fn default() -> Self {
        TiffCompression::Deflate {
            level: 6,
            predictor: TiffPredictor::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TiffPredictor {
    #[default]
    None,
    Horizontal,
    FloatingPoint,
}
impl std::fmt::Display for TiffPredictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
impl TiffPredictor {
    pub fn label(&self) -> &'static str {
        match self {
            TiffPredictor::None => "None",
            TiffPredictor::Horizontal => "Horizontal",
            TiffPredictor::FloatingPoint => "Floating Point",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TiffLayout {
    Strip { rows_per_strip: u32 },
    Tile { tile_width: u32, tile_height: u32 },
}
impl Default for TiffLayout {
    fn default() -> Self {
        TiffLayout::Strip { rows_per_strip: 8 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TiffByteOrder {
    #[default]
    LittleEndian,
    BigEndian,
}
impl std::fmt::Display for TiffByteOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}
impl TiffByteOrder {
    pub fn label(&self) -> &'static str {
        match self {
            TiffByteOrder::LittleEndian => "Little Endian",
            TiffByteOrder::BigEndian => "Big Endian",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum TiffVariant {
    #[default]
    Classic,
    BigTiff,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TiffExportConfig {
    pub color_type: TiffColorType,
    pub bit_depth: TiffBitDepth,
    pub compression: TiffCompression,
    pub layout: TiffLayout,
    pub byte_order: TiffByteOrder,
    pub tiff_variant: TiffVariant,
    pub pyramid: bool,
    pub lossy_quality: u8,
    pub multipage: bool,
    pub embed_dpi: bool,
    pub embed_icc: bool,
    pub embed_exif: bool,
}

impl TiffExportConfig {
    pub fn to_vips_options(&self) -> String {
        let mut opts = Vec::new();
        match self.compression {
            TiffCompression::None => {}
            TiffCompression::Deflate { level, predictor } => {
                opts.push(format!("compression=deflate,level={level}"));
                match predictor {
                    TiffPredictor::None => {}
                    TiffPredictor::Horizontal => opts.push("predictor=horizontal".into()),
                    TiffPredictor::FloatingPoint => opts.push("predictor=float".into()),
                }
            }
            TiffCompression::Lzw { predictor } => {
                opts.push("compression=lzw".into());
                match predictor {
                    TiffPredictor::None => {}
                    TiffPredictor::Horizontal => opts.push("predictor=horizontal".into()),
                    TiffPredictor::FloatingPoint => opts.push("predictor=float".into()),
                }
            }
            TiffCompression::PackBits => {
                opts.push("compression=packbits".into());
            }
            TiffCompression::Jpeg => {
                let q = if self.lossy_quality > 0 {
                    self.lossy_quality
                } else {
                    75
                };
                opts.push(format!("compression=jpeg,Q={q}"));
            }
        }
        match self.layout {
            TiffLayout::Tile {
                tile_width,
                tile_height,
            } => {
                opts.push(format!(
                    "tile,tile_width={tile_width},tile_height={tile_height}"
                ));
            }
            TiffLayout::Strip { .. } => {}
        }
        if self.pyramid {
            opts.push("pyramid".into());
        }
        if self.tiff_variant == TiffVariant::BigTiff {
            opts.push("bigtiff".into());
        }
        if opts.is_empty() {
            String::new()
        } else {
            format!("[{}]", opts.join(","))
        }
    }
}
