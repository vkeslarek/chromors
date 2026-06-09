/// A typed parameter value — mirrors Slang scalar, struct, and region types.
#[derive(Clone, Debug, PartialEq)]
pub enum Param {
    I32(i32),
    U32(u32),
    F32(f32),
    Struct {
        name: &'static str,
        fields: Vec<(&'static str, Param)>,
    },
    /// A buffer region — source or target buffer with layout metadata.
    Region {
        /// Slang variable name for this region binding.
        name: String,
        /// BufferRegion fields: stride, x, y, width, height.
        stride: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    },
}

impl Param {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        self.write_bytes(&mut b);
        b
    }

    fn write_bytes(&self, buf: &mut Vec<u8>) {
        match self {
            Param::I32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            Param::U32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            Param::F32(v) => buf.extend_from_slice(&v.to_le_bytes()),
            Param::Struct { fields, .. } => {
                for (_, f) in fields {
                    f.write_bytes(buf);
                }
            }
            Param::Region {
                stride,
                x,
                y,
                width,
                height,
                ..
            } => {
                buf.extend_from_slice(&stride.to_le_bytes());
                buf.extend_from_slice(&x.to_le_bytes());
                buf.extend_from_slice(&y.to_le_bytes());
                buf.extend_from_slice(&width.to_le_bytes());
                buf.extend_from_slice(&height.to_le_bytes());
            }
        }
    }

    pub fn struct_of(name: &'static str, fields: Vec<(&'static str, Param)>) -> Self {
        Param::Struct { name, fields }
    }

    pub fn region(
        name: impl Into<String>,
        stride: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> Self {
        Param::Region {
            name: name.into(),
            stride,
            x,
            y,
            width,
            height,
        }
    }
}

pub trait IntoParam {
    fn into_param(self) -> Param;
}

impl IntoParam for i32 {
    fn into_param(self) -> Param {
        Param::I32(self)
    }
}
impl IntoParam for u32 {
    fn into_param(self) -> Param {
        Param::U32(self)
    }
}
impl IntoParam for f32 {
    fn into_param(self) -> Param {
        Param::F32(self)
    }
}

#[derive(Clone, Copy)]
pub struct Matrix3 {
    pub m: [f32; 9],
}

impl Matrix3 {
    pub fn from_engine(m: crate::color::matrix::Matrix3x3) -> Self {
        // crate::color::matrix::Matrix3x3 is column-major `self.0[col][row]`.
        // We need row-major for the shader `m[row * 3 + col]`.
        Self {
            m: [
                m.0[0][0], m.0[1][0], m.0[2][0], m.0[0][1], m.0[1][1], m.0[2][1], m.0[0][2],
                m.0[1][2], m.0[2][2],
            ],
        }
    }
}

impl IntoParam for Matrix3 {
    fn into_param(self) -> Param {
        Param::struct_of(
            "Matrix3",
            vec![
                ("a00", Param::F32(self.m[0])),
                ("a01", Param::F32(self.m[1])),
                ("a02", Param::F32(self.m[2])),
                ("_pad0", Param::F32(0.0)),
                ("a10", Param::F32(self.m[3])),
                ("a11", Param::F32(self.m[4])),
                ("a12", Param::F32(self.m[5])),
                ("_pad1", Param::F32(0.0)),
                ("a20", Param::F32(self.m[6])),
                ("a21", Param::F32(self.m[7])),
                ("a22", Param::F32(self.m[8])),
                ("_pad2", Param::F32(0.0)),
            ],
        )
    }
}

fn default_channel_and_model(
    _cs: crate::color::space::ColorSpace,
) -> (u32, crate::color::model::ColorModelTransform) {
    (0, crate::color::model::ColorModelTransform::None)
}

#[derive(Clone, Copy)]
pub struct GpuPixelEncoding {
    pub transfer: u32,
    pub alpha: u32,
    pub model: u32,
    pub channels: u32,
    pub transform: Matrix3,
}

impl GpuPixelEncoding {
    /// Create encoding that converts from `meta` to sRGB (hub space),
    /// matching libvips `compositing_space = sRGB`.
    pub fn from_meta(meta: &crate::pixel::PixelMeta, is_source: bool) -> Self {
        let cs = meta.color_space;
        let hub = crate::color::space::ColorSpace::SRGB;
        let (from, to) = if is_source { (cs, hub) } else { (hub, cs) };
        let mat = crate::color::matrix::rgb_to_rgb_transform(
            from.primaries(),
            from.white_point(),
            to.primaries(),
            to.white_point(),
        )
        .unwrap_or(crate::color::matrix::Matrix3x3::IDENTITY);

        Self {
            transfer: cs.transfer() as u32,
            alpha: meta.alpha_policy.to_shader(),
            model: meta.format.model_transform() as u32,
            channels: super::param::gpu_channel_layout(meta.format),
            transform: Matrix3::from_engine(mat),
        }
    }

    pub fn new_source(cs: crate::color::space::ColorSpace) -> Self {
        let (channels, model) = default_channel_and_model(cs);
        let mat = crate::color::matrix::rgb_to_rgb_transform(
            cs.primaries(),
            cs.white_point(),
            crate::color::space::ColorSpace::ACES_CG.primaries(),
            crate::color::space::ColorSpace::ACES_CG.white_point(),
        )
        .unwrap_or(crate::color::matrix::Matrix3x3::IDENTITY);
        Self {
            transfer: cs.transfer() as u32,
            alpha: crate::pixel::AlphaPolicy::Straight.to_shader(),
            model: model as u32,
            channels,
            transform: Matrix3::from_engine(mat),
        }
    }

    pub fn new_dst(cs: crate::color::space::ColorSpace) -> Self {
        let (channels, model) = default_channel_and_model(cs);
        let mat = crate::color::matrix::rgb_to_rgb_transform(
            crate::color::space::ColorSpace::ACES_CG.primaries(),
            crate::color::space::ColorSpace::ACES_CG.white_point(),
            cs.primaries(),
            cs.white_point(),
        )
        .unwrap_or(crate::color::matrix::Matrix3x3::IDENTITY);
        Self {
            transfer: cs.transfer() as u32,
            alpha: crate::pixel::AlphaPolicy::Straight.to_shader(),
            model: model as u32,
            channels,
            transform: Matrix3::from_engine(mat),
        }
    }
}

impl IntoParam for GpuPixelEncoding {
    fn into_param(self) -> Param {
        let mut fields = vec![
            ("transfer_fn", Param::U32(self.transfer)),
            ("alpha_policy", Param::U32(self.alpha)),
            ("model", Param::U32(self.model)),
            ("channels", Param::U32(self.channels)),
        ];
        if let Param::Struct {
            fields: mut mat_fields,
            ..
        } = self.transform.into_param()
        {
            fields.append(&mut mat_fields);
        }
        Param::struct_of("ColorSpace", fields)
    }
}

use crate::pixel::PixelFormat;

/// Returns the GPU shader `ChannelLayout` enum value for a pixel format.
///
/// Maps to the Slang enum: `Rgba=0, Rgb=1, Gray=2, GrayA=3, CmykA=4`.
pub fn gpu_channel_layout(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Rgba8
        | PixelFormat::Rgba16
        | PixelFormat::RgbaF16
        | PixelFormat::RgbaF32
        | PixelFormat::Argb32 => 0,
        PixelFormat::Rgb8
        | PixelFormat::Rgb16
        | PixelFormat::RgbF16
        | PixelFormat::RgbF32
        | PixelFormat::YCbCr8
        | PixelFormat::YCbCrF16
        | PixelFormat::YCbCrF32
        | PixelFormat::Lab8
        | PixelFormat::Lab16 => 1,
        PixelFormat::Gray8 | PixelFormat::Gray16 | PixelFormat::GrayF16 | PixelFormat::GrayF32 => 2,
        PixelFormat::GrayA8
        | PixelFormat::GrayA16
        | PixelFormat::GrayAF16
        | PixelFormat::GrayAF32 => 3,
        PixelFormat::Cmyk8
        | PixelFormat::Cmyk16
        | PixelFormat::CmykF16
        | PixelFormat::CmykF32
        | PixelFormat::CmykA8
        | PixelFormat::CmykA16
        | PixelFormat::CmykAF16
        | PixelFormat::CmykAF32 => 4,
        PixelFormat::LabF32
        | PixelFormat::XyzF32
        | PixelFormat::YxyF32
        | PixelFormat::LChF32
        | PixelFormat::HsvU8
        | PixelFormat::HsvF32
        | PixelFormat::OklabF32
        | PixelFormat::OkLChF32 => 1,
        PixelFormat::ScRgbF32 => 0,
    }
}
