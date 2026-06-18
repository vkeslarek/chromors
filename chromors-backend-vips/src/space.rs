use crate::ColorModel;
use crate::ColorSpace;
use crate::RgbPrimaries;
use crate::TransferFn;
use crate::WhitePoint;
use crate::AlphaState;
use crate::ffi;

pub fn to_vips_interpretation(model: ColorModel, cs: ColorSpace) -> Option<i32> {
    use crate::ffi::{
        VipsInterpretation_VIPS_INTERPRETATION_B_W as B_W,
        VipsInterpretation_VIPS_INTERPRETATION_CMYK as CMYK,
        VipsInterpretation_VIPS_INTERPRETATION_LAB as LAB,
        VipsInterpretation_VIPS_INTERPRETATION_XYZ as XYZ,
        VipsInterpretation_VIPS_INTERPRETATION_sRGB as SRGB,
        VipsInterpretation_VIPS_INTERPRETATION_scRGB as SC_RGB,
    };
    match model {
        ColorModel::Gray => Some(B_W),
        ColorModel::Lab => Some(LAB),
        ColorModel::Xyz => Some(XYZ),
        ColorModel::Cmyk => Some(CMYK),
        ColorModel::ScRgb => Some(SC_RGB),
        ColorModel::Rgb if cs == ColorSpace::SRGB => Some(SRGB),
        ColorModel::Rgb if cs == ColorSpace::LINEAR_SRGB => Some(SC_RGB),
        _ => None,
    }
}

pub fn from_vips_interpretation(interp: i32, bands: i32) -> (ColorModel, AlphaState, ColorSpace) {
    use crate::ffi::{
        VipsInterpretation_VIPS_INTERPRETATION_B_W as B_W,
        VipsInterpretation_VIPS_INTERPRETATION_CMYK as CMYK,
        VipsInterpretation_VIPS_INTERPRETATION_GREY16 as GREY16,
        VipsInterpretation_VIPS_INTERPRETATION_HSV as HSV,
        VipsInterpretation_VIPS_INTERPRETATION_LAB as LAB,
        VipsInterpretation_VIPS_INTERPRETATION_LABQ as LABQ,
        VipsInterpretation_VIPS_INTERPRETATION_LABS as LABS,
        VipsInterpretation_VIPS_INTERPRETATION_LCH as LCH,
        VipsInterpretation_VIPS_INTERPRETATION_RGB as RGB,
        VipsInterpretation_VIPS_INTERPRETATION_RGB16 as RGB16,
        VipsInterpretation_VIPS_INTERPRETATION_XYZ as XYZ,
        VipsInterpretation_VIPS_INTERPRETATION_YXY as YXY,
        VipsInterpretation_VIPS_INTERPRETATION_sRGB as SRGB,
        VipsInterpretation_VIPS_INTERPRETATION_scRGB as SC_RGB,
    };

    let connection = ColorSpace::new(RgbPrimaries::Bt709, WhitePoint::D50, TransferFn::Linear);

    let (model, default_cs) = match interp {
        B_W | GREY16 => (ColorModel::Gray, ColorSpace::SRGB),
        XYZ => (ColorModel::Xyz, connection),
        LAB | LABS | LABQ => (ColorModel::Lab, connection),
        LCH => (ColorModel::Lch, connection),
        YXY => (ColorModel::Yxy, connection),
        CMYK => (ColorModel::Cmyk, ColorSpace::SRGB),
        SC_RGB => (ColorModel::ScRgb, ColorSpace::LINEAR_SRGB),
        SRGB | RGB | RGB16 => (ColorModel::Rgb, ColorSpace::SRGB),
        HSV => (ColorModel::Hsv, ColorSpace::SRGB),
        _ => {
            return match bands {
                1 => (ColorModel::Gray, AlphaState::None, ColorSpace::SRGB),
                3 => (ColorModel::Rgb, AlphaState::None, ColorSpace::SRGB),
                4 => (ColorModel::Rgb, AlphaState::Straight, ColorSpace::SRGB),
                n => (ColorModel::Multiband(n.clamp(0, 255) as u8), AlphaState::None, ColorSpace::SRGB),
            };
        }
    };

    let alpha = if bands > model.color_channels() as i32 { AlphaState::Straight } else { AlphaState::None };
    (model, alpha, default_cs)
}
