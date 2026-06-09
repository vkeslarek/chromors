use crate::backend::vips::IntoVipsEnum;
use crate::backend::vips::gobject::VipsGObject;
use crate::operation::SdfShape;

pub trait GenerateOperation {
    type Output;
    fn op_name() -> &'static [u8];
    fn build(&self, op: &mut VipsGObject);
}

pub struct Black {
    pub width: i32,
    pub height: i32,
}
impl GenerateOperation for Black {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"black\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct Grey {
    pub width: i32,
    pub height: i32,
}
impl GenerateOperation for Grey {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"grey\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct Xyz {
    pub width: i32,
    pub height: i32,
}
impl GenerateOperation for Xyz {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"xyz\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct GaussMat {
    pub sigma: f64,
    pub minimum_amplitude: f64,
}
impl GenerateOperation for GaussMat {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"gaussmat\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_double("sigma", self.sigma);
        op.set_double("min_ampl", self.minimum_amplitude);
    }
}

pub struct LogMat {
    pub sigma: f64,
    pub minimum_amplitude: f64,
}
impl GenerateOperation for LogMat {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"logmat\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_double("sigma", self.sigma);
        op.set_double("min_ampl", self.minimum_amplitude);
    }
}

pub struct Text {
    pub text: String,
    pub font: Option<String>,
    pub width: Option<i32>,
    pub rgba: Option<bool>,
}
impl GenerateOperation for Text {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"text\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_string("text", &self.text);
        if let Some(font) = &self.font {
            op.set_string("font", font);
        }
        if let Some(width) = self.width {
            op.set_int("width", width);
        }
        if let Some(rgba) = self.rgba {
            op.set_bool("rgba", rgba);
        }
    }
}

pub struct GaussNoise {
    pub width: i32,
    pub height: i32,
}
impl GenerateOperation for GaussNoise {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"gaussnoise\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct Eye {
    pub width: i32,
    pub height: i32,
}
impl GenerateOperation for Eye {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"eye\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct Sines {
    pub width: i32,
    pub height: i32,
}
impl GenerateOperation for Sines {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"sines\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct Zone {
    pub width: i32,
    pub height: i32,
}
impl GenerateOperation for Zone {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"zone\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct Sdf {
    pub width: i32,
    pub height: i32,
    pub shape: SdfShape,
}
impl GenerateOperation for Sdf {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"sdf\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_int("shape", self.shape.into_vips());
    }
}

pub struct Identity;
impl GenerateOperation for Identity {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"identity\0"
    }
    fn build(&self, _op: &mut VipsGObject) {}
}

pub struct Tonelut;
impl GenerateOperation for Tonelut {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"tonelut\0"
    }
    fn build(&self, _op: &mut VipsGObject) {}
}

pub struct FractSurf {
    pub width: i32,
    pub height: i32,
    pub fractal_dimension: f64,
}
impl GenerateOperation for FractSurf {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"fractsurf\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("fractal_dimension", self.fractal_dimension);
    }
}

pub struct Worley {
    pub width: i32,
    pub height: i32,
}
impl GenerateOperation for Worley {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"worley\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct Perlin {
    pub width: i32,
    pub height: i32,
}
impl GenerateOperation for Perlin {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"perlin\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct MaskIdeal {
    pub width: i32,
    pub height: i32,
    pub frequency_cutoff: f64,
    pub uchar: Option<bool>,
    pub nodc: Option<bool>,
    pub reject: Option<bool>,
    pub optical: Option<bool>,
}
impl GenerateOperation for MaskIdeal {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"mask_ideal\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("frequency_cutoff", self.frequency_cutoff);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
        if let Some(v) = self.nodc {
            op.set_bool("nodc", v);
        }
        if let Some(v) = self.reject {
            op.set_bool("reject", v);
        }
        if let Some(v) = self.optical {
            op.set_bool("optical", v);
        }
    }
}

pub struct MaskIdealBand {
    pub width: i32,
    pub height: i32,
    pub frequency_cutoff_x: f64,
    pub frequency_cutoff_y: f64,
    pub radius: f64,
    pub uchar: Option<bool>,
    pub nodc: Option<bool>,
    pub reject: Option<bool>,
    pub optical: Option<bool>,
}
impl GenerateOperation for MaskIdealBand {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"mask_ideal_band\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("frequency_cutoff_x", self.frequency_cutoff_x);
        op.set_double("frequency_cutoff_y", self.frequency_cutoff_y);
        op.set_double("radius", self.radius);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
        if let Some(v) = self.nodc {
            op.set_bool("nodc", v);
        }
        if let Some(v) = self.reject {
            op.set_bool("reject", v);
        }
        if let Some(v) = self.optical {
            op.set_bool("optical", v);
        }
    }
}

pub struct MaskIdealRing {
    pub width: i32,
    pub height: i32,
    pub frequency_cutoff: f64,
    pub ringwidth: f64,
    pub uchar: Option<bool>,
    pub nodc: Option<bool>,
    pub reject: Option<bool>,
    pub optical: Option<bool>,
}
impl GenerateOperation for MaskIdealRing {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"mask_ideal_ring\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("frequency_cutoff", self.frequency_cutoff);
        op.set_double("ringwidth", self.ringwidth);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
        if let Some(v) = self.nodc {
            op.set_bool("nodc", v);
        }
        if let Some(v) = self.reject {
            op.set_bool("reject", v);
        }
        if let Some(v) = self.optical {
            op.set_bool("optical", v);
        }
    }
}

pub struct MaskButterworth {
    pub width: i32,
    pub height: i32,
    pub order: f64,
    pub frequency_cutoff: f64,
    pub amplitude_cutoff: f64,
    pub uchar: Option<bool>,
    pub nodc: Option<bool>,
    pub reject: Option<bool>,
    pub optical: Option<bool>,
}
impl GenerateOperation for MaskButterworth {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"mask_butterworth\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("order", self.order);
        op.set_double("frequency_cutoff", self.frequency_cutoff);
        op.set_double("amplitude_cutoff", self.amplitude_cutoff);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
        if let Some(v) = self.nodc {
            op.set_bool("nodc", v);
        }
        if let Some(v) = self.reject {
            op.set_bool("reject", v);
        }
        if let Some(v) = self.optical {
            op.set_bool("optical", v);
        }
    }
}

pub struct MaskButterworthBand {
    pub width: i32,
    pub height: i32,
    pub order: f64,
    pub frequency_cutoff_x: f64,
    pub frequency_cutoff_y: f64,
    pub radius: f64,
    pub amplitude_cutoff: f64,
    pub uchar: Option<bool>,
    pub nodc: Option<bool>,
    pub reject: Option<bool>,
    pub optical: Option<bool>,
}
impl GenerateOperation for MaskButterworthBand {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"mask_butterworth_band\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("order", self.order);
        op.set_double("frequency_cutoff_x", self.frequency_cutoff_x);
        op.set_double("frequency_cutoff_y", self.frequency_cutoff_y);
        op.set_double("radius", self.radius);
        op.set_double("amplitude_cutoff", self.amplitude_cutoff);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
        if let Some(v) = self.nodc {
            op.set_bool("nodc", v);
        }
        if let Some(v) = self.reject {
            op.set_bool("reject", v);
        }
        if let Some(v) = self.optical {
            op.set_bool("optical", v);
        }
    }
}

pub struct MaskButterworthRing {
    pub width: i32,
    pub height: i32,
    pub order: f64,
    pub frequency_cutoff: f64,
    pub amplitude_cutoff: f64,
    pub ringwidth: f64,
    pub uchar: Option<bool>,
    pub nodc: Option<bool>,
    pub reject: Option<bool>,
    pub optical: Option<bool>,
}
impl GenerateOperation for MaskButterworthRing {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"mask_butterworth_ring\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("order", self.order);
        op.set_double("frequency_cutoff", self.frequency_cutoff);
        op.set_double("amplitude_cutoff", self.amplitude_cutoff);
        op.set_double("ringwidth", self.ringwidth);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
        if let Some(v) = self.nodc {
            op.set_bool("nodc", v);
        }
        if let Some(v) = self.reject {
            op.set_bool("reject", v);
        }
        if let Some(v) = self.optical {
            op.set_bool("optical", v);
        }
    }
}

pub struct MaskGaussian {
    pub width: i32,
    pub height: i32,
    pub frequency_cutoff: f64,
    pub amplitude_cutoff: f64,
    pub uchar: Option<bool>,
    pub nodc: Option<bool>,
    pub reject: Option<bool>,
    pub optical: Option<bool>,
}
impl GenerateOperation for MaskGaussian {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"mask_gaussian\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("frequency_cutoff", self.frequency_cutoff);
        op.set_double("amplitude_cutoff", self.amplitude_cutoff);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
        if let Some(v) = self.nodc {
            op.set_bool("nodc", v);
        }
        if let Some(v) = self.reject {
            op.set_bool("reject", v);
        }
        if let Some(v) = self.optical {
            op.set_bool("optical", v);
        }
    }
}

pub struct MaskGaussianBand {
    pub width: i32,
    pub height: i32,
    pub frequency_cutoff_x: f64,
    pub frequency_cutoff_y: f64,
    pub radius: f64,
    pub amplitude_cutoff: f64,
    pub uchar: Option<bool>,
    pub nodc: Option<bool>,
    pub reject: Option<bool>,
    pub optical: Option<bool>,
}
impl GenerateOperation for MaskGaussianBand {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"mask_gaussian_band\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("frequency_cutoff_x", self.frequency_cutoff_x);
        op.set_double("frequency_cutoff_y", self.frequency_cutoff_y);
        op.set_double("radius", self.radius);
        op.set_double("amplitude_cutoff", self.amplitude_cutoff);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
        if let Some(v) = self.nodc {
            op.set_bool("nodc", v);
        }
        if let Some(v) = self.reject {
            op.set_bool("reject", v);
        }
        if let Some(v) = self.optical {
            op.set_bool("optical", v);
        }
    }
}

pub struct MaskGaussianRing {
    pub width: i32,
    pub height: i32,
    pub frequency_cutoff: f64,
    pub amplitude_cutoff: f64,
    pub ringwidth: f64,
    pub uchar: Option<bool>,
    pub nodc: Option<bool>,
    pub reject: Option<bool>,
    pub optical: Option<bool>,
}
impl GenerateOperation for MaskGaussianRing {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"mask_gaussian_ring\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("frequency_cutoff", self.frequency_cutoff);
        op.set_double("amplitude_cutoff", self.amplitude_cutoff);
        op.set_double("ringwidth", self.ringwidth);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
        if let Some(v) = self.nodc {
            op.set_bool("nodc", v);
        }
        if let Some(v) = self.reject {
            op.set_bool("reject", v);
        }
        if let Some(v) = self.optical {
            op.set_bool("optical", v);
        }
    }
}

pub struct MaskFractal {
    pub width: i32,
    pub height: i32,
    pub fractal_dimension: f64,
    pub uchar: Option<bool>,
    pub nodc: Option<bool>,
    pub reject: Option<bool>,
    pub optical: Option<bool>,
}
impl GenerateOperation for MaskFractal {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn op_name() -> &'static [u8] {
        b"mask_fractal\0"
    }
    fn build(&self, op: &mut VipsGObject) {
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        op.set_double("fractal_dimension", self.fractal_dimension);
        if let Some(v) = self.uchar {
            op.set_bool("uchar", v);
        }
        if let Some(v) = self.nodc {
            op.set_bool("nodc", v);
        }
        if let Some(v) = self.reject {
            op.set_bool("reject", v);
        }
        if let Some(v) = self.optical {
            op.set_bool("optical", v);
        }
    }
}
