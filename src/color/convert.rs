use crate::backend::vips::VipsBackend;
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::{IntoVipsBandFormat, IntoVipsInterpretation};
use crate::data::image::Image;
use crate::error::Error;
use crate::pixel::PixelMeta;

pub struct ColorConversion {
    pub from: PixelMeta,
    pub to: PixelMeta,
}

impl ColorConversion {
    pub fn new(from: PixelMeta, to: PixelMeta) -> Self {
        ColorConversion { from, to }
    }

    pub fn execute(&self, image: &Image<VipsBackend>) -> Result<Image<VipsBackend>, Error> {
        let mut img = image.clone();

        let from_premultiplied = matches!(
            self.from.alpha_policy,
            crate::pixel::AlphaPolicy::PremultiplyOnPack
        );
        let to_straight = matches!(self.to.alpha_policy, crate::pixel::AlphaPolicy::Straight);
        let to_premultiplied = matches!(
            self.to.alpha_policy,
            crate::pixel::AlphaPolicy::PremultiplyOnPack
        );
        let to_opaque = matches!(self.to.alpha_policy, crate::pixel::AlphaPolicy::OpaqueDrop);

        if from_premultiplied && (to_straight || to_opaque) && img.has_alpha() {
            let mut op = VipsGObject::new(b"unpremultiply\0")?;
            op.set_image("in", img.vips_ptr());
            img = op.run()?;
        }

        if self.from.color_space != self.to.color_space {
            let interp = self.to.color_space.into_vips_interpretation();
            let mut op = VipsGObject::new(b"colourspace\0")?;
            op.set_image("in", img.vips_ptr());
            op.set_int("space", interp);
            img = op.run()?;
        }

        if to_premultiplied && !from_premultiplied && img.has_alpha() {
            let mut op = VipsGObject::new(b"premultiply\0")?;
            op.set_image("in", img.vips_ptr());
            img = op.run()?;
        } else if to_opaque && img.has_alpha() {
            let mut op = VipsGObject::new(b"flatten\0")?;
            op.set_image("in", img.vips_ptr());
            img = op.run()?;
        }

        let target_bands = self.to.format.channels() as i32;
        let has_alpha = img.has_alpha();
        if target_bands > img.bands() && !has_alpha && target_bands == img.bands() + 1 {
            let mut op = VipsGObject::new(b"addalpha\0")?;
            op.set_image("in", img.vips_ptr());
            img = op.run()?;
        }

        if self.to.format != self.from.format {
            let mut op = VipsGObject::new(b"cast\0")?;
            op.set_image("in", img.vips_ptr());
            op.set_int("format", self.to.format.into_vips_band_format());
            img = op.run()?;
        }

        Ok(img)
    }
}
