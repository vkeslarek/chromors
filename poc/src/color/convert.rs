use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::{IntoVipsBandFormat, IntoVipsInterpretation, VipsHandle};
use crate::error::Error;
use crate::ffi;
use crate::pixel::PixelMeta;

pub struct ColorConversion {
    pub from: PixelMeta,
    pub to: PixelMeta,
}

impl ColorConversion {
    pub fn new(from: PixelMeta, to: PixelMeta) -> Self {
        ColorConversion { from, to }
    }

    pub fn execute(&self, image: &VipsHandle) -> Result<VipsHandle, Error> {
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

        let has_alpha = unsafe { ffi::vips_image_hasalpha(img.ptr) } != 0;
        let _bands = unsafe { ffi::vips_image_get_bands(img.ptr) };

        if from_premultiplied && (to_straight || to_opaque) && has_alpha {
            let mut op = VipsGObject::new(b"unpremultiply\0")?;
            op.set_image("in", img.ptr);
            img = op.run()?;
        }

        if self.from.color_space != self.to.color_space {
            let interp = self.to.color_space.into_vips_interpretation();
            let mut op = VipsGObject::new(b"colourspace\0")?;
            op.set_image("in", img.ptr);
            op.set_int("space", interp);
            img = op.run()?;
        }

        let has_alpha_now = unsafe { ffi::vips_image_hasalpha(img.ptr) } != 0;
        if to_premultiplied && !from_premultiplied && has_alpha_now {
            let mut op = VipsGObject::new(b"premultiply\0")?;
            op.set_image("in", img.ptr);
            img = op.run()?;
        } else if to_opaque && has_alpha_now {
            let mut op = VipsGObject::new(b"flatten\0")?;
            op.set_image("in", img.ptr);
            img = op.run()?;
        }

        let target_bands = self.to.format.channels() as i32;
        let bands_now = unsafe { ffi::vips_image_get_bands(img.ptr) };
        if target_bands > bands_now && !has_alpha_now && target_bands == bands_now + 1 {
            let mut op = VipsGObject::new(b"addalpha\0")?;
            op.set_image("in", img.ptr);
            img = op.run()?;
        }

        if self.to.format != self.from.format {
            let mut op = VipsGObject::new(b"cast\0")?;
            op.set_image("in", img.ptr);
            op.set_int("format", self.to.format.into_vips_band_format());
            img = op.run()?;
        }

        Ok(img)
    }
}
