//! `Convert` — the universal color/format conversion operation
//! (`docs/native-color-management.md` §6).
//!
//! GPU lowering needs no bespoke kernel for the common case: the generic
//! `copy_kernel` (`lib/io.slang`) plus a `color_read_wrap`
//! (`lib/color/interp.slang`, §5.12) does the whole XYZ(D50)-hub conversion
//! on read, writing through the target layout's own codec sandwich. The one
//! exception is a 5-channel (CmykA) source: its K-alpha 5th sample can't ride
//! through the generic `float4` read wrap, so `convert_cmyka_kernel`
//! (`lib/color/convert.slang`, §6.1.3) binds the source's storage codec
//! directly.

use std::hash::Hasher;

use crate::backend::Backend;
use crate::color::intent::RenderingIntent;
use crate::color::matrix::Matrix3x3;
use crate::color::model::ColorModel;
use crate::color::space::ColorSpace;
use crate::data::image::{Image2D, ImageKind};
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::pixel::{PixelLayout, Storage};
use crate::work_unit::{Region, WorkUnit};

/// Converts an image to `target` — storage, color model, alpha state and/or
/// color space, in one pass. Pointwise: the output region equals the input
/// region (`demand`).
pub struct Convert<B: Backend> {
    pub input: Input<ImageKind, B>,
    /// The full destination pixel layout.
    pub target: PixelLayout,
    /// Reserved for future gamut mapping (§10); `lower` doesn't branch on it yet.
    pub intent: RenderingIntent,
}

impl<B: Backend> Operation<B> for Convert<B>
where
    Convert<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }

    fn output_spec(&self) -> ImageKind {
        self.input.spec.with_layout(self.target)
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write(format!("{:?}", self.target).as_bytes());
        state.write_u8(match self.intent {
            RenderingIntent::Relative => 0,
        });
    }
}

impl<B: Backend> Image2D<B>
where
    Convert<B>: Lower<B>,
{
    /// Converts this image to `target` — storage/model/alpha/color-space, in
    /// one pass.
    pub fn convert(&self, target: PixelLayout) -> Self {
        self.push(Convert {
            input: self.as_input(),
            target,
            intent: RenderingIntent::Relative,
        })
    }

    /// Changes the color space (primaries/white point/transfer), keeping
    /// storage/model/alpha identical (`docs/native-color-management.md`
    /// §6.2).
    pub fn to_color_space(&self, cs: ColorSpace) -> Self {
        let mut target = self.spec.layout;
        target.color_space = cs;
        self.convert(target)
    }

    /// Changes the sample storage type, keeping model/alpha/color-space
    /// identical (`docs/native-color-management.md` §6.2).
    pub fn to_storage(&self, storage: Storage) -> Self {
        let mut target = self.spec.layout;
        target.storage = storage;
        self.convert(target)
    }

    /// Changes the color model (e.g. `Rgb` -> `Lab`), keeping
    /// storage/alpha/color-space identical (`docs/native-color-management.md`
    /// §6.2).
    pub fn to_model(&self, model: ColorModel) -> Self {
        let mut target = self.spec.layout;
        target.model = model;
        self.convert(target)
    }

    /// Switches to the linear variant of the current color space — a pure
    /// transfer-function change (`docs/native-color-management.md` §6.2).
    pub fn linearize(&self) -> Self {
        self.to_color_space(self.spec.layout.color_space.as_linear())
    }
}
