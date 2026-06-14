//! Rendering intent for [`crate::operation::color::Convert`]
//! (`docs/native-color-management.md` §6.1).
//!
//! Only `Relative` (colorimetric, no gamut mapping) is implemented today.
//! `Perceptual`/`Saturation` (3D LUT gamut mapping, §10) are future work —
//! `Convert::lower` doesn't branch on this field yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RenderingIntent {
    #[default]
    Relative,
}
