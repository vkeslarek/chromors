//! Storage — how a single channel sample is quantized in memory.
//!
//! `Storage` is AGNOSTIC: it carries no color meaning and no backend
//! knowledge. Backend mappings (`gpu_codec`, `into_vips_band_format`) are
//! trait impls defined by the owning backend (see `CLAUDE.md` §2/§3.6).

use serde::{Deserialize, Serialize};

use crate::backend::vips::{FromVipsBandFormat, IntoVipsBandFormat};

/// Sample quantization for one channel. No color meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Storage {
    /// 8-bit unsigned, normalized to `[0,1]` by `/255`.
    U8 = 0,
    /// 16-bit unsigned, normalized to `[0,1]` by `/65535`.
    U16 = 1,
    /// 16-bit IEEE half float.
    F16 = 2,
    /// 32-bit IEEE float, already in working units.
    F32 = 3,
}

impl Storage {
    /// Bytes occupied by one sample of this storage type.
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Storage::U8 => 1,
            Storage::U16 => 2,
            Storage::F16 => 2,
            Storage::F32 => 4,
        }
    }

    /// Normalization divisor bringing a raw integer sample into `[0,1]`;
    /// `1.0` for float storage (already normalized).
    pub const fn component_max(self) -> f32 {
        match self {
            Storage::U8 => 255.0,
            Storage::U16 => 65535.0,
            Storage::F16 | Storage::F32 => 1.0,
        }
    }
}

/// Maps `Storage` to a `VipsBandFormat` enum value (`docs/native-color-management.md`
/// §3.6). Vips has no half-float band format, so `F16` maps to `FLOAT` like `F32` —
/// the closest representable type.
impl IntoVipsBandFormat for Storage {
    fn into_vips_band_format(self) -> i32 {
        match self {
            Storage::U8 => crate::ffi::VipsBandFormat_VIPS_FORMAT_UCHAR,
            Storage::U16 => crate::ffi::VipsBandFormat_VIPS_FORMAT_USHORT,
            Storage::F16 | Storage::F32 => crate::ffi::VipsBandFormat_VIPS_FORMAT_FLOAT,
        }
    }
}

/// Reconstructs `Storage` from a `VipsBandFormat` enum value
/// (`docs/native-color-management.md` §3.6). `bands` is unused — storage
/// quantization is independent of band count. Unknown formats default to
/// `U8`.
impl FromVipsBandFormat for Storage {
    fn from_vips_band_format(raw: i32, _bands: i32) -> Self {
        match raw {
            2 => Storage::U16,
            6 | 8 => Storage::F32,
            _ => Storage::U8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_per_sample() {
        assert_eq!(Storage::U8.bytes_per_sample(), 1);
        assert_eq!(Storage::U16.bytes_per_sample(), 2);
        assert_eq!(Storage::F16.bytes_per_sample(), 2);
        assert_eq!(Storage::F32.bytes_per_sample(), 4);
    }

    #[test]
    fn component_max() {
        assert_eq!(Storage::U8.component_max(), 255.0);
        assert_eq!(Storage::U16.component_max(), 65535.0);
        assert_eq!(Storage::F16.component_max(), 1.0);
        assert_eq!(Storage::F32.component_max(), 1.0);
    }
}
