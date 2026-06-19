use crate::ffi;
use crate::{FromVipsBandFormat, IntoVipsBandFormat};
use chromors_core::pixel::Storage;

impl IntoVipsBandFormat for Storage {
    fn into_vips_band_format(self) -> i32 {
        match self {
            Storage::U8 => ffi::VipsBandFormat_VIPS_FORMAT_UCHAR,
            Storage::U16 => ffi::VipsBandFormat_VIPS_FORMAT_USHORT,
            Storage::F16 | Storage::F32 => ffi::VipsBandFormat_VIPS_FORMAT_FLOAT,
        }
    }
}

impl FromVipsBandFormat for Storage {
    fn from_vips_band_format(raw: i32, _bands: i32) -> Self {
        match raw {
            2 => Storage::U16,
            6 | 8 => Storage::F32,
            _ => Storage::U8,
        }
    }
}
