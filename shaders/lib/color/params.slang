// lib/color/params.slang — uniform parameters for the XYZ(D50)-hub colour
// pipeline (`color_convert`, lib/color/convert.slang).
//
// Must mirror the Rust `ConvertParams` (#[repr(C)] Pod,
// `src/backend/gpu/color_params.rs`) byte-for-byte: std430 scalar-array
// stride == element size, so `float a[12]`/`float b[12]` need no padding to
// match Rust's `[f32; 12]`.

import lib.color.transfer;
import lib.color.model;

struct ColorConvertParams
{
    TransferFn transfer_src;
    TransferFn transfer_dst;

    // src linear RGB -> CIE XYZ (D50 hub). 3 rows, each padded to 4 floats.
    float a[12];
    // CIE XYZ (D50 hub) -> dst linear RGB. 3 rows, each padded to 4 floats.
    float b[12];

    // 0 = straight/none, 1 = premultiplied (see `AlphaState::to_shader`).
    uint alpha_src;
    uint alpha_dst;

    ColorModel model_src;
    ColorModel model_dst;

    uint nchan_src;
    uint nchan_dst;
};
