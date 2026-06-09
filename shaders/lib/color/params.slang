// lib/color/params.slang — Uniform parameters for the XYZ-hub colour pipeline.
//
// Must mirror the Rust `ColorConvertParams` in
// `crates/pixors-shader/src/kernel/color.rs`.

import lib.region;
import lib.pixel;
import lib.color.transfer;
import lib.color.model;

struct ColorConvertParams
{
    BufferRegion src_region;
    uint  out_width;
    uint  out_height;
    TransferFn transfer_src;
    TransferFn transfer_dst;

    // src linear RGB → CIE XYZ (D50 hub). Row-major, each row padded to vec4.
    float a00; float a01; float a02; float _pa0;
    float a10; float a11; float a12; float _pa1;
    float a20; float a21; float a22; float _pa2;

    // CIE XYZ (D50 hub) → dst linear RGB. Row-major, each row padded to vec4.
    float b00; float b01; float b02; float _pb0;
    float b10; float b11; float b12; float _pb1;
    float b20; float b21; float b22; float _pb2;

    AlphaPolicy   alpha_policy_src;
    AlphaPolicy   alpha_policy_dst;
    ColorModel    src_model;
    ColorModel    dst_model;
    ChannelLayout src_channels;
    ChannelLayout dst_channels;
};
