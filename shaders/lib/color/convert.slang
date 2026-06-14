// lib/color/convert.slang — XYZ(D50)-hub colour conversion (`color_convert`).
//
// Pipeline per pixel:
//   src raw channels (decoded by the storage codec)
//     -> model decode  (Cmyk/YCbCr/Hsv -> encoded RGB; Lab/Lch/Xyz/Yxy/Oklab/
//                        Oklch -> XYZ(D50); Rgb/Gray/ScRgb -> encoded RGB)
//     -> OETF^-1                (skipped for the XYZ(D50)-valued models)
//     -> unpremultiply if alpha_src == Premultiplied
//     -> matrix A (linear src RGB -> XYZ(D50))     (skipped if already XYZ)
//     -> matrix B (XYZ(D50) -> linear dst RGB)     (skipped if dst is XYZ-valued)
//     -> premultiply if alpha_dst == Premultiplied
//     -> model encode  (inverse of the decode step, OETF where applicable)
//
// XYZ(D50) is the hub — every space converts through it, so adding a new
// model only needs `<model> <-> XYZ(D50)` math, not pairwise conversions.

import lib.color.params;
import lib.color.transfer;
import lib.color.model;
import lib.region;
import lib.codecs;

// ── shared constants ─────────────────────────────────────────────────────

static const float TAU = 6.283185307179586;

// ── Lab <-> XYZ(D50) ──────────────────────────────────────────────────────

static const float LAB_DELTA  = 6.0 / 29.0;
static const float LAB_DELTA2 = 3.0 * LAB_DELTA * LAB_DELTA;
static const float LAB_DELTA3 = LAB_DELTA * LAB_DELTA * LAB_DELTA;
static const float XN_D50 = 0.96422;
static const float YN_D50 = 1.0;
static const float ZN_D50 = 0.82521;

float3 lab_to_xyz_d50(float L_norm, float a_norm, float b_norm)
{
    float L = L_norm * 100.0;
    float a = a_norm * 128.0;
    float b = b_norm * 128.0;
    float fy = (L + 16.0) / 116.0;
    float fx = a / 500.0 + fy;
    float fz = fy - b / 200.0;
    float x = XN_D50 * (fx > LAB_DELTA ? fx*fx*fx : LAB_DELTA2*(fx - 4.0/29.0));
    float y = YN_D50 * (fy > LAB_DELTA ? fy*fy*fy : LAB_DELTA2*(fy - 4.0/29.0));
    float z = ZN_D50 * (fz > LAB_DELTA ? fz*fz*fz : LAB_DELTA2*(fz - 4.0/29.0));
    return float3(x, y, z);
}

float3 xyz_d50_to_lab(float3 xyz)
{
    float fx = xyz.x / XN_D50; fx = fx > LAB_DELTA3 ? pow(fx, 1.0/3.0) : (1.0/LAB_DELTA2)*fx + 4.0/29.0;
    float fy = xyz.y / YN_D50; fy = fy > LAB_DELTA3 ? pow(fy, 1.0/3.0) : (1.0/LAB_DELTA2)*fy + 4.0/29.0;
    float fz = xyz.z / ZN_D50; fz = fz > LAB_DELTA3 ? pow(fz, 1.0/3.0) : (1.0/LAB_DELTA2)*fz + 4.0/29.0;
    float L = 116.0 * fy - 16.0;
    float a = 500.0 * (fx - fy);
    float b = 200.0 * (fy - fz);
    return float3(L / 100.0, a / 128.0, b / 128.0);
}

// ── Lab <-> Lch / Oklab <-> Oklch (generic polar transform) ───────────────
//
// Scale-agnostic: `a`/`b` and `c` (chroma) share whatever scale the model
// uses (Lab's /128-normalized a*/b*, or Oklab's natural ~[-0.5, 0.5] range) —
// `c = length(a, b)` and back, so the same pair of functions serves both
// Lab<->Lch and Oklab<->Oklch.

float3 lab_to_lch(float3 lab)
{
    float c = length(lab.yz);
    float h = atan2(lab.z, lab.y) / TAU;
    if (h < 0.0) { h += 1.0; }
    return float3(lab.x, c, h);
}

float3 lch_to_lab(float3 lch)
{
    float angle = lch.z * TAU;
    return float3(lch.x, lch.y * cos(angle), lch.y * sin(angle));
}

// ── CIE xyY <-> XYZ ─────────────────────────────────────────────────────────
//
// Channels: `(x, y, Y)` — chromaticity coordinates `x`,`y` and luminance `Y`.

float3 xyy_to_xyz(float3 v)
{
    if (v.y < 1e-6) { return float3(0.0, 0.0, 0.0); }
    float scale = v.z / v.y;
    return float3(v.x * scale, v.z, (1.0 - v.x - v.y) * scale);
}

float3 xyz_to_xyy(float3 xyz)
{
    float sum = xyz.x + xyz.y + xyz.z;
    if (sum < 1e-6) { return float3(0.0, 0.0, xyz.y); }
    return float3(xyz.x / sum, xyz.y / sum, xyz.y);
}

// ── HSV <-> RGB ───────────────────────────────────────────────────────────
//
// Operates on encoded (gamma) RGB, like the model hop for Cmyk/YCbCr —
// `color_convert` runs this immediately before/after the transfer step.

float3 rgb_to_hsv(float3 c)
{
    float mx = max(c.r, max(c.g, c.b));
    float mn = min(c.r, min(c.g, c.b));
    float d = mx - mn;
    float h = 0.0;
    if (d > 1e-8) {
        if (mx == c.r) { h = fmod((c.g - c.b) / d, 6.0); }
        else if (mx == c.g) { h = (c.b - c.r) / d + 2.0; }
        else { h = (c.r - c.g) / d + 4.0; }
        h /= 6.0;
        if (h < 0.0) { h += 1.0; }
    }
    float s = (mx <= 1e-8) ? 0.0 : d / mx;
    return float3(h, s, mx);
}

float3 hsv_to_rgb(float3 c)
{
    float h = c.x * 6.0, s = c.y, v = c.z;
    float chroma = v * s;
    float x = chroma * (1.0 - abs(fmod(h, 2.0) - 1.0));
    float m = v - chroma;
    float3 rgb;
    if (h < 1.0)      { rgb = float3(chroma, x, 0.0); }
    else if (h < 2.0) { rgb = float3(x, chroma, 0.0); }
    else if (h < 3.0) { rgb = float3(0.0, chroma, x); }
    else if (h < 4.0) { rgb = float3(0.0, x, chroma); }
    else if (h < 5.0) { rgb = float3(x, 0.0, chroma); }
    else              { rgb = float3(chroma, 0.0, x); }
    return rgb + float3(m, m, m);
}

// ── Oklab <-> XYZ(D50) ────────────────────────────────────────────────────
//
// Oklab's published matrices (Björn Ottosson) are defined relative to
// XYZ(D65); our hub is XYZ(D50), so we Bradford-adapt at the boundary.

float cbrt_signed(float x)
{
    return sign(x) * pow(abs(x), 1.0 / 3.0);
}

float3 oklab_to_xyz_d50(float3 lab)
{
    float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;

    float3 xyz_d65 = float3(
        1.2270138511 * l - 0.5577999807 * m + 0.2812561490 * s,
       -0.0405801784 * l + 1.1122568696 * m - 0.0716766787 * s,
       -0.0763812845 * l - 0.4214819784 * m + 1.5861632204 * s);

    // Bradford D65 -> D50
    return float3(
        1.0478112 * xyz_d65.x + 0.0228866 * xyz_d65.y - 0.0501270 * xyz_d65.z,
        0.0295424 * xyz_d65.x + 0.9904844 * xyz_d65.y - 0.0170491 * xyz_d65.z,
       -0.0092345 * xyz_d65.x + 0.0150436 * xyz_d65.y + 0.7521316 * xyz_d65.z);
}

float3 xyz_d50_to_oklab(float3 xyz_d50)
{
    // Bradford D50 -> D65
    float3 xyz_d65 = float3(
        0.9555766 * xyz_d50.x - 0.0230393 * xyz_d50.y + 0.0631636 * xyz_d50.z,
       -0.0282895 * xyz_d50.x + 1.0099416 * xyz_d50.y + 0.0210077 * xyz_d50.z,
        0.0122982 * xyz_d50.x - 0.0204830 * xyz_d50.y + 1.3299098 * xyz_d50.z);

    float l = 0.8189330101 * xyz_d65.x + 0.3618667424 * xyz_d65.y - 0.1288597137 * xyz_d65.z;
    float m = 0.0329845436 * xyz_d65.x + 0.9293118715 * xyz_d65.y + 0.0361456387 * xyz_d65.z;
    float s = 0.0482003018 * xyz_d65.x + 0.2643662691 * xyz_d65.y - 0.6338517070 * xyz_d65.z;

    float l_ = cbrt_signed(l);
    float m_ = cbrt_signed(m);
    float s_ = cbrt_signed(s);

    return float3(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_);
}

// ── color_convert ─────────────────────────────────────────────────────────

/// `extra_alpha`: CmykA's 5th channel (from `decode_extra`); `1.0` otherwise.
/// `p.model_src`/`p.model_dst` identify the channel interpretation on each
/// side; `p.a`/`p.b` carry the XYZ(D50)-hub matrices for the RGB-valued
/// models (ignored when that side is XYZ(D50)-valued: Lab/Lch/Xyz/Yxy/Oklab/
/// Oklch).
float4 color_convert(float4 raw, float extra_alpha, ColorConvertParams p)
{
    // ── 1. Source model decode -> `c` is encoded RGB, or XYZ(D50) if `src_is_xyz` ──
    float3 c;
    float a;
    bool src_is_xyz;

    switch (p.model_src) {
    case ColorModel::Cmyk: {
        float scale = 1.0 - raw.a; // raw.a == K
        c = (float3(1.0) - raw.rgb) * scale;
        a = (p.nchan_src >= 5) ? extra_alpha : 1.0;
        src_is_xyz = false;
        break;
    }
    case ColorModel::YCbCr: {
        float cb = raw.g - 0.5;
        float cr = raw.b - 0.5;
        c = float3(raw.r + 1.402 * cr, raw.r - 0.344136 * cb - 0.714136 * cr, raw.r + 1.772 * cb);
        a = (p.nchan_src >= 4) ? raw.a : 1.0;
        src_is_xyz = false;
        break;
    }
    case ColorModel::Hsv: {
        c = hsv_to_rgb(raw.rgb);
        a = (p.nchan_src >= 4) ? raw.a : 1.0;
        src_is_xyz = false;
        break;
    }
    case ColorModel::Lab: {
        c = lab_to_xyz_d50(raw.r, raw.g, raw.b);
        a = (p.nchan_src >= 4) ? raw.a : 1.0;
        src_is_xyz = true;
        break;
    }
    case ColorModel::Lch: {
        float3 lab = lch_to_lab(raw.rgb);
        c = lab_to_xyz_d50(lab.x, lab.y, lab.z);
        a = (p.nchan_src >= 4) ? raw.a : 1.0;
        src_is_xyz = true;
        break;
    }
    case ColorModel::Xyz: {
        c = raw.rgb;
        a = (p.nchan_src >= 4) ? raw.a : 1.0;
        src_is_xyz = true;
        break;
    }
    case ColorModel::Yxy: {
        c = xyy_to_xyz(raw.rgb);
        a = (p.nchan_src >= 4) ? raw.a : 1.0;
        src_is_xyz = true;
        break;
    }
    case ColorModel::Oklab: {
        c = oklab_to_xyz_d50(raw.rgb);
        a = (p.nchan_src >= 4) ? raw.a : 1.0;
        src_is_xyz = true;
        break;
    }
    case ColorModel::Oklch: {
        c = oklab_to_xyz_d50(lch_to_lab(raw.rgb));
        a = (p.nchan_src >= 4) ? raw.a : 1.0;
        src_is_xyz = true;
        break;
    }
    default: { // Rgb, Gray, ScRgb, Multiband
        c = raw.rgb;
        a = (p.nchan_src >= 4) ? raw.a : 1.0;
        src_is_xyz = false;
        break;
    }
    }

    // ── 2. OETF^-1 -> linear light (skip for XYZ(D50)-valued models) ─────
    if (!src_is_xyz) {
        c = float3(decode_tf(c.r, p.transfer_src), decode_tf(c.g, p.transfer_src), decode_tf(c.b, p.transfer_src));
    }

    // ── 3. Un-premultiply if source carries premultiplied alpha ──────────
    if (p.alpha_src == 1u) {
        c /= max(a, 1e-8);
    }

    // ── 4. Matrix A: src linear RGB -> XYZ(D50). Skip if already XYZ(D50). ──
    float3 xyz = src_is_xyz ? c : float3(
        p.a[0] * c.x + p.a[1] * c.y + p.a[2]  * c.z,
        p.a[4] * c.x + p.a[5] * c.y + p.a[6]  * c.z,
        p.a[8] * c.x + p.a[9] * c.y + p.a[10] * c.z);

    // ── 5. Matrix B: XYZ(D50) -> dst linear RGB. Skip if dst is XYZ(D50)-valued. ──
    bool dst_is_xyz = p.model_dst == ColorModel::Lab || p.model_dst == ColorModel::Lch
        || p.model_dst == ColorModel::Xyz || p.model_dst == ColorModel::Yxy
        || p.model_dst == ColorModel::Oklab || p.model_dst == ColorModel::Oklch;

    float3 out3 = dst_is_xyz ? xyz : float3(
        p.b[0] * xyz.x + p.b[1] * xyz.y + p.b[2]  * xyz.z,
        p.b[4] * xyz.x + p.b[5] * xyz.y + p.b[6]  * xyz.z,
        p.b[8] * xyz.x + p.b[9] * xyz.y + p.b[10] * xyz.z);

    // ── 6. Premultiply for destination if needed ─────────────────────────
    if (p.alpha_dst == 1u) {
        out3 *= a;
    }

    // ── 7. Destination model encode ──────────────────────────────────────
    switch (p.model_dst) {
    case ColorModel::Lab:
        return float4(xyz_d50_to_lab(out3), a);
    case ColorModel::Lch:
        return float4(lab_to_lch(xyz_d50_to_lab(out3)), a);
    case ColorModel::Xyz:
        return float4(out3, a);
    case ColorModel::Yxy:
        return float4(xyz_to_xyy(out3), a);
    case ColorModel::Oklab:
        return float4(xyz_d50_to_oklab(out3), a);
    case ColorModel::Oklch:
        return float4(lab_to_lch(xyz_d50_to_oklab(out3)), a);
    case ColorModel::Cmyk: {
        float k = 1.0 - max(out3.r, max(out3.g, out3.b));
        float inv = 1.0 - k;
        float3 cmy = inv > 1e-6 ? (float3(1.0) - out3 / inv) : float3(0.0);
        return float4(cmy, k);
    }
    case ColorModel::YCbCr: {
        float3 enc = float3(encode_tf(out3.r, p.transfer_dst), encode_tf(out3.g, p.transfer_dst), encode_tf(out3.b, p.transfer_dst));
        float y = 0.299 * enc.r + 0.587 * enc.g + 0.114 * enc.b;
        return float4(y, 0.5 + (enc.b - y) / 1.772, 0.5 + (enc.r - y) / 1.402, a);
    }
    case ColorModel::Hsv: {
        float3 enc = float3(encode_tf(out3.r, p.transfer_dst), encode_tf(out3.g, p.transfer_dst), encode_tf(out3.b, p.transfer_dst));
        return float4(rgb_to_hsv(enc), a);
    }
    default: // Rgb, Gray, ScRgb, Multiband
        return float4(
            encode_tf(out3.r, p.transfer_dst),
            encode_tf(out3.g, p.transfer_dst),
            encode_tf(out3.b, p.transfer_dst),
            a);
    }
}

// ── CmykA-source kernel (§6.1.3) ──────────────────────────────────────────
//
// A 5-channel source (e.g. CmykA) can't ride through the generic float4
// `ColorReadView` wrap: `decode` only yields the first 4 channels (C, M, Y,
// K) and the real alpha lives in the codec's `extra` 5th channel. This kernel
// binds the source `CodecRegion<C, N>` directly so it can call both
// `decode` and `decode_extra`, then runs the same `color_convert` hub as
// every other path before handing off to the destination's writable region.
void convert_cmyka_kernel<C : IStorageCodec, let N : uint, W : IWritableRegion>(
    uint2 idx, CodecRegion<C, N> input, W output, ColorConvertParams cc)
{
    if (idx.x >= input.view.width || idx.y >= input.view.height) { return; }
    uint i = region_index(input.view, idx.x, idx.y);
    float4 raw = C.decode(input.buf, i, N);
    float extra_alpha = C.decode_extra(input.buf, i, N);
    output.write(idx, color_convert(raw, extra_alpha, cc));
}
