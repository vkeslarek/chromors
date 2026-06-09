// lib/convert.slang — XYZ-hub colour conversion kernel.
//
// Pipeline per pixel:
//   src bytes
//     → codec.decode                       (raw channels as float4)
//     → model_decode (Lab/CMYK/YCbCr → linear RGB)
//     → transfer_decode_src                (OETF⁻¹: encoded → linear, skipped for Lab)
//     → unpremultiply if needed
//     → matrix A (linear src RGB → CIE XYZ at D50)
//     → matrix B (CIE XYZ at D50 → linear dst RGB)
//     → premultiply if needed
//     → model_encode (linear RGB → Lab/...) OR transfer_encode_dst (OETF)
//     → codec.encode                       → dst bytes
//
// XYZ is the hub — every space converts through it, so adding a new model
// only needs `<model>↔XYZ` math, not pairwise `<model>↔<every other model>`.

import lib.codecs;
import lib.pixel;
import lib.color.params;
import lib.color.transfer;
import lib.color.model;
import lib.region;

// ── Lab ↔ XYZ (D50) — local helpers ──────────────────────────────────────

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

// ── color_convert ────────────────────────────────────────────────────────

/// `extra_alpha`: 5th channel for CmykA (from `decode_extra`); 1.0 otherwise.
float4 color_convert(float4 raw, float extra_alpha, ColorConvertParams p)
{
    // ── 1. Source model decode → linear RGB (or XYZ-equivalent for Lab) ──
    float r = raw.r, g = raw.g, b = raw.b, a = raw.a;
    bool src_is_xyz = false;

    if (p.src_model == ColorModel::CmykToRgb || p.src_model == ColorModel::CmykAToRgb) {
        float scale = 1.0 - raw.a;                 // raw.a == K
        r = (1.0 - raw.r) * scale;
        g = (1.0 - raw.g) * scale;
        b = (1.0 - raw.b) * scale;
        a = (p.src_model == ColorModel::CmykAToRgb) ? extra_alpha : 1.0;
    } else if (p.src_model == ColorModel::YCbCrToRgb) {
        float cb = raw.g - 0.5;
        float cr = raw.b - 0.5;
        r = raw.r + 1.402 * cr;
        g = raw.r - 0.344136 * cb - 0.714136 * cr;
        b = raw.r + 1.772 * cb;
    } else if (p.src_model == ColorModel::Lab) {
        float3 xyz = lab_to_xyz_d50(raw.r, raw.g, raw.b);
        r = xyz.x; g = xyz.y; b = xyz.z;
        a = 1.0;
        src_is_xyz = true; // skip transfer + matrix A (we're already at the XYZ hub)
    }

    // ── 2. OETF⁻¹ → linear light (skip for Lab path) ─────────────────────
    if (!src_is_xyz) {
        r = decode_tf(r, p.transfer_src);
        g = decode_tf(g, p.transfer_src);
        b = decode_tf(b, p.transfer_src);
    }

    // ── 3. Un-premultiply if source carries premultiplied alpha ──────────
    if (p.alpha_policy_src == AlphaPolicy::PremultiplyOnPack) {
        float safe_a = max(a, 1e-8);
        r /= safe_a; g /= safe_a; b /= safe_a;
    }

    // ── 4. Matrix A: src linear RGB → CIE XYZ (D50). Skip if already XYZ. ──
    float xx, yy, zz;
    if (src_is_xyz) {
        xx = r; yy = g; zz = b;
    } else {
        xx = p.a00 * r + p.a01 * g + p.a02 * b;
        yy = p.a10 * r + p.a11 * g + p.a12 * b;
        zz = p.a20 * r + p.a21 * g + p.a22 * b;
    }

    // ── 5. Matrix B: CIE XYZ → dst linear RGB. Skip if dst is Lab/XYZ. ───
    float ro, go, bo;
    bool dst_is_xyz = (p.dst_model == ColorModel::Lab);
    if (dst_is_xyz) {
        ro = xx; go = yy; bo = zz;
    } else {
        ro = p.b00 * xx + p.b01 * yy + p.b02 * zz;
        go = p.b10 * xx + p.b11 * yy + p.b12 * zz;
        bo = p.b20 * xx + p.b21 * yy + p.b22 * zz;
    }

    // ── 6. Premultiply for destination if needed ────────────────────────
    if (p.alpha_policy_dst == AlphaPolicy::PremultiplyOnPack
        || p.alpha_policy_dst == AlphaPolicy::OpaqueDrop) {
        ro *= a; go *= a; bo *= a;
    }

    // ── 7. Destination model encode ─────────────────────────────────────
    if (p.dst_model == ColorModel::Lab) {
        float3 lab = xyz_d50_to_lab(float3(ro, go, bo));
        return float4(lab.x, lab.y, lab.z, a);
    }

    // ── 8. OETF for RGB-family destinations ─────────────────────────────
    return float4(
        encode_tf(ro, p.transfer_dst),
        encode_tf(go, p.transfer_dst),
        encode_tf(bo, p.transfer_dst),
        a
    );
}

// ── Generic kernel template ─────────────────────────────────────────────

void cc_kernel<S : IPixelCodec, D : IPixelCodec>(
    uint3                              id,
    StructuredBuffer<uint>             inputBuffer,
    RWStructuredBuffer<Atomic<uint> >  outputBuffer,
    ColorConvertParams                 p
) {
    if (id.x >= p.out_width || id.y >= p.out_height) return;
    uint src_idx = region_index(p.src_region, id.x, id.y);
    uint dst_idx = id.y * p.out_width + id.x;
    float4 raw   = S.decode(inputBuffer, src_idx, p.src_channels);
    float  extra = S.decode_extra(inputBuffer, src_idx, p.src_channels);
    D.encode(outputBuffer, dst_idx, color_convert(raw, extra, p), p.dst_channels);
}
