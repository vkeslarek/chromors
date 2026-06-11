// working.slang — bracket any pixel format/space around a canonical working
// space so ops (blur, compose, …) operate on a single, well-defined float4.
//
// Working space = SRGB Gamma Encoded, straight (un-premultiplied) alpha,
// RGBA float4. A `ColorSpace` is self-contained: it carries the model,
// transfer, alpha policy, channel layout, and the 3×3 matrix that takes it to
// (source) or from (destination) the SRGB Linear space. An op sandwiches its math:
//
//   c = to_working(codec.decode(...), extra, src)   // → SRGB Gamma straight
//   c = <op in working space>
//   from_working(c, dst, packed, extra)             // → dst format/space
//   codec.encode(packed, ...)

import lib.pixel;
import lib.codecs;
import lib.region;
import lib.color.transfer;
import lib.color.model;

/// Row-major 3×3 matrix, std430 layout (9 tightly-packed floats).
struct Matrix3 {
    float m0; float m1; float m2;
    float m3; float m4; float m5;
    float m6; float m7; float m8;
};

float3 mat3_apply(Matrix3 m, float3 v) {
    return float3(
        m.m0 * v.x + m.m1 * v.y + m.m2 * v.z,
        m.m3 * v.x + m.m4 * v.y + m.m5 * v.z,
        m.m6 * v.x + m.m7 * v.y + m.m8 * v.z);
}

/// Everything needed to move one side (source or destination) between its
/// stored representation and the working space. `transform` is space→working
/// for a source, working→space for a destination.
struct ColorSpace {
    TransferFn    transfer;
    AlphaPolicy   alpha;
    ColorModel    model;
    ChannelLayout channels;
    Matrix3       transform;
};

// ── Lab ↔ XYZ(D50) (only used by the Lab model hop) ─────────────────────────

static const float LAB_DELTA  = 6.0 / 29.0;
static const float LAB_DELTA2 = 3.0 * LAB_DELTA * LAB_DELTA;
static const float LAB_DELTA3 = LAB_DELTA * LAB_DELTA * LAB_DELTA;
static const float XN_D50 = 0.96422;
static const float YN_D50 = 1.0;
static const float ZN_D50 = 0.82521;

static const float3 SRGB_FROM_XYZ_R = float3( 3.1338561, -1.6168667, -0.4906146);
static const float3 SRGB_FROM_XYZ_G = float3(-0.9787684,  1.9161415,  0.0334540);
static const float3 SRGB_FROM_XYZ_B = float3( 0.0719453, -0.2289914,  1.4052427);
static const float3 XYZ_FROM_SRGB_R = float3( 0.4360747,  0.3850649,  0.1430804);
static const float3 XYZ_FROM_SRGB_G = float3( 0.2225045,  0.7168786,  0.0606169);
static const float3 XYZ_FROM_SRGB_B = float3( 0.0139322,  0.0971045,  0.7141733);

float3 lab_to_linear_srgb(float3 lab) {
    float L = lab.x * 100.0, a = lab.y * 128.0, b = lab.z * 128.0;
    float fy = (L + 16.0) / 116.0;
    float fx = a / 500.0 + fy;
    float fz = fy - b / 200.0;
    float x = XN_D50 * (fx > LAB_DELTA ? fx*fx*fx : LAB_DELTA2*(fx - 4.0/29.0));
    float y = YN_D50 * (fy > LAB_DELTA ? fy*fy*fy : LAB_DELTA2*(fy - 4.0/29.0));
    float z = ZN_D50 * (fz > LAB_DELTA ? fz*fz*fz : LAB_DELTA2*(fz - 4.0/29.0));
    return float3(dot(SRGB_FROM_XYZ_R, float3(x,y,z)),
                  dot(SRGB_FROM_XYZ_G, float3(x,y,z)),
                  dot(SRGB_FROM_XYZ_B, float3(x,y,z)));
}

float3 linear_srgb_to_lab(float3 rgb) {
    float x = dot(XYZ_FROM_SRGB_R, rgb);
    float y = dot(XYZ_FROM_SRGB_G, rgb);
    float z = dot(XYZ_FROM_SRGB_B, rgb);
    float fx = x / XN_D50; fx = fx > LAB_DELTA3 ? pow(fx, 1.0/3.0) : (1.0/LAB_DELTA2)*fx + 4.0/29.0;
    float fy = y / YN_D50; fy = fy > LAB_DELTA3 ? pow(fy, 1.0/3.0) : (1.0/LAB_DELTA2)*fy + 4.0/29.0;
    float fz = z / ZN_D50; fz = fz > LAB_DELTA3 ? pow(fz, 1.0/3.0) : (1.0/LAB_DELTA2)*fz + 4.0/29.0;
    return float3((116.0*fy - 16.0)/100.0, (500.0*(fx-fy))/128.0, (200.0*(fy-fz))/128.0);
}

// ── to_working: stored representation → sRGB Gamma straight RGBA ────────────

/// `extra`: 5th channel for CmykA (from `decode_extra`); 1.0 otherwise.
float4 to_working(float4 raw, float extra, ColorSpace cs) {
    float3 rgb = raw.rgb;
    float  a   = raw.a;
    bool is_lab = false;

    if (cs.model == ColorModel::CmykToRgb || cs.model == ColorModel::CmykAToRgb) {
        float scale = 1.0 - raw.a;            // raw.a == K
        rgb = (float3(1.0) - raw.rgb) * scale;
        a   = (cs.model == ColorModel::CmykAToRgb) ? extra : 1.0;
    } else if (cs.model == ColorModel::YCbCrToRgb) {
        float cb = raw.g - 0.5, cr = raw.b - 0.5;
        rgb = float3(raw.r + 1.402*cr, raw.r - 0.344136*cb - 0.714136*cr, raw.r + 1.772*cb);
    } else if (cs.model == ColorModel::Lab) {
        rgb = lab_to_linear_srgb(raw.rgb);
        a = 1.0;
        is_lab = true;                        // already linear sRGB
    }

    if (!is_lab) {
        rgb = float3(
            decode_tf(rgb.r, (TransferFn)cs.transfer),
            decode_tf(rgb.g, (TransferFn)cs.transfer),
            decode_tf(rgb.b, (TransferFn)cs.transfer)
        );
    }
    rgb = mat3_apply(cs.transform, rgb);
    
    // Encode linear SRGB to working space (SRGB Gamma)
    rgb = float3(
        encode_tf(rgb.r, TransferFn::Srgb),
        encode_tf(rgb.g, TransferFn::Srgb),
        encode_tf(rgb.b, TransferFn::Srgb)
    );

    if (cs.alpha == AlphaPolicy::PremultiplyOnPack) {
        rgb /= max(a, 1e-8);
    }

    return float4(rgb, a);
}

// ── from_working: sRGB Gamma straight RGBA → stored representation ───────────

/// Inverse of `to_working`. `cs.transform` is working→dst here. Returns packed
/// channels in `out_rgba` and the CmykA 5th channel in `out_extra`.
void from_working(float4 c, ColorSpace cs, out float4 out_rgba, out float out_extra) {
    // Decode working space (SRGB Gamma) back to linear SRGB
    float3 rgb = float3(
        decode_tf(c.r, TransferFn::Srgb),
        decode_tf(c.g, TransferFn::Srgb),
        decode_tf(c.b, TransferFn::Srgb)
    );
    rgb = mat3_apply(cs.transform, rgb);
    float  a   = c.a;
    out_extra  = 1.0;

    if (cs.alpha == AlphaPolicy::PremultiplyOnPack || cs.alpha == AlphaPolicy::OpaqueDrop) {
        rgb *= a;
    }

    if (cs.model == ColorModel::Lab) {
        out_rgba = float4(linear_srgb_to_lab(rgb), a);
        return;
    }
    if (cs.model == ColorModel::YCbCrToRgb) {
        float r = rgb.x;
        float g = rgb.y;
        float b = rgb.z;
        float y = 0.299*r + 0.587*g + 0.114*b;
        out_rgba = float4(y, 0.5 + (b - y)/1.772, 0.5 + (r - y)/1.402, a);
        return;
    }
    if (cs.model == ColorModel::CmykToRgb || cs.model == ColorModel::CmykAToRgb) {
        float r = rgb.x;
        float g = rgb.y;
        float b = rgb.z;
        float k = 1.0 - max(r, max(g, b));
        float inv = 1.0 - k;
        float3 cmy = inv > 1e-6 ? (float3(1.0) - float3(r,g,b)/inv) : float3(0.0);
        out_extra = (cs.model == ColorModel::CmykAToRgb) ? a : 1.0;
        out_rgba  = float4(cmy, k);
        return;
    }

    rgb = float3(
        encode_tf(rgb.r, (TransferFn)cs.transfer),
        encode_tf(rgb.g, (TransferFn)cs.transfer),
        encode_tf(rgb.b, (TransferFn)cs.transfer)
    );
    out_rgba = float4(rgb, a);
}

// ── OOP views: read/write a buffer straight in working space ────────────────
//
// Hide the decode → to_working and from_working → encode boilerplate so a
// kernel's core deals only with `float4` in ACEScg.

/// Reads a (possibly sub-rect) source buffer as working-space RGBA.
struct WorkingSource<C : IPixelCodec> {
    StructuredBuffer<uint> buf;
    BufferRegion             region;
    ColorSpace               space;

    float4 sample_clamped(int lx, int ly) {
        uint idx = region_index_clamped(region, lx, ly);
        float4 raw   = C.decode(buf, idx, space.channels);
        float  extra = C.decode_extra(buf, idx, space.channels);
        return to_working(raw, extra, space);
    }

    float4 sample_transparent(int lx, int ly) {
        if (lx < 0 || ly < 0 || lx >= region.width || ly >= region.height) {
            return float4(0.0);
        }
        uint idx = region_index(region, uint(lx), uint(ly));
        float4 raw   = C.decode(buf, idx, space.channels);
        float  extra = C.decode_extra(buf, idx, space.channels);
        return to_working(raw, extra, space);
    }

    float4 sample(uint lx, uint ly) {
        uint idx = region_index(region, lx, ly);
        float4 raw   = C.decode(buf, idx, space.channels);
        float  extra = C.decode_extra(buf, idx, space.channels);
        return to_working(raw, extra, space);
    }
}

/// Writes working-space RGBA into a tightly-packed destination buffer. The
/// buffer is `Atomic<uint>` because sub-word codecs (Rgb8, Gray8, …) need
/// atomic read-modify-write to pack channels that share a u32 word.
struct WorkingSink<C : IPixelCodec> {
    RWStructuredBuffer<Atomic<uint> > buf;
    uint                              stride;   // dst is tight: stride == width
    ColorSpace                        space;

    void write(uint x, uint y, float4 working) {
        float4 packed;
        float  extra;
        from_working(working, space, packed, extra);
        C.encode(buf, y * stride + x, packed, space.channels);
    }
}

// ── IRegion wrappers for the fused POC pipeline ──────────────────────────────

/// Wraps any IRegion (e.g. CodecRegion) and applies `to_working` on every read.
/// Intermediate temp buffers stay in working space — only source regions get this.
struct WorkingDecodeRegion<R: IRegion> : IRegion {
    R inner;
    ColorSpace src_cs;

    float4 read(uint2 idx) {
        return to_working(inner.read(idx), 1.0, src_cs);
    }
    float4 read_clamped(int2 idx) {
        return to_working(inner.read_clamped(idx), 1.0, src_cs);
    }
}

/// Encodes a working-space float4 via `from_working` and writes to a buffer.
void working_encode_write(RWStructuredBuffer<float4> buf, BufferRegion view, uint2 idx, float4 c, ColorSpace dst_cs) {
    float4 packed;
    float extra;
    from_working(c, dst_cs, packed, extra);
    uint linear = region_index(view, idx.x, idx.y);
    if (idx.x < view.width && idx.y < view.height) {
        buf[linear] = packed;
    }
}
