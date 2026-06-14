// lib/codecs.slang — Storage codec implementations
//
// `IStorageCodec` interface + U8, U16, F16, F32 codecs. Pure quantizers:
// bytes <-> normalized float4 (+ a 5th `extra` channel for 5-band layouts
// such as CmykA). They know only `(storage, channel count)` — never the
// color model, transfer function, or any matrix. Color math lives in the
// `Convert` operation / the view-wrap color wrappers, not here.

import lib.pixel;

// ── IStorageCodec interface ───────────────────────────────────────────────────

// Output buffers are typed `Atomic<uint>`: sub-word formats (1/2/3-byte
// strides, …) pack several channels into one u32, and adjacent pixels share
// a word, so concurrent writes must be atomic read-modify-write. Word-aligned
// formats (4-channel u8, f32) just `store` whole words. Input/decode reads
// from a plain `StructuredBuffer<uint>` (a separate src buffer, no
// contention).
interface IStorageCodec
{
    // Decode up to 4 channels into rgba (missing channels => 0, alpha => 1 if absent).
    static float4 decode(StructuredBuffer<uint> buf, uint idx, uint nchan);
    // 5th channel (e.g. CmykA's K) or 1.0 if nchan < 5.
    static float  decode_extra(StructuredBuffer<uint> buf, uint idx, uint nchan);
    static void   encode(RWStructuredBuffer<Atomic<uint> > buf, uint idx, float4 c, uint nchan);
    static void   encode_extra(RWStructuredBuffer<Atomic<uint> > buf, uint idx, float v, uint nchan);
};

// ── Byte-level helpers (shared across codecs) ─────────────────────────────────

uint byte_buf(uint bo, StructuredBuffer<uint> buf)
{
    uint wi = bo / 4u;
    uint shift = (bo % 4u) * 8u;
    return (buf[wi] >> shift) & 0xFFu;
}

float _u16_norm(StructuredBuffer<uint> buf, uint bo)
{
    uint wi  = bo / 4u;
    uint rem = bo % 4u;
    uint w   = buf[wi];
    uint val;
    if      (rem == 0u) val = w & 0xFFFFu;
    else if (rem == 2u) val = (w >> 16u) & 0xFFFFu;
    else                val = ((w >> (rem * 8u)) | (buf[wi + 1u] << (32u - rem * 8u))) & 0xFFFFu;
    return float(val) / 65535.0;
}

void _write_byte(RWStructuredBuffer<Atomic<uint> > buf, uint bo, uint v)
{
    uint shift = (bo % 4u) * 8u;
    uint mask = ~(0xFFu << shift);
    uint bits = (v & 0xFFu) << shift;

    uint wi = bo / 4u;
    buf[wi].and(mask);
    buf[wi].or(bits);
}

void _write_u16(RWStructuredBuffer<Atomic<uint> > buf, uint bo, uint v)
{
    uint rem = bo % 4u;
    uint wi = bo / 4u;
    if (rem == 0u) {
        buf[wi].and(0xFFFF0000u);
        buf[wi].or(v & 0xFFFFu);
    } else if (rem == 2u) {
        buf[wi].and(0x0000FFFFu);
        buf[wi].or((v & 0xFFFFu) << 16u);
    } else {
        _write_byte(buf, bo,      v & 0xFFu);
        _write_byte(buf, bo + 1u, (v >> 8u) & 0xFFu);
    }
}

// ── F16 Software Decoder/Encoder ─────────────────────────────────────────────
// Manual bit-twiddling avoids SPIR-V OpTypeInt 16 emitting,
// which satisfies Naga/WGPU strict validation natively without shaderInt16.

float _half_to_float(uint h) {
    uint s = (h & 0x8000u) << 16u;
    uint em = h & 0x7FFFu;
    if (em == 0u) return asfloat(s);

    uint e = em >> 10u;
    uint m = em & 0x03FFu;

    if (e == 0u) {
        m <<= 1u;
        e = 0u;
        while ((m & 0x0400u) == 0u) { m <<= 1u; e++; }
        m &= 0x03FFu;
        return asfloat(s | ((127u - 15u - e) << 23u) | (m << 13u));
    }
    if (e == 31u) return asfloat(s | 0x7F800000u | (m << 13u));
    return asfloat(s | ((e - 15u + 127u) << 23u) | (m << 13u));
}

uint _float_to_half(float f) {
    uint u = asuint(f);
    uint s = (u >> 16u) & 0x8000u;
    uint e = (u >> 23u) & 0xFFu;
    uint m = u & 0x007FFFFFu;

    if (e == 0u) return s;
    if (e == 255u) {
        if (m == 0u) return s | 0x7C00u;
        return s | 0x7C00u | (m >> 13u);
    }

    int i = int(e) - 127 + 15;
    if (i <= 0) {
        uint sub_m = (0x00800000u | m) >> uint(1 - i);
        return s | (sub_m >> 13u);
    }
    if (i >= 31) return s | 0x7C00u;
    return s | (uint(i) << 10u) | (m >> 13u);
}

/// Read one f16 from byte offset `bo` (handles any alignment).
float _f16_at(StructuredBuffer<uint> buf, uint bo)
{
    uint wi  = bo / 4u;
    uint rem = bo % 4u;
    uint w   = buf[wi];
    uint val;
    if      (rem == 0u) val = w & 0xFFFFu;
    else if (rem == 2u) val = (w >> 16u) & 0xFFFFu;
    else                val = ((w >> (rem * 8u)) | (buf[wi + 1u] << (32u - rem * 8u))) & 0xFFFFu;
    return _half_to_float(val);
}

/// Write one f16 to byte offset `bo` (handles any alignment).
void _write_f16(RWStructuredBuffer<Atomic<uint> > buf, uint bo, float v)
{
    uint h   = _float_to_half(v);
    uint rem = bo % 4u;
    uint wi = bo / 4u;
    if (rem == 0u) {
        buf[wi].and(0xFFFF0000u);
        buf[wi].or(h & 0xFFFFu);
    } else if (rem == 2u) {
        buf[wi].and(0x0000FFFFu);
        buf[wi].or((h & 0xFFFFu) << 16u);
    } else {
        _write_byte(buf, bo,      h & 0xFFu);
        _write_byte(buf, bo + 1u, (h >> 8u) & 0xFFu);
    }
}

// ── U8Codec ──────────────────────────────────────────────────────────────────
// 1 byte per sample.

struct U8Codec : IStorageCodec
{
    static float4 decode(StructuredBuffer<uint> buf, uint idx, uint nchan)
    {
        uint base = idx * nchan;
        float4 c = float4(0.0, 0.0, 0.0, 1.0);
        for (uint k = 0u; k < min(nchan, 4u); k++)
            c[k] = float(byte_buf(base + k, buf)) / 255.0;
        return c;
    }

    static float decode_extra(StructuredBuffer<uint> buf, uint idx, uint nchan)
    {
        return (nchan >= 5u) ? float(byte_buf(idx * nchan + 4u, buf)) / 255.0 : 1.0;
    }

    static void encode(RWStructuredBuffer<Atomic<uint> > buf, uint idx, float4 c, uint nchan)
    {
        uint base = idx * nchan;
        for (uint k = 0u; k < min(nchan, 4u); k++)
            _write_byte(buf, base + k, uint(saturate(c[k]) * 255.0 + 0.5));
    }

    static void encode_extra(RWStructuredBuffer<Atomic<uint> > buf, uint idx, float v, uint nchan)
    {
        if (nchan >= 5u)
            _write_byte(buf, idx * nchan + 4u, uint(saturate(v) * 255.0 + 0.5));
    }
};

// ── U16Codec ─────────────────────────────────────────────────────────────────
// 2 bytes per sample.

struct U16Codec : IStorageCodec
{
    static float4 decode(StructuredBuffer<uint> buf, uint idx, uint nchan)
    {
        uint base = idx * nchan * 2u;
        float4 c = float4(0.0, 0.0, 0.0, 1.0);
        for (uint k = 0u; k < min(nchan, 4u); k++)
            c[k] = _u16_norm(buf, base + k * 2u);
        return c;
    }

    static float decode_extra(StructuredBuffer<uint> buf, uint idx, uint nchan)
    {
        return (nchan >= 5u) ? _u16_norm(buf, idx * nchan * 2u + 8u) : 1.0;
    }

    static void encode(RWStructuredBuffer<Atomic<uint> > buf, uint idx, float4 c, uint nchan)
    {
        uint base = idx * nchan * 2u;
        for (uint k = 0u; k < min(nchan, 4u); k++)
            _write_u16(buf, base + k * 2u, uint(saturate(c[k]) * 65535.0 + 0.5));
    }

    static void encode_extra(RWStructuredBuffer<Atomic<uint> > buf, uint idx, float v, uint nchan)
    {
        if (nchan >= 5u)
            _write_u16(buf, idx * nchan * 2u + 8u, uint(saturate(v) * 65535.0 + 0.5));
    }
};

// ── F16Codec ─────────────────────────────────────────────────────────────────
// 2 bytes per sample, IEEE-754 half float. Not saturated on encode (HDR-safe).

struct F16Codec : IStorageCodec
{
    static float4 decode(StructuredBuffer<uint> buf, uint idx, uint nchan)
    {
        uint base = idx * nchan * 2u;
        float4 c = float4(0.0, 0.0, 0.0, 1.0);
        for (uint k = 0u; k < min(nchan, 4u); k++)
            c[k] = _f16_at(buf, base + k * 2u);
        return c;
    }

    static float decode_extra(StructuredBuffer<uint> buf, uint idx, uint nchan)
    {
        return (nchan >= 5u) ? _f16_at(buf, idx * nchan * 2u + 8u) : 1.0;
    }

    static void encode(RWStructuredBuffer<Atomic<uint> > buf, uint idx, float4 c, uint nchan)
    {
        uint base = idx * nchan * 2u;
        for (uint k = 0u; k < min(nchan, 4u); k++)
            _write_f16(buf, base + k * 2u, c[k]);
    }

    static void encode_extra(RWStructuredBuffer<Atomic<uint> > buf, uint idx, float v, uint nchan)
    {
        if (nchan >= 5u)
            _write_f16(buf, idx * nchan * 2u + 8u, v);
    }
};

// ── F32Codec ─────────────────────────────────────────────────────────────────
// 4 bytes per sample, one word each. Not saturated on encode (HDR-safe).

struct F32Codec : IStorageCodec
{
    static float4 decode(StructuredBuffer<uint> buf, uint idx, uint nchan)
    {
        uint base = idx * nchan;
        float4 c = float4(0.0, 0.0, 0.0, 1.0);
        for (uint k = 0u; k < min(nchan, 4u); k++)
            c[k] = asfloat(buf[base + k]);
        return c;
    }

    static float decode_extra(StructuredBuffer<uint> buf, uint idx, uint nchan)
    {
        return (nchan >= 5u) ? asfloat(buf[idx * nchan + 4u]) : 1.0;
    }

    static void encode(RWStructuredBuffer<Atomic<uint> > buf, uint idx, float4 c, uint nchan)
    {
        uint base = idx * nchan;
        for (uint k = 0u; k < min(nchan, 4u); k++)
            buf[base + k].store(asuint(c[k]));
    }

    static void encode_extra(RWStructuredBuffer<Atomic<uint> > buf, uint idx, float v, uint nchan)
    {
        if (nchan >= 5u)
            buf[idx * nchan + 4u].store(asuint(v));
    }
};
