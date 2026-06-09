// io.slang — Slang ergonomics layer for non-Image2D I/O types (PHASE 4)
//
// Provides GrayIn, GrayOut, HistogramOut, PointOut plus FlipReader and
// RotateReader for the zero-copy view variants (PHASE 5).

import lib.region;

// ── Gray channel reader ───────────────────────────────────────────────────────

/// Reads one channel from an RGBA F32 buffer, returning it as a grayscale float4.
struct GrayIn {
    StructuredBuffer<uint> buf;
    BufferRegion view;
    uint channel_index; // 0=R, 1=G, 2=B, 3=A

    float4 read(uint2 idx) {
        uint base = region_index(view, idx.x, idx.y) * 4u;
        float v = asfloat(buf[base + channel_index]);
        return float4(v, v, v, 1.0);
    }

    float4 read_clamped(int2 idx) {
        uint2 c = uint2(clamp(idx.x, 0, int(view.width) - 1),
                        clamp(idx.y, 0, int(view.height) - 1));
        return read(c);
    }
}

// ── Gray scalar writer ────────────────────────────────────────────────────────

/// Writes a single float per pixel (for GrayF32 outputs).
struct GrayOut {
    RWStructuredBuffer<float> buf;
    BufferRegion view;

    void write(uint2 idx, float v) {
        buf[region_index(view, idx.x, idx.y)] = v;
    }
}

// ── Histogram accumulator ────────────────────────────────────────────────────

/// Fixed-size atomic histogram accumulator.
struct HistogramOut {
    RWStructuredBuffer<uint> bins;
    uint bin_count;

    void accumulate(float value) {
        uint bin = uint(clamp(value, 0.0, 0.9999) * float(bin_count));
        InterlockedAdd(bins[bin], 1u);
    }
}

// ── Atomic point list ────────────────────────────────────────────────────────

/// Atomic-append point list. Counter at buf[0]; points start at buf[1].
struct PointOut {
    RWStructuredBuffer<uint> buf;
    uint capacity;

    void append(uint2 coord) {
        uint slot = 0u;
        InterlockedAdd(buf[0], 1u, slot);
        if (slot < capacity) {
            buf[1u + slot * 2u] = coord.x;
            buf[2u + slot * 2u] = coord.y;
        }
    }
}

// ── Flip reader ──────────────────────────────────────────────────────────────

/// Reads from a buffer with optional horizontal/vertical axis flip.
struct FlipReader {
    StructuredBuffer<uint> buf;
    BufferRegion view;
    uint flip_h; // 0 or 1
    uint flip_v; // 0 or 1

    float4 read(uint2 idx) {
        uint x = flip_h != 0u ? (view.width - 1u - idx.x) : idx.x;
        uint y = flip_v != 0u ? (view.height - 1u - idx.y) : idx.y;
        // Decode as float4 from raw uint buffer (F32Codec RGBA layout)
        uint base = region_index(view, x, y) * 4u;
        return float4(asfloat(buf[base]), asfloat(buf[base + 1u]),
                      asfloat(buf[base + 2u]), asfloat(buf[base + 3u]));
    }

    float4 read_clamped(int2 idx) {
        return read(uint2(clamp(idx.x, 0, int(view.width) - 1),
                          clamp(idx.y, 0, int(view.height) - 1)));
    }
}

// ── Rotate reader ────────────────────────────────────────────────────────────

/// Reads from a buffer with 90/180/270-degree CW rotation.
struct RotateReader {
    StructuredBuffer<uint> buf;
    BufferRegion view;
    uint rotation; // 1=90 CW, 2=180, 3=270 CW

    float4 read(uint2 idx) {
        uint2 src;
        if (rotation == 1u) {
            src = uint2(idx.y, view.height - 1u - idx.x); // 90 CW
        } else if (rotation == 2u) {
            src = uint2(view.width - 1u - idx.x, view.height - 1u - idx.y); // 180
        } else {
            src = uint2(view.width - 1u - idx.y, idx.x); // 270 CW
        }
        uint base = region_index(view, src.x, src.y) * 4u;
        return float4(asfloat(buf[base]), asfloat(buf[base + 1u]),
                      asfloat(buf[base + 2u]), asfloat(buf[base + 3u]));
    }

    float4 read_clamped(int2 idx) {
        return read(uint2(idx));
    }
}
