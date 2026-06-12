struct BufferRegion {
    uint stride;
    uint x;
    uint y;
    uint width;
    uint height;
};

uint region_index(BufferRegion r, uint lx, uint ly) {
    return (r.y + ly) * r.stride + (r.x + lx);
}

uint region_index_clamped(BufferRegion r, int lx, int ly) {
    uint cx = uint(clamp(lx, 0, int(r.width) - 1));
    uint cy = uint(clamp(ly, 0, int(r.height) - 1));
    return (r.y + cy) * r.stride + (r.x + cx);
}

import lib.codecs;
import lib.pixel;

interface IRegion {
    float4 read(uint2 idx);
    float4 read_clamped(int2 idx);
}

struct CodecRegion<C: IPixelCodec, let CH: uint> : IRegion {
    StructuredBuffer<uint> buf;
    BufferRegion view;

    float4 read(uint2 idx) {
        if (idx.x >= view.width || idx.y >= view.height) { return float4(0); }
        return C.decode(buf, region_index(view, idx.x, idx.y), CH);
    }

    float4 read_clamped(int2 idx) {
        return C.decode(buf, region_index_clamped(view, idx.x, idx.y), CH);
    }
}

struct RWCodecRegion<C: IPixelCodec, let CH: uint> : IRegion {
    RWStructuredBuffer<Atomic<uint>> buf;
    BufferRegion view;

    float4 read(uint2 idx) { return float4(0); }
    float4 read_clamped(int2 idx) { return float4(0); }

    void write(uint2 idx, float4 value) {
        if (idx.x < view.width && idx.y < view.height) {
            C.encode(buf, region_index(view, idx.x, idx.y), value, (ChannelLayout)CH);
        }
    }
}


struct Region : IRegion {
    StructuredBuffer<float4> buf;
    BufferRegion view;

    float4 read(uint2 idx) {
        if (idx.x >= view.width || idx.y >= view.height) { return float4(0); }
        return buf[region_index(view, idx.x, idx.y)];
    }

    float4 read_clamped(int2 idx) {
        return buf[region_index_clamped(view, idx.x, idx.y)];
    }
}

struct RWRegion : IRegion {
    RWStructuredBuffer<float4> buf;
    BufferRegion view;

    float4 read(uint2 idx) {
        if (idx.x >= view.width || idx.y >= view.height) { return float4(0); }
        return buf[region_index(view, idx.x, idx.y)];
    }

    float4 read_clamped(int2 idx) {
        return buf[region_index_clamped(view, idx.x, idx.y)];
    }

    void write(uint2 idx, float4 value) {
        if (idx.x < view.width && idx.y < view.height) {
            buf[region_index(view, idx.x, idx.y)] = value;
        }
    }
}

// ── Band-view wrapper ───────────────────────────────────────────────────────
//
// Zero-cost "extract band" view: wraps ANY `IRegion` (a source decode or a
// prior step's temp), reads one component of its `float4`, and broadcasts it
// the same way a Gray codec decode would (`float4(v, v, v, 1)`), so a
// downstream kernel — or the Gray codec's luma-based encode
// (`0.2126 r + 0.7152 g + 0.0722 b == v` when r=g=b=v) — sees a normal
// one-band working pixel. No extra kernel pass: `ExtractBand` just aliases
// its input through this view (see `GpuBuilder::alias`).
struct SwizzleView<R: IRegion> : IRegion {
    R inner;
    uint channel;

    float4 read(uint2 idx) {
        float v = inner.read(idx)[channel];
        return float4(v, v, v, 1.0);
    }

    float4 read_clamped(int2 idx) {
        float v = inner.read_clamped(idx)[channel];
        return float4(v, v, v, 1.0);
    }
}

// ── Coordinate-remap view ─────────────────────────────────────────────────────
enum RemapKind : uint {
    Identity   = 0u,
    FlipH      = 1u,
    FlipV      = 2u,
    Rot180     = 3u,
    Scale      = 4u,
    Tile       = 5u,
    Translate  = 6u,
    Rot90      = 7u,
    Rot270     = 8u,
};

struct RemapView<R: IRegion> : IRegion {
    R inner;
    uint kind;
    uint out_w, out_h;
    float sx, sy;
    uint in_w, in_h;
    int tx, ty;

    int2 remap(int2 idx) {
        switch (kind) {
            case 1u /*FlipH*/: return int2(int(out_w) - 1 - idx.x, idx.y);
            case 2u /*FlipV*/: return int2(idx.x, int(out_h) - 1 - idx.y);
            case 3u /*Rot180*/: return int2(int(out_w) - 1 - idx.x, int(out_h) - 1 - idx.y);
            case 4u /*Scale*/: return int2(int(float(idx.x) * sx), int(float(idx.y) * sy));
            case 5u /*Tile*/: return int2(idx.x % int(in_w), idx.y % int(in_h));
            case 6u /*Translate*/: return int2(idx.x + tx, idx.y + ty);
            case 7u /*Rot90*/: return int2(idx.y, int(out_w) - 1 - idx.x);
            case 8u /*Rot270*/: return int2(int(out_h) - 1 - idx.y, idx.x);
            default: return idx;
        }
    }

    float4 read(uint2 idx) {
        int2 r = remap(int2(idx));
        return inner.read_clamped(r);
    }

    float4 read_clamped(int2 idx) {
        return inner.read_clamped(remap(idx));
    }
}

// ── Non-image buffer wrappers ─────────────────────────────────────────────────

struct ComplexRegion {
    StructuredBuffer<float2> buf;
    BufferRegion view;

    float2 read(uint idx) {
        return buf[view.x + idx];
    }

    float2 read2d(uint2 idx) {
        if (idx.x >= view.width || idx.y >= view.height) { return float2(0); }
        return buf[region_index(view, idx.x, idx.y)];
    }
}

struct RWComplexRegion {
    RWStructuredBuffer<float2> buf;
    BufferRegion view;

    float2 read2d(uint2 idx) {
        if (idx.x >= view.width || idx.y >= view.height) { return float2(0); }
        return buf[region_index(view, idx.x, idx.y)];
    }

    void write(uint idx, float2 value) {
        buf[view.x + idx] = value;
    }

    void write2d(uint2 idx, float2 value) {
        if (idx.x < view.width && idx.y < view.height) {
            buf[region_index(view, idx.x, idx.y)] = value;
        }
    }
}

struct MaskRegion {
    StructuredBuffer<float> buf;
    BufferRegion view;

    float read(uint idx) {
        return buf[view.x + idx];
    }

    float read2d(uint2 idx) {
        if (idx.x >= view.width || idx.y >= view.height) { return 0.0f; }
        return buf[region_index(view, idx.x, idx.y)];
    }
}

struct RWMaskRegion {
    RWStructuredBuffer<float> buf;
    BufferRegion view;

    float read(uint idx) {
        return buf[view.x + idx];
    }

    void write(uint idx, float value) {
        buf[view.x + idx] = value;
    }

    void write2d(uint2 idx, float value) {
        if (idx.x < view.width && idx.y < view.height) {
            buf[region_index(view, idx.x, idx.y)] = value;
        }
    }
}
