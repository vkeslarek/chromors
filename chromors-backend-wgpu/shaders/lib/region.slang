struct BufferRegion {
    uint stride;
    uint x;
    uint y;
    uint width;
    uint height;
    int pad_x;
    int pad_y;
};

int region_index(BufferRegion r, int lx, int ly) {
    int cx = lx - r.pad_x;
    int cy = ly - r.pad_y;
    if (cx < 0 || cx >= int(r.width) || cy < 0 || cy >= int(r.height)) { return -1; }
    return int((r.y + uint(cy)) * r.stride + (r.x + uint(cx)));
}

uint region_index_clamped(BufferRegion r, int lx, int ly) {
    uint cx = uint(clamp(lx - r.pad_x, 0, int(r.width) - 1));
    uint cy = uint(clamp(ly - r.pad_y, 0, int(r.height) - 1));
    return (r.y + cy) * r.stride + (r.x + cx);
}

import lib.codecs;
import lib.pixel;

interface IRegion {
    float4 read(uint2 idx);
    float4 read_clamped(int2 idx);
}

/// A region that can be written to. Conformed by every `RW*Region` so a
/// generic write-side wrapper (`ColorWriteSink<W: IWritableRegion>`, §5.6) can
/// nest around any of them.
interface IWritableRegion {
    void write(uint2 idx, float4 value);
}

struct CodecRegion<C: IStorageCodec, let N: uint> : IRegion {
    StructuredBuffer<uint> buf;
    BufferRegion view;

    float4 read(uint2 idx) {
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float4(0); }
        return C.decode(buf, uint(idx_1d), N);
    }

    float4 read_clamped(int2 idx) {
        return C.decode(buf, region_index_clamped(view, idx.x, idx.y), N);
    }
}

struct RWCodecRegion<C: IStorageCodec, let N: uint> : IRegion, IWritableRegion {
    RWStructuredBuffer<Atomic<uint>> buf;
    BufferRegion view;

    float4 read(uint2 idx) { return float4(0); }
    float4 read_clamped(int2 idx) { return float4(0); }

    void write(uint2 idx, float4 value) {
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d >= 0) {
            C.encode(buf, uint(idx_1d), value, N);
        }
    }
}


struct Region : IRegion {
    StructuredBuffer<float4> buf;
    BufferRegion view;

    float4 read(uint2 idx) {
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float4(0); }
        return buf[uint(idx_1d)];
    }

    float4 read_clamped(int2 idx) {
        return buf[region_index_clamped(view, idx.x, idx.y)];
    }
}

struct RWRegion : IRegion, IWritableRegion {
    RWStructuredBuffer<float4> buf;
    BufferRegion view;

    float4 read(uint2 idx) {
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float4(0); }
        return buf[uint(idx_1d)];
    }

    float4 read_clamped(int2 idx) {
        return buf[region_index_clamped(view, idx.x, idx.y)];
    }

    void write(uint2 idx, float4 value) {
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d >= 0) {
            buf[uint(idx_1d)] = value;
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

// Geometry payload for `RemapView` — layout matches Rust `RemapParams`
// (`operation::geometry::RemapParams`, `#[repr(C)]` + `bytemuck::Pod`).
struct RemapGeo {
    uint out_w, out_h;
    float sx, sy;
    uint in_w, in_h;
    int tx, ty;
};

struct RemapView<R: IRegion> : IRegion {
    R inner;
    uint kind;
    RemapGeo geo;

    int2 remap(int2 idx) {
        switch (kind) {
            case 1u /*FlipH*/: return int2(int(geo.out_w) - 1 - idx.x, idx.y);
            case 2u /*FlipV*/: return int2(idx.x, int(geo.out_h) - 1 - idx.y);
            case 3u /*Rot180*/: return int2(int(geo.out_w) - 1 - idx.x, int(geo.out_h) - 1 - idx.y);
            case 4u /*Scale*/: return int2(int(float(idx.x) * geo.sx), int(float(idx.y) * geo.sy));
            case 5u /*Tile*/: return int2(idx.x % int(geo.in_w), idx.y % int(geo.in_h));
            case 6u /*Translate*/: return int2(idx.x + geo.tx, idx.y + geo.ty);
            case 7u /*Rot90*/: return int2(idx.y, int(geo.out_w) - 1 - idx.x);
            case 8u /*Rot270*/: return int2(int(geo.out_h) - 1 - idx.y, idx.x);
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
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float2(0); }
        return buf[uint(idx_1d)];
    }
}

struct RWComplexRegion {
    RWStructuredBuffer<float2> buf;
    BufferRegion view;

    float2 read2d(uint2 idx) {
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float2(0); }
        return buf[uint(idx_1d)];
    }

    void write(uint idx, float2 value) {
        buf[view.x + idx] = value;
    }

    void write2d(uint2 idx, float2 value) {
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d >= 0) {
            buf[uint(idx_1d)] = value;
        }
    }
}

// ── Mask view ──────────────────────────────────────────────────────────────
//
// Raw f32 read/write, broadcast to float4(v,v,v,1) like a Gray decode — no
// codec, no color space. Backs `Mask2DKind` (convolution masks, morphology
// elements, recombination matrices).
struct MaskRegion : IRegion {
    StructuredBuffer<float> buf;
    BufferRegion view;

    float4 read(uint2 idx) {
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float4(0); }
        float v = buf[uint(idx_1d)];
        return float4(v, v, v, 1.0);
    }

    float4 read_clamped(int2 idx) {
        float v = buf[region_index_clamped(view, idx.x, idx.y)];
        return float4(v, v, v, 1.0);
    }
}

struct RWMaskRegion : IRegion, IWritableRegion {
    RWStructuredBuffer<float> buf;
    BufferRegion view;

    float4 read(uint2 idx) {
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float4(0); }
        float v = buf[uint(idx_1d)];
        return float4(v, v, v, 1.0);
    }

    float4 read_clamped(int2 idx) {
        float v = buf[region_index_clamped(view, idx.x, idx.y)];
        return float4(v, v, v, 1.0);
    }

    void write(uint2 idx, float4 value) {
        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d >= 0) {
            buf[uint(idx_1d)] = value.x;
        }
    }
}
