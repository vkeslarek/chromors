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
