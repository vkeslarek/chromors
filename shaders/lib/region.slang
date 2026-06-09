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
