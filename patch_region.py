import re

with open("shaders/lib/region.slang", "r") as f:
    code = f.read()

# Update BufferRegion struct
code = code.replace(
"""struct BufferRegion {
    uint stride;
    uint x;
    uint y;
    uint width;
    uint height;
};""",
"""struct BufferRegion {
    uint stride;
    uint x;
    uint y;
    uint width;
    uint height;
    int pad_x;
    int pad_y;
};"""
)

# Update region_index_clamped
code = code.replace(
"""uint region_index_clamped(BufferRegion r, int lx, int ly) {
    uint cx = uint(clamp(lx, 0, int(r.width) - 1));
    uint cy = uint(clamp(ly, 0, int(r.height) - 1));
    return (r.y + cy) * r.stride + (r.x + cx);
}""",
"""uint region_index_clamped(BufferRegion r, int lx, int ly) {
    uint cx = uint(clamp(lx - r.pad_x, 0, int(r.width) - 1));
    uint cy = uint(clamp(ly - r.pad_y, 0, int(r.height) - 1));
    return (r.y + cy) * r.stride + (r.x + cx);
}"""
)

# Replace all plain region_index calls with inline bounds checks
# We can just change region_index to take int and return an int, -1 if OOB

code = code.replace(
"""uint region_index(BufferRegion r, uint lx, uint ly) {
    return (r.y + ly) * r.stride + (r.x + lx);
}""",
"""int region_index(BufferRegion r, int lx, int ly) {
    int cx = lx - r.pad_x;
    int cy = ly - r.pad_y;
    if (cx < 0 || cx >= int(r.width) || cy < 0 || cy >= int(r.height)) { return -1; }
    return int((r.y + uint(cy)) * r.stride + (r.x + uint(cx)));
}"""
)

# Update CodecRegion::read
code = code.replace(
"""        if (idx.x >= view.width || idx.y >= view.height) { return float4(0); }
        return C.decode(buf, region_index(view, idx.x, idx.y), N);""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float4(0); }
        return C.decode(buf, uint(idx_1d), N);"""
)

# Update RWCodecRegion::write
code = code.replace(
"""        if (idx.x < view.width && idx.y < view.height) {
            C.encode(buf, region_index(view, idx.x, idx.y), value, N);
        }""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d >= 0) {
            C.encode(buf, uint(idx_1d), value, N);
        }"""
)

# Update Region::read
code = code.replace(
"""        if (idx.x >= view.width || idx.y >= view.height) { return float4(0); }
        return buf[region_index(view, idx.x, idx.y)];""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float4(0); }
        return buf[uint(idx_1d)];"""
)

# Update RWRegion::read
code = code.replace(
"""        if (idx.x >= view.width || idx.y >= view.height) { return float4(0); }
        return buf[region_index(view, idx.x, idx.y)];""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float4(0); }
        return buf[uint(idx_1d)];"""
)

# Update RWRegion::write
code = code.replace(
"""        if (idx.x < view.width && idx.y < view.height) {
            buf[region_index(view, idx.x, idx.y)] = value;
        }""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d >= 0) {
            buf[uint(idx_1d)] = value;
        }"""
)

# Update ComplexRegion::read2d
code = code.replace(
"""        if (idx.x >= view.width || idx.y >= view.height) { return float2(0); }
        return buf[region_index(view, idx.x, idx.y)];""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float2(0); }
        return buf[uint(idx_1d)];"""
)

# Update RWComplexRegion::read2d
code = code.replace(
"""        if (idx.x >= view.width || idx.y >= view.height) { return float2(0); }
        return buf[region_index(view, idx.x, idx.y)];""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float2(0); }
        return buf[uint(idx_1d)];"""
)

# Update RWComplexRegion::write2d
code = code.replace(
"""        if (idx.x < view.width && idx.y < view.height) {
            buf[region_index(view, idx.x, idx.y)] = value;
        }""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d >= 0) {
            buf[uint(idx_1d)] = value;
        }"""
)

# Update MaskRegion::read
code = code.replace(
"""        if (idx.x >= view.width || idx.y >= view.height) { return float4(0); }
        float v = buf[region_index(view, idx.x, idx.y)];""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float4(0); }
        float v = buf[uint(idx_1d)];"""
)

# Update RWMaskRegion::read
code = code.replace(
"""        if (idx.x >= view.width || idx.y >= view.height) { return float4(0); }
        float v = buf[region_index(view, idx.x, idx.y)];""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d < 0) { return float4(0); }
        float v = buf[uint(idx_1d)];"""
)

# Update RWMaskRegion::write
code = code.replace(
"""        if (idx.x < view.width && idx.y < view.height) {
            buf[region_index(view, idx.x, idx.y)] = value.x;
        }""",
"""        int idx_1d = region_index(view, int(idx.x), int(idx.y));
        if (idx_1d >= 0) {
            buf[uint(idx_1d)] = value.x;
        }"""
)

with open("shaders/lib/region.slang", "w") as f:
    f.write(code)
