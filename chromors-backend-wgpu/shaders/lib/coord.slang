public struct GenCoord {
    public int2  lod_coord;   // absolute LOD-space integer coord
    public float2 full_px;    // full-res pixel center
    public float2 uv;         // normalized [0,1)
    public int   lod;
}

public GenCoord gen_coord(uint2 idx, int ox, int oy, int lod, int fw, int fh) {
    GenCoord c;
    c.lod_coord = int2(ox, oy) + int2(idx);
    int scale   = 1 << lod;
    c.full_px   = (float2(c.lod_coord) + 0.5) * float(scale);
    c.uv        = c.full_px / float2(fw, fh);
    c.lod       = lod;
    return c;
}
