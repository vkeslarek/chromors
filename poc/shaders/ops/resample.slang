import lib.region;

// Bilinear interpolation resampling kernel
public void resize_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float inv_hscale, float inv_vscale) {
    // Standard pixel-center convention
    float sx = (float(idx.x) + 0.5) * inv_hscale - 0.5;
    float sy = (float(idx.y) + 0.5) * inv_vscale - 0.5;

    // Floor to get the top-left texel index
    float fx = floor(sx);
    float fy = floor(sy);
    int2 origin = int2(fx, fy);

    // Fractional parts for blending
    float wx = sx - fx;
    float wy = sy - fy;

    // Read the 4 neighboring pixels (clamped by the region bounds implicitly if we use read_clamped or just safe bounds)
    // Actually, `input.read(uint2)` takes uint2. We should max(0, origin) or rely on `input.read_clamped(origin)` if it exists.
    // Wait, the region interface only provides `read(uint2)`. Negative coords will wrap/underflow unless handled.
    // VIPS handles this via the demand() halo, but we need to ensure idx >= 0.
    uint2 p00 = uint2(max(0, origin.x), max(0, origin.y));
    uint2 p10 = uint2(max(0, origin.x + 1), max(0, origin.y));
    uint2 p01 = uint2(max(0, origin.x), max(0, origin.y + 1));
    uint2 p11 = uint2(max(0, origin.x + 1), max(0, origin.y + 1));

    float4 c00 = input.read(p00);
    float4 c10 = input.read(p10);
    float4 c01 = input.read(p01);
    float4 c11 = input.read(p11);

    // Bilinear interpolation
    float4 c0 = lerp(c00, c10, wx);
    float4 c1 = lerp(c01, c11, wx);
    float4 res = lerp(c0, c1, wy);

    output.write(idx, res);
}
