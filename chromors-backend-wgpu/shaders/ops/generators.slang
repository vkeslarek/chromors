import lib.region;
import lib.coord;

public void constant_kernel(
    uint2 idx,
    RWRegion output,
    int gen_ox, int gen_oy, int gen_lod, int gen_fw, int gen_fh,
    float color_r, float color_g, float color_b, float color_a
) {
    // Constant doesn't even need coordinates, it just writes the color.
    output.write(idx, float4(color_r, color_g, color_b, color_a));
}

import lib.rng;

public void linear_gradient_kernel(
    uint2 idx,
    RWRegion output,
    int gen_ox, int gen_oy, int gen_lod, int gen_fw, int gen_fh,
    float c0_r, float c0_g, float c0_b, float c0_a,
    float c1_r, float c1_g, float c1_b, float c1_a,
    float angle
) {
    GenCoord c = gen_coord(idx, gen_ox, gen_oy, gen_lod, gen_fw, gen_fh);
    
    float fw = max(float(gen_fw), 1.0);
    float fh = max(float(gen_fh), 1.0);
    float2 uv = c.full_px / float2(fw, fh);
    
    float dir_x = cos(angle);
    float dir_y = sin(angle);
    float t = clamp(uv.x * dir_x + uv.y * dir_y, 0.0, 1.0);
    
    float4 c0 = float4(c0_r, c0_g, c0_b, c0_a);
    float4 c1 = float4(c1_r, c1_g, c1_b, c1_a);
    float4 color_val = lerp(c0, c1, t);
    
    output.write(idx, color_val);
}

public void xyz_kernel(
    uint2 idx,
    RWRegion output,
    int gen_ox, int gen_oy, int gen_lod, int gen_fw, int gen_fh
) {
    GenCoord c = gen_coord(idx, gen_ox, gen_oy, gen_lod, gen_fw, gen_fh);
    // VIPS xyz produces (x, y) coordinates of the pixel in original space
    // VIPS puts it into two bands (channels), so we write (x, y, 0, 1)
    output.write(idx, float4(c.full_px.x, c.full_px.y, 0.0, 1.0));
}

public void gaussnoise_kernel(
    uint2 idx,
    RWRegion output,
    int gen_ox, int gen_oy, int gen_lod, int gen_fw, int gen_fh,
    float mean, float sigma, uint seed
) {
    GenCoord c = gen_coord(idx, gen_ox, gen_oy, gen_lod, gen_fw, gen_fh);
    float g0 = gauss(c.lod_coord, gen_lod, seed + 0u);
    float g1 = gauss(c.lod_coord, gen_lod, seed + 1u);
    float g2 = gauss(c.lod_coord, gen_lod, seed + 2u);
    output.write(idx, float4(mean + sigma * g0, mean + sigma * g1, mean + sigma * g2, 1.0));
}
