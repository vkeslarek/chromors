import lib.region;

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

public void embed_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, int ox, int oy, uint in_w, uint in_h, uint extend_mode, float bg_r, float bg_g, float bg_b) {
    int sx = int(idx.x) - ox;
    int sy = int(idx.y) - oy;

    if (sx >= 0 && sx < int(in_w) && sy >= 0 && sy < int(in_h)) {
        output.write(idx, input.read(uint2(sx, sy)));
    } else {
        if (extend_mode == 0) { // Black
            output.write(idx, float4(0.0, 0.0, 0.0, 1.0));
        } else if (extend_mode == 4) { // White
            output.write(idx, float4(1.0, 1.0, 1.0, 1.0));
        } else if (extend_mode == 5) { // Background
            output.write(idx, float4(bg_r, bg_g, bg_b, 1.0));
        } else if (extend_mode == 2) { // Repeat
            // handling negative mod correctly
            int wx = sx % int(in_w);
            int wy = sy % int(in_h);
            if (wx < 0) wx += int(in_w);
            if (wy < 0) wy += int(in_h);
            output.write(idx, input.read(uint2(wx, wy)));
        } else if (extend_mode == 3) { // Mirror
            // handling negative mod correctly for mirror
            int wx = sx;
            if (wx < 0) wx = -wx - 1;
            wx = (wx / int(in_w)) % 2 == 0 ? (wx % int(in_w)) : (int(in_w) - 1 - (wx % int(in_w)));

            int wy = sy;
            if (wy < 0) wy = -wy - 1;
            wy = (wy / int(in_h)) % 2 == 0 ? (wy % int(in_h)) : (int(in_h) - 1 - (wy % int(in_h)));
            output.write(idx, input.read(uint2(wx, wy)));
        } else { // Copy
            int cx = clamp(sx, 0, int(in_w) - 1);
            int cy = clamp(sy, 0, int(in_h) - 1);
            output.write(idx, input.read(uint2(cx, cy)));
        }
    }
}

public void rotate_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float inv_r00, float inv_r01, float inv_r10, float inv_r11, float cx, float cy, float ox, float oy, float bg_r, float bg_g, float bg_b, uint in_w, uint in_h) {
    float x = float(idx.x) - ox;
    float y = float(idx.y) - oy;

    float sx = inv_r00 * x + inv_r01 * y + cx;
    float sy = inv_r10 * x + inv_r11 * y + cy;

    float fx = floor(sx);
    float fy = floor(sy);
    int2 origin = int2(fx, fy);
    float wx = sx - fx;
    float wy = sy - fy;

    if (origin.x < -1 || origin.x >= int(in_w) || origin.y < -1 || origin.y >= int(in_h)) {
        output.write(idx, float4(bg_r, bg_g, bg_b, 1.0));
        return;
    }

    uint2 p00 = uint2(clamp(origin.x, 0, int(in_w)-1), clamp(origin.y, 0, int(in_h)-1));
    uint2 p10 = uint2(clamp(origin.x + 1, 0, int(in_w)-1), clamp(origin.y, 0, int(in_h)-1));
    uint2 p01 = uint2(clamp(origin.x, 0, int(in_w)-1), clamp(origin.y + 1, 0, int(in_h)-1));
    uint2 p11 = uint2(clamp(origin.x + 1, 0, int(in_w)-1), clamp(origin.y + 1, 0, int(in_h)-1));

    float4 c00 = input.read(p00);
    float4 c10 = input.read(p10);
    float4 c01 = input.read(p01);
    float4 c11 = input.read(p11);

    if (origin.x < 0) { c00 = float4(bg_r, bg_g, bg_b, 1.0); c01 = c00; }
    if (origin.x >= int(in_w)-1) { c10 = float4(bg_r, bg_g, bg_b, 1.0); c11 = c10; }
    if (origin.y < 0) { c00 = float4(bg_r, bg_g, bg_b, 1.0); c10 = c00; }
    if (origin.y >= int(in_h)-1) { c01 = float4(bg_r, bg_g, bg_b, 1.0); c11 = c01; }

    float4 c0 = lerp(c00, c10, wx);
    float4 c1 = lerp(c01, c11, wx);
    float4 res = lerp(c0, c1, wy);

    output.write(idx, res);
}

public void shrink_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint h_factor, uint v_factor)
{
    uint sx0 = idx.x * h_factor;
    uint sy0 = idx.y * v_factor;

    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    uint count = 0u;

    for (uint dy = 0u; dy < v_factor; dy++) {
        for (uint dx = 0u; dx < h_factor; dx++) {
            sum += input.read_clamped(int2(int(sx0 + dx), int(sy0 + dy)));
            count += 1u;
        }
    }

    output.write(idx, sum / float(count));
}

public void grid_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint in_w, uint tile_height, uint across) {
    uint col = idx.x / in_w;
    uint row = idx.y / tile_height;
    uint strip = row * across + col;

    uint sx = idx.x % in_w;
    uint sy = strip * tile_height + (idx.y % tile_height);

    output.write(idx, input.read(uint2(sx, sy)));
}

