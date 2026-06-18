import lib.region;
import lib.io;

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

// Input IRegion wraps the full source image (tight layout, local coord = source coord).
// out_x, out_y: global output-tile origin (idx is tile-local).
// ox, oy: where the source is placed in the output canvas.
public void embed_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, int out_x, int out_y, int ox, int oy, uint in_w, uint in_h, uint extend_mode, float bg_r, float bg_g, float bg_b) {
    int gx = out_x + int(idx.x);
    int gy = out_y + int(idx.y);
    int sx = gx - ox; // source x
    int sy = gy - oy; // source y

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
            int wx = sx % int(in_w);
            int wy = sy % int(in_h);
            if (wx < 0) wx += int(in_w);
            if (wy < 0) wy += int(in_h);
            output.write(idx, input.read(uint2(wx, wy)));
        } else if (extend_mode == 3) { // Mirror
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

// out_x, out_y: global output-tile origin so tile-local idx maps to correct source.
public void shrink_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, int out_x, int out_y, uint h_factor, uint v_factor)
{
    uint sx0 = (out_x + int(idx.x)) * h_factor;
    uint sy0 = (out_y + int(idx.y)) * v_factor;

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

public void grid_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, int out_x, int out_y, uint in_w, uint tile_height, uint across) {
    uint x = out_x + idx.x;
    uint y = out_y + idx.y;
    uint col = x / in_w;
    uint row = y / tile_height;
    uint strip = row * across + col;

    uint sx = x % in_w;
    uint sy = strip * tile_height + (y % tile_height);

    output.write(idx, input.read(uint2(sx, sy)));
}


// Local-gradient "interestingness" score for a pixel: sum of the forward
// luma differences to its right and below neighbors, scaled to an integer.
// Used by `smartcrop_col_score_kernel` / `smartcrop_row_score_kernel` as a
// cheap proxy for "detail" — not vips' own ENTROPY/ATTENTION heuristic, but
// a reasonable approximation for picking a high-detail crop window.
uint smartcrop_score<G: IRegion>(uint2 idx, G input) {
    float4 c = input.read(idx);
    float4 cx1 = input.read_clamped(int2(int(idx.x) + 1, int(idx.y)));
    float4 cy1 = input.read_clamped(int2(int(idx.x), int(idx.y) + 1));
    float luma = c.r * 0.2126 + c.g * 0.7152 + c.b * 0.0722;
    float lx = cx1.r * 0.2126 + cx1.g * 0.7152 + cx1.b * 0.0722;
    float ly = cy1.r * 0.2126 + cy1.g * 0.7152 + cy1.b * 0.0722;
    return uint((abs(lx - luma) + abs(ly - luma)) * 1024.0);
}

// Sum of `smartcrop_score` over column `idx.x`, into a `width`-bin histogram.
// dispatch domain = (width, height).
public void smartcrop_col_score_kernel<G: IRegion>(uint2 idx, G input, HistogramOut output) {
    output.accumulate_at(idx.x, smartcrop_score(idx, input));
}

// Sum of `smartcrop_score` over row `idx.y`, into a `height`-bin histogram.
// dispatch domain = (width, height).
public void smartcrop_row_score_kernel<G: IRegion>(uint2 idx, G input, HistogramOut output) {
    output.accumulate_at(idx.y, smartcrop_score(idx, input));
}

// Crop a `width x height` window starting at `(off_x, off_y)` out of `input`.
public void smartcrop_kernel<G: IRegion>(uint2 idx, G input, RWRegion output, uint off_x, uint off_y) {
    output.write(idx, input.read(uint2(idx.x + off_x, idx.y + off_y)));
}
