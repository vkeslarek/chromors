import lib.region;

// Porter-Duff compositing — premultiplied formulation matching engine.
// `mode` matches vips VipsBlendMode: 0 Clear ... 13 Saturate.
// `bg` = base (dst), `fg` = overlay (src).

public void compose_kernel<R1: IRegion, R2: IRegion>(
    uint2 idx, R1 bg, R2 fg, RWRegion output, uint mode, int ox, int oy)
{
    float4 base    = bg.read(idx);
    float4 overlay = fg.read(uint2(int2(idx) - int2(ox, oy)));

    float aA = overlay.a;
    float aB = base.a;

    float3 A = overlay.rgb * aA;
    float3 B = base.rgb    * aB;

    float aR;
    float3 R;

    switch (mode) {
        case 0u:  aR = 0.0;                          R = float3(0.0); break;
        case 1u:  aR = aA;                            R = A; break;
        case 2u:  aR = aA + aB * (1.0 - aA);         R = A + B * (1.0 - aA); break;
        case 3u:  aR = aA * aB;                       R = A * aB; break;
        case 4u:  aR = aA * (1.0 - aB);               R = A * (1.0 - aB); break;
        case 5u:  aR = aB;                            R = A + B * (1.0 - aA); break;
        case 6u:  aR = aB;                            R = B; break;
        case 7u:  aR = aB + aA * (1.0 - aB);         R = B + A * (1.0 - aB); break;
        case 8u:  aR = aA * aB;                       R = B * aA; break;
        case 9u:  aR = aB * (1.0 - aA);               R = B * (1.0 - aA); break;
        case 10u: aR = aA;                            R = B + A * (1.0 - aB); break;
        case 11u: aR = aA + aB - 2.0 * aA * aB;      R = A * (1.0 - aB) + B * (1.0 - aA); break;
        case 12u: aR = min(1.0, aA + aB);            R = A + B; break;
        case 13u:
        {
            aR = min(1.0, aA + aB);
            float st = min(aA, 1.0 - aB);
            R = st * A + B;
            break;
        }
        default:  aR = 0.0; R = float3(0.0); break;
    }

    float3 result_rgb = (aR > 0.0) ? (R / aR) : float3(0.0);
    output.write(idx, float4(result_rgb, aR));
}

// ── Spatial join (vips_join) ──────────────────────────────────────────────────
//
// Stitches two images side-by-side (horizontal) or one-on-top-of-the-other
// (vertical).  `shim` is a pixel gap between them.  Pixels outside both
// images are filled with `(bg_r, bg_g, bg_b, 0.0)`.
//
// `src0_w` / `src0_h` — dimensions of the first image in this output.
// For horizontal join: src1_x = src0_w + shim, src1_y = 0.
// For vertical join:   src1_x = 0, src1_y = src0_h + shim.

public void join_kernel<R1: IRegion, R2: IRegion>(
    uint2 idx, R1 src0, R2 src1, RWRegion output,
    uint direction, int shim,
    uint src0_w, uint src0_h, uint src1_w, uint src1_h,
    float bg_r, float bg_g, float bg_b)
{
    float4 col;
    bool hit = false;

    if (direction == 0u)
    {
        // Horizontal: src0 at (0,0), src1 to the right with gap `shim`.
        if (idx.x < src0_w && idx.y < src0_h)
        {
            col = src0.read(idx);
            hit = true;
        }
        else if (idx.x >= int(src0_w) + shim && idx.x < int(src0_w) + shim + int(src1_w)
              && idx.y < src1_h)
        {
            int local_x = idx.x - int(src0_w) - shim;
            col = src1.read(uint2(local_x, idx.y));
            hit = true;
        }
    }
    else
    {
        // Vertical: src0 at (0,0), src1 below with gap `shim`.
        if (idx.y < src0_h && idx.x < src0_w)
        {
            col = src0.read(idx);
            hit = true;
        }
        else if (idx.y >= int(src0_h) + shim && idx.y < int(src0_h) + shim + int(src1_h)
              && idx.x < src1_w)
        {
            int local_y = idx.y - int(src0_h) - shim;
            col = src1.read(uint2(idx.x, local_y));
            hit = true;
        }
    }

    if (!hit) {
        col = float4(bg_r, bg_g, bg_b, 0.0);
    }
    output.write(idx, col);
}

// ── Spatial insert (vips_insert) ──────────────────────────────────────────────
//
// Pastes `sub` into `main` at position (x, y).  Pixels within `sub`'s bounds
// (shifted by x,y) are read from `sub`; everywhere else reads from `main`.
// Background pixels fall through to `(bg_r, bg_g, bg_b, 0.0)`.

public void insert_kernel<R1: IRegion, R2: IRegion>(
    uint2 idx, R1 main, R2 sub, RWRegion output,
    int x, int y,
    uint sub_w, uint sub_h,
    float bg_r, float bg_g, float bg_b)
{
    float4 col;
    bool hit = false;

    if (idx.x >= x && idx.x < x + int(sub_w)
     && idx.y >= y && idx.y < y + int(sub_h))
    {
        col = sub.read(uint2(idx.x - x, idx.y - y));
        hit = true;
    }
    else
    {
        col = main.read(idx);
        if (col.a > 0.0 || col.r > 0.0 || col.g > 0.0 || col.b > 0.0)
            hit = true;
    }

    if (!hit) {
        col = float4(bg_r, bg_g, bg_b, 0.0);
    }
    output.write(idx, col);
}
