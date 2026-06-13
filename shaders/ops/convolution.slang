import lib.region;

// ── 2-D convolution with a mask image ─────────────────────────────────────────
//
// `mask` is a small 1-band image whose pixels are the convolution weights.
// `mask_w` / `mask_h` are the mask dimensions.
// Halo = (mask_w/2, mask_h/2) — the kernel reads a neighbourhood around each
// output pixel, multiplies by the mask weight, and accumulates.
//
// Out-of-bounds input reads clamp to the edge pixel, matching Vips' EXTEND_COPY
// pre-pass (`vips_embed(..., "extend", VIPS_EXTEND_COPY, ...)`) before convolving.

public void convolution_kernel<R1: IRegion, R2: IRegion>(
    uint2 idx, R1 input, R2 mask, RWRegion output,
    uint mask_w, uint mask_h)
{
    int hw = int(mask_w) >> 1;
    int hh = int(mask_h) >> 1;
    float4 sum = float4(0.0, 0.0, 0.0, 0.0);
    float weight_sum = 0.0;

    for (uint my = 0u; my < mask_h; my++)
    {
        for (uint mx = 0u; mx < mask_w; mx++)
        {
            float w = mask.read(uint2(mx, my)).r;
            int2 src_pos = int2(idx) + int2(int(mx) - hw, int(my) - hh);
            float4 src = input.read_clamped(src_pos);
            sum += src * w;
            weight_sum += w;
        }
    }

    // Normalise by the sum of weights (matching Vips behaviour).
    float4 result;
    if (weight_sum != 0.0) {
        result = sum / weight_sum;
    } else {
        result = sum;
    }
    output.write(idx, result);
}

// ── Morphology (erode / dilate) ───────────────────────────────────────────────
//
// vips_morph is a BITWISE hit-or-miss transform on uchar pixels, not a min/max
// filter: the mask must be boolean (0 / 128 / 255 per `vips__image_intize`).
// 128 = "don't care" (cell excluded); 255 = OR/AND the pixel byte as-is;
// 0 = OR/AND the bitwise-complement (255 - byte) of the pixel byte.
//   erode  (morph=0): result = AND over included cells, seeded 0xFF.
//   dilate (morph=1): result = OR  over included cells, seeded 0x00.
//
// Out-of-bounds input reads clamp to the edge pixel, matching Vips' EXTEND_COPY
// pre-pass (`vips_embed(..., "extend", VIPS_EXTEND_COPY, ...)`) before the morph.
//
// `src_max` recovers the pre-codec raw sample value (255 for u8, 65535 for
// u16, 1 for float formats) from our normalised [0,1] working float, so the
// clip-to-byte below matches `vips_cast(in, UCHAR, shift=false)` -- a CLAMP
// of the raw value to [0,255], not a rescale.

public void morph_kernel<R1: IRegion, R2: IRegion>(
    uint2 idx, R1 input, R2 mask, RWRegion output,
    uint morph, uint mask_w, uint mask_h, float src_max)
{
    int hw = int(mask_w) >> 1;
    int hh = int(mask_h) >> 1;
    bool erode = (morph == 0u);
    uint4 result = erode ? uint4(255u) : uint4(0u);

    for (uint my = 0u; my < mask_h; my++)
    {
        for (uint mx = 0u; mx < mask_w; mx++)
        {
            float m = mask.read(uint2(mx, my)).r;

            if (m != 128.0)
            {
                int2 src_pos = int2(idx) + int2(int(mx) - hw, int(my) - hh);
                float4 src = input.read_clamped(src_pos);
                float4 raw = clamp(src, 0.0, 1.0) * src_max;
                uint4 byte_val = uint4(clamp(round(raw), 0.0, 255.0));
                uint4 cell = (m == 0.0) ? (uint4(255u) - byte_val) : byte_val;

                result = erode ? (result & cell) : (result | cell);
            }
        }
    }

    output.write(idx, float4(result) / 255.0);
}

// ── Median (rank filter) ───────────────────────────────────────────────────────
//
// `sz x sz` neighbourhood, per-channel sorted; output is the middle element
// (matches vips_rank with index = sz*sz/2). Max window 11x11 (121 samples).

void insertion_sort_121(inout float arr[121], int n)
{
    for (int i = 1; i < n; i++)
    {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

public void median_kernel<R1: IRegion>(
    uint2 idx, R1 input, RWRegion output,
    uint sz)
{
    int half = int(sz) >> 1;
    int count = 0;
    float r[121];
    float g[121];
    float b[121];
    float a[121];

    for (uint my = 0u; my < sz; my++)
    {
        for (uint mx = 0u; mx < sz; mx++)
        {
            int2 src_pos = int2(idx) + int2(int(mx) - half, int(my) - half);
            float4 src = input.read(uint2(src_pos));
            r[count] = src.r;
            g[count] = src.g;
            b[count] = src.b;
            a[count] = src.a;
            count++;
        }
    }

    insertion_sort_121(r, count);
    insertion_sort_121(g, count);
    insertion_sort_121(b, count);
    insertion_sort_121(a, count);

    int mid = count / 2;
    output.write(idx, float4(r[mid], g[mid], b[mid], a[mid]));
}

// ── Compass ────────────────────────────────────────────────────────────────────
//
// Applies `mask` and a 90°-rotated copy of `mask`, returning the per-pixel
// max of the absolute responses (matches vips_compass's default
// times=2/angle=D90/combine=MAX — two 45° steps net out to a 90° rotation).
// For a square mask, rotating 90° clockwise maps cell (mx, my) to
// (my, mask_h - 1 - mx); this is exact for any odd square mask (vips itself
// requires square masks for compass).

uint2 compass_rotate90(uint mx, uint my, uint mask_h)
{
    return uint2(my, mask_h - 1u - mx);
}

public void compass_kernel<R1: IRegion, R2: IRegion>(
    uint2 idx, R1 input, R2 mask, RWRegion output,
    uint mask_w, uint mask_h)
{
    int hw = int(mask_w) >> 1;
    int hh = int(mask_h) >> 1;

    float4 sum_a = float4(0.0, 0.0, 0.0, 0.0);
    float4 sum_b = float4(0.0, 0.0, 0.0, 0.0);
    float wsum_a = 0.0;
    float wsum_b = 0.0;

    for (uint my = 0u; my < mask_h; my++)
    {
        for (uint mx = 0u; mx < mask_w; mx++)
        {
            int2 src_pos = int2(idx) + int2(int(mx) - hw, int(my) - hh);
            float4 src = input.read_clamped(src_pos);

            float wa = mask.read(uint2(mx, my)).r;
            sum_a += src * wa;
            wsum_a += wa;

            uint2 rpos = compass_rotate90(mx, my, mask_h);
            float wb = mask.read(rpos).r;
            sum_b += src * wb;
            wsum_b += wb;
        }
    }

    float4 res_a = (wsum_a != 0.0) ? sum_a / wsum_a : sum_a;
    float4 res_b = (wsum_b != 0.0) ? sum_b / wsum_b : sum_b;
    output.write(idx, max(abs(res_a), abs(res_b)));
}
