import lib.region;

// ── 2-D convolution with a mask image ─────────────────────────────────────────
//
// `mask` is a small 1-band image whose pixels are the convolution weights.
// `mask_w` / `mask_h` are the mask dimensions.
// Halo = (mask_w/2, mask_h/2) — the kernel reads a neighbourhood around each
// output pixel, multiplies by the mask weight, and accumulates.
//
// Out-of-bounds input reads return 0, matching Vips EXTEND_BLACK at edges.

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
            float4 src = input.read(uint2(src_pos));
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
// `morph`: 0 = erode (min), 1 = dilate (max).
// `mask` is the structuring element — only pixels where mask > 0 are considered.

public void morph_kernel<R1: IRegion, R2: IRegion>(
    uint2 idx, R1 input, R2 mask, RWRegion output,
    uint morph, uint mask_w, uint mask_h)
{
    int hw = int(mask_w) >> 1;
    int hh = int(mask_h) >> 1;
    float4 val;
    bool first = true;

    if (morph == 0u)
    {
        // Erode: min over neighbourhood
        val = float4(1.0, 1.0, 1.0, 1.0);
        for (uint my = 0u; my < mask_h; my++)
        {
            for (uint mx = 0u; mx < mask_w; mx++)
            {
                float w = mask.read(uint2(mx, my)).r;
                if (w > 0.0)
                {
                    int2 src_pos = int2(idx) + int2(int(mx) - hw, int(my) - hh);
                    float4 src = input.read(uint2(src_pos));
                    val = min(val, src);
                }
            }
        }
    }
    else
    {
        // Dilate: max over neighbourhood
        val = float4(0.0, 0.0, 0.0, 0.0);
        for (uint my = 0u; my < mask_h; my++)
        {
            for (uint mx = 0u; mx < mask_w; mx++)
            {
                float w = mask.read(uint2(mx, my)).r;
                if (w > 0.0)
                {
                    int2 src_pos = int2(idx) + int2(int(mx) - hw, int(my) - hh);
                    float4 src = input.read(uint2(src_pos));
                    val = max(val, src);
                }
            }
        }
    }

    output.write(idx, val);
}
