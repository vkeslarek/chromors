import lib.region;
import lib.color.params;

uint2 compass_rotate90(uint mx, uint my, uint mh) {
    return uint2(my, mh - 1u - mx);
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

public void morph_kernel<R1: IRegion, R2: IRegion>(
    uint2 idx, R1 input, R2 mask, RWRegion output,
    uint morph, uint mask_w, uint mask_h, float src_max)
{
    int hw = int(mask_w) >> 1;
    int hh = int(mask_h) >> 1;

    float4 val;
    if (morph == 0u)
    {
        // Erode: min over neighbourhood
        val = float4(1.0, 1.0, 1.0, 1.0);
        for (uint my = 0u; my < mask_h; my++)
        {
            for (uint mx = 0u; mx < mask_w; mx++)
            {
                float w = mask.read(uint2(mx, my)).r;
                if (w != 128.0)
                {
                    int2 src_pos = int2(idx) + int2(int(mx) - hw, int(my) - hh);
                    float4 src = input.read_clamped(src_pos);
                    if (w == 0.0) src = 1.0 - src; // invert for negative mask logic
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
                if (w != 128.0)
                {
                    int2 src_pos = int2(idx) + int2(int(mx) - hw, int(my) - hh);
                    float4 src = input.read_clamped(src_pos);
                    if (w == 0.0) src = 1.0 - src; // invert for negative mask logic
                    val = max(val, src);
                }
            }
        }
    }
    output.write(idx, val);
}

