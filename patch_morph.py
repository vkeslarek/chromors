code = """public void morph_kernel<R1: IRegion, R2: IRegion>(
    uint2 idx, R1 input, R2 mask, RWRegion output,
    uint morph, uint mask_w, uint mask_h, float src_max)
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
                if (w != 128.0)
                {
                    int2 src_pos = int2(idx) + int2(int(mx) - hw, int(my) - hh);
                    float4 src = input.read_clamped(src_pos);
                    if (w == 0.0) src = 1.0 - src; // Invert if 0
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
                    if (w == 0.0) src = 1.0 - src; // Invert if 0
                    val = max(val, src);
                }
            }
        }
    }

    output.write(idx, val);
}"""

import re
with open("shaders/ops/convolution.slang", "r") as f:
    text = f.read()

text = re.sub(r'public void morph_kernel[\s\S]*?output\.write\(idx, float4\(result\) / src_max\);\n}', code, text)

with open("shaders/ops/convolution.slang", "w") as f:
    f.write(text)
