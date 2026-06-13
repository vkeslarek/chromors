import lib.region;

public void sharpen_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float sigma, float m1)
{
    float4 orig = input.read_clamped(int2(idx));
    int r = int(ceil(sigma * 3.0));
    if (r == 0) {
        output.write(idx, orig);
        return;
    }
    float two_sigma2 = 2.0 * sigma * sigma;
    float4 acc = float4(0);
    float wsum = 0.0;
    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            float w = exp(-float(x * x + y * y) / two_sigma2);
            acc += w * input.read_clamped(int2(idx) + int2(x, y));
            wsum += w;
        }
    }
    float4 blurred = acc / wsum;
    float4 result = orig + m1 * (orig - blurred);
    output.write(idx, clamp(result, float4(0.0), float4(1.0)));
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

public void blur_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float sigma, int radius)
{
    if (radius == 0) {
        output.write(idx, input.read_clamped(int2(idx)));
        return;
    }
    float two_sigma2 = 2.0 * sigma * sigma;
    float4 acc = float4(0);
    float wsum = 0.0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float w = exp(-float(x * x + y * y) / two_sigma2);
            acc += w * input.read_clamped(int2(idx) + int2(x, y));
            wsum += w;
        }
    }
    output.write(idx, acc / wsum);
}

