import lib.region;

public void blur_h_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float sigma)
{
    int r = int(ceil(sigma * 3.0));
    if (r == 0) {
        output.write(idx, input.read_clamped(int2(idx)));
        return;
    }
    float two_sigma2 = 2.0 * sigma * sigma;
    float4 acc = float4(0);
    float wsum = 0.0;
    for (int x = -r; x <= r; x++) {
        float w = exp(-float(x * x) / two_sigma2);
        acc += w * input.read_clamped(int2(idx) + int2(x, 0));
        wsum += w;
    }
    output.write(idx, acc / wsum);
}

public void blur_v_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float sigma)
{
    int r = int(ceil(sigma * 3.0));
    if (r == 0) {
        output.write(idx, input.read_clamped(int2(idx)));
        return;
    }
    float two_sigma2 = 2.0 * sigma * sigma;
    float4 acc = float4(0);
    float wsum = 0.0;
    for (int j = -r; j <= r; j++) {
        float w = exp(-float(j * j) / two_sigma2);
        acc += w * input.read_clamped(int2(idx) + int2(0, j));
        wsum += w;
    }
    output.write(idx, acc / wsum);
}
