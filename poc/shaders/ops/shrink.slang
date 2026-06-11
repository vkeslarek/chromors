import lib.region;

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
