import lib.region;

float _luma(float3 c)
{
    return 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
}

public void saturation_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float amount)
{
    float4 c = input.read(idx);
    float l = _luma(c.rgb);
    c.rgb = lerp(float3(l, l, l), c.rgb, amount);
    output.write(idx, c);
}
