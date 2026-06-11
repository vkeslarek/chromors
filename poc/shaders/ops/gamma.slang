import lib.region;

public void gamma_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float exponent)
{
    float4 c = input.read(idx);
    c.rgb = pow(max(c.rgb, float3(0.0, 0.0, 0.0)), float3(exponent, exponent, exponent));
    output.write(idx, c);
}
