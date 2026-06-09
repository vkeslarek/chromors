import lib.region;

public void invert_kernel<R: IRegion>(uint2 idx, R input, RWRegion output)
{
    float4 c = input.read(idx);
    c.rgb = float3(1.0, 1.0, 1.0) - c.rgb;
    output.write(idx, c);
}
