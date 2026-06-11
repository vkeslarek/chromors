import lib.region;

public void brightness_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float factor)
{
    float4 c = input.read(idx);
    c.rgb *= factor;
    output.write(idx, c);
}
