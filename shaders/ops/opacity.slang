import lib.region;

public void opacity_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float amount)
{
    float4 c = input.read(idx);
    c.a *= amount;
    output.write(idx, c);
}
