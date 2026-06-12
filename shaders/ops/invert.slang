import lib.region;

public void invert_kernel<R: IRegion>(uint2 idx, R input, RWRegion output)
{
    // Negate every band (matches libvips `invert`, which inverts alpha too).
    float4 c = input.read(idx);
    output.write(idx, float4(1.0, 1.0, 1.0, 1.0) - c);
}
