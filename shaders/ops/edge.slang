import lib.region;

public void invert_kernel<R: IRegion>(uint2 idx, R input, RWRegion output)
{
    // Negate every band (matches libvips `invert`, which inverts alpha too).
    float4 c = input.read(idx);
    output.write(idx, float4(1.0, 1.0, 1.0, 1.0) - c);
}

public void sign_kernel<R: IRegion>(uint2 idx, R input, RWRegion output) {
    float4 c = input.read(idx);
    output.write(idx, sign(c));
}

public void abs_kernel<R: IRegion>(uint2 idx, R input, RWRegion output) {
    float4 c = input.read(idx);
    output.write(idx, abs(c));
}

public void hypot_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    output.write(idx, float4(sqrt(a.rgb * a.rgb + b.rgb * b.rgb), a.a));
}

