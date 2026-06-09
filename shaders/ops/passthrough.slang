import lib.region;

public void passthrough_kernel<R: IRegion>(uint2 idx, R input, RWRegion output)
{
    output.write(idx, input.read(idx));
}
