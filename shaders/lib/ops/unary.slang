import lib.region;

public void abs_kernel<R: IRegion>(uint2 idx, R input, RWRegion output) {
    float4 c = input.read(idx);
    output.write(idx, abs(c));
}

public void sign_kernel<R: IRegion>(uint2 idx, R input, RWRegion output) {
    float4 c = input.read(idx);
    output.write(idx, sign(c));
}

public void msb_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, int band) {
    float4 c = input.read(idx);
    // Vips `msb` extracts the most-significant byte.
    // For 8-bit formats, floor(c.rgb * 255.0) is the MSB.
    // We scale it back to [0,1] for the working space.
    // Note: this is an approximation valid only for 8-bit source data.
    float4 msb = floor(c * 255.0) / 255.0;
    float4 res;
    if (band >= 0) {
        float val = msb[band % 4];
        res = float4(val, val, val, msb.a);
    } else {
        res = msb;
    }
    output.write(idx, res);
}
