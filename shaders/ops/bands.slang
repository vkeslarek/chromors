import lib.region;

float band_get(float4 v, uint index, uint bands) {
    switch (clamp(index, 0u, 3u)) {
        case 0u: return v.x;
        case 1u: return v.y;
        case 2u: return v.z;
        default: return v.w;
    }
}

float4 band_set(float4 v, uint index, uint bands, float value) {
    switch (clamp(index, 0u, 3u)) {
        case 0u: v.x = value; break;
        case 1u: v.y = value; break;
        case 2u: v.z = value; break;
        default: v.w = value; break;
    }
    return v;
}

// ── Bandjoin: combine 1–5 single-band sources into a multi-band output ────────

public void bandjoin1_kernel<R0: IRegion>(uint2 idx, R0 i0, RWRegion output, uint ch0) {
    output.write(idx, float4(i0.read(idx)[ch0], 0.0, 0.0, 1.0));
}

public void bandjoin2_kernel<R0: IRegion, R1: IRegion>(uint2 idx, R0 i0, R1 i1, RWRegion output, uint ch0, uint ch1) {
    output.write(idx, float4(i0.read(idx)[ch0], i1.read(idx)[ch1], 0.0, 1.0));
}

public void bandjoin3_kernel<R0: IRegion, R1: IRegion, R2: IRegion>(uint2 idx, R0 i0, R1 i1, R2 i2, RWRegion output, uint ch0, uint ch1, uint ch2) {
    output.write(idx, float4(i0.read(idx)[ch0], i1.read(idx)[ch1], i2.read(idx)[ch2], 1.0));
}

public void bandjoin4_kernel<R0: IRegion, R1: IRegion, R2: IRegion, R3: IRegion>(uint2 idx, R0 i0, R1 i1, R2 i2, R3 i3, RWRegion output, uint ch0, uint ch1, uint ch2, uint ch3) {
    output.write(idx, float4(i0.read(idx)[ch0], i1.read(idx)[ch1], i2.read(idx)[ch2], i3.read(idx)[ch3]));
}

public void bandjoin5_kernel<R0: IRegion, R1: IRegion, R2: IRegion, R3: IRegion, R4: IRegion>(uint2 idx, R0 i0, R1 i1, R2 i2, R3 i3, R4 i4, RWRegion output, uint ch0, uint ch1, uint ch2, uint ch3, uint ch4) {
    // 5-band: pack first 4 into rgba, discard 5th (max 4 in float4)
    output.write(idx, float4(i0.read(idx)[ch0], i1.read(idx)[ch1], i2.read(idx)[ch2], i3.read(idx)[ch3]));
}

public void bandbool_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint boolean_op, uint bands)
{
    float4 a = input.read(idx);
    int4 va = int4(a);
    int r = va.x;
    if (bands >= 2u) {
        switch (boolean_op) {
            case 0u: r = r & va.y; break;
            case 1u: r = r | va.y; break;
            case 2u: r = r ^ va.y; break;
            case 3u: r = r << va.y; break;
            case 4u: r = r >> va.y; break;
        }
    }
    if (bands >= 3u) {
        switch (boolean_op) {
            case 0u: r = r & va.z; break;
            case 1u: r = r | va.z; break;
            case 2u: r = r ^ va.z; break;
            case 3u: r = r << va.z; break;
            case 4u: r = r >> va.z; break;
        }
    }
    if (bands >= 4u) {
        switch (boolean_op) {
            case 0u: r = r & va.w; break;
            case 1u: r = r | va.w; break;
            case 2u: r = r ^ va.w; break;
            case 3u: r = r << va.w; break;
            case 4u: r = r >> va.w; break;
        }
    }
    float fr = float(r);
    output.write(idx, float4(fr, fr, fr, 1.0));
}

public void bandfold_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint factor, uint in_bands, uint out_bands)
{
    uint f = max(factor, 1u);
    float4 result = float4(0.0, 0.0, 0.0, 1.0);

    for (uint k = 0u; k < f; k++)
    {
        float4 c = input.read(uint2(idx.x * f + k, idx.y));
        for (uint ci = 0u; ci < in_bands; ci++)
        {
            uint out_band = k * in_bands + ci;
            if (out_band < out_bands)
                result = band_set(result, out_band, out_bands, band_get(c, ci, in_bands));
        }
    }

    output.write(idx, result);
}

public void bandunfold_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint factor, uint in_bands, uint out_bands)
{
    uint f = max(factor, 1u);
    uint x_in = idx.x / f;
    uint k = idx.x % f;
    float4 c = input.read(uint2(x_in, idx.y));
    float4 result = float4(0.0, 0.0, 0.0, 1.0);

    for (uint b = 0u; b < out_bands; b++)
    {
        uint in_band = k * out_bands + b;
        if (in_band < in_bands)
            result = band_set(result, b, out_bands, band_get(c, in_band, in_bands));
    }

    output.write(idx, result);
}

public void bandmean_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint bands)
{
    float4 c = input.read(idx);
    float sum = c.r;
    if (bands >= 2u) sum += c.g;
    if (bands >= 3u) sum += c.b;
    if (bands >= 4u) sum += c.a;
    float mean = sum / max(float(bands), 1.0);
    output.write(idx, float4(mean, mean, mean, 1.0));
}

public void extract_band_range_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint band, uint count)
{
    float4 c = input.read(idx);
    float4 out = float4(0.0, 0.0, 0.0, 1.0);

    if (count >= 1u) out[0u] = c[min(band + 0u, 3u)];
    if (count >= 2u) out[1u] = c[min(band + 1u, 3u)];
    if (count >= 3u) out[2u] = c[min(band + 2u, 3u)];
    if (count >= 4u) out[3u] = c[min(band + 3u, 3u)];

    output.write(idx, out);
}

