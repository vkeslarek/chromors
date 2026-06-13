import lib.region;

/// Extracts channel `band_idx` and replicates it to all four output channels.
/// Output: float4(val, val, val, 1.0) where val = input.channel[band_idx].
public void extract_band_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint band_idx)
{
    float4 c = input.read(idx);
    float val;
    if      (band_idx == 0u) val = c.r;
    else if (band_idx == 1u) val = c.g;
    else if (band_idx == 2u) val = c.b;
    else                     val = c.a;
    output.write(idx, float4(val, val, val, 1.0));
}

/// Extracts `count` bands starting at `band`, packing them into output
/// channels 0 .. count-1.  Remaining channels are zeroed; alpha = 1.
/// Vips equivalent: extract_band(band, n=count).
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

/// Multiplies a specific channel by `factor`, leaves all other channels unchanged.
public void scale_band_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint band_idx, float factor)
{
    float4 c = input.read(idx);
    if      (band_idx == 0u) c.r *= factor;
    else if (band_idx == 1u) c.g *= factor;
    else if (band_idx == 2u) c.b *= factor;
    else                     c.a *= factor;
    output.write(idx, c);
}

/// Adds a constant offset to a specific channel, clamping to [0, 1].
public void add_to_band_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint band_idx, float offset)
{
    float4 c = input.read(idx);
    if      (band_idx == 0u) c.r = saturate(c.r + offset);
    else if (band_idx == 1u) c.g = saturate(c.g + offset);
    else if (band_idx == 2u) c.b = saturate(c.b + offset);
    else                     c.a = saturate(c.a + offset);
    output.write(idx, c);
}

// ── Bandjoin kernels (N-source → single float4 output) ────────────────────────
//
// Each source contributes one band to the output.  For N sources the output
// layout is  r=src[0], g=src[1], b=src[2], a=src[3] (…).
//
// For 5-input joins (CMYKA) only the first 4 bands fit in a float4 pixel;
// the fifth band is stored in the alpha channel (overwriting band 3) and the
// caller should use the CPU Vips path for lossless 5-band images.

// 1- and 2-band outputs use the Gray/GrayA codecs, which encode from a
// broadcast (r=g=b=gray, a=alpha) working value -- not from distinct r/g
// components -- so the joined band(s) must be broadcast across rgb.

public void bandjoin1_kernel<R: IRegion>(uint2 idx, R src0, RWRegion output, uint ch0)
{
    float4 c0 = src0.read(idx);
    float v = c0[ch0];
    output.write(idx, float4(v, v, v, 1.0));
}

public void bandjoin2_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 src0, R2 src1, RWRegion output, uint ch0, uint ch1)
{
    float4 c0 = src0.read(idx);
    float4 c1 = src1.read(idx);
    float v = c0[ch0];
    output.write(idx, float4(v, v, v, c1[ch1]));
}

public void bandjoin3_kernel<R1: IRegion, R2: IRegion, R3: IRegion>(uint2 idx, R1 src0, R2 src1, R3 src2, RWRegion output, uint ch0, uint ch1, uint ch2)
{
    float4 c0 = src0.read(idx);
    float4 c1 = src1.read(idx);
    float4 c2 = src2.read(idx);
    output.write(idx, float4(c0[ch0], c1[ch1], c2[ch2], 1.0));
}

public void bandjoin4_kernel<R1: IRegion, R2: IRegion, R3: IRegion, R4: IRegion>(uint2 idx, R1 src0, R2 src1, R3 src2, R4 src3, RWRegion output, uint ch0, uint ch1, uint ch2, uint ch3)
{
    float4 c0 = src0.read(idx);
    float4 c1 = src1.read(idx);
    float4 c2 = src2.read(idx);
    float4 c3 = src3.read(idx);
    output.write(idx, float4(c0[ch0], c1[ch1], c2[ch2], c3[ch3]));
}

public void bandjoin5_kernel<R1: IRegion, R2: IRegion, R3: IRegion, R4: IRegion, R5: IRegion>(uint2 idx, R1 src0, R2 src1, R3 src2, R4 src3, R5 src4, RWRegion output, uint ch0, uint ch1, uint ch2, uint ch3, uint ch4)
{
    float4 c0 = src0.read(idx);
    float4 c1 = src1.read(idx);
    float4 c2 = src2.read(idx);
    float4 c3 = src3.read(idx);
    float4 c4 = src4.read(idx);
    // CMYKA: C=c0[ch0] M=c1[ch1] Y=c2[ch2] K=c3[ch3] (dropped) A=c4[ch4]
    // The 5th band (alpha) overwrites the 4th band (K) to fit in float4.
    output.write(idx, float4(c0[ch0], c1[ch1], c2[ch2], c4[ch4]));
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

// Bandfold/bandunfold pack/unpack the X axis into band planes (vips
// `bandfold`/`bandunfold`: out pixel = memcpy of `factor` consecutive input
// pixels' worth of bytes). A working float4 holds at most 4 bands, and the
// Gray/GrayA codecs decode/encode a 1- or 2-band image as a *broadcast*
// (r=g=b=v, a=alpha) rather than packing distinct values into r/g -- so band
// `c` of an `n`-band pixel must be read/written through these helpers, not
// `c[band]` directly.

float band_get(float4 c, uint band, uint n_bands)
{
    if (n_bands <= 1u) return c.r;
    if (n_bands == 2u) return (band == 0u) ? c.r : c.a;
    if (band == 0u) return c.r;
    if (band == 1u) return c.g;
    if (band == 2u) return c.b;
    return c.a;
}

float4 band_set(float4 c, uint band, uint n_bands, float val)
{
    if (n_bands <= 1u) { c.r = val; c.g = val; c.b = val; return c; }
    if (n_bands == 2u) {
        if (band == 0u) { c.r = val; c.g = val; c.b = val; }
        else { c.a = val; }
        return c;
    }
    if (band == 0u) c.r = val;
    else if (band == 1u) c.g = val;
    else if (band == 2u) c.b = val;
    else c.a = val;
    return c;
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
