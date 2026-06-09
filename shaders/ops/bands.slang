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

public void bandjoin1_kernel<R: IRegion>(uint2 idx, R src0, RWRegion output, uint ch0)
{
    float4 c0 = src0.read(idx);
    output.write(idx, float4(c0[ch0], 0.0, 0.0, 1.0));
}

public void bandjoin2_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 src0, R2 src1, RWRegion output, uint ch0, uint ch1)
{
    float4 c0 = src0.read(idx);
    float4 c1 = src1.read(idx);
    output.write(idx, float4(c0[ch0], c1[ch1], 0.0, 1.0));
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

public void bandfold_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint factor)
{
    uint f = max(factor, 1u);
    uint src_y = idx.y / f;
    uint band = idx.y % f;
    float4 c = input.read(uint2(idx.x, src_y));
    float val;
    if (band == 0u) val = c.r;
    else if (band == 1u) val = c.g;
    else if (band == 2u) val = c.b;
    else val = c.a;
    output.write(idx, float4(val, val, val, 1.0));
}

public void bandunfold_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint factor)
{
    float4 out_val = float4(0.0, 0.0, 0.0, 1.0);
    uint f = max(factor, 1u);
    if (f >= 1u) out_val.r = input.read(uint2(idx.x, idx.y * f + 0u)).r;
    if (f >= 2u) out_val.g = input.read(uint2(idx.x, idx.y * f + 1u)).r;
    if (f >= 3u) out_val.b = input.read(uint2(idx.x, idx.y * f + 2u)).r;
    if (f >= 4u) out_val.a = input.read(uint2(idx.x, idx.y * f + 3u)).r;
    output.write(idx, out_val);
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
