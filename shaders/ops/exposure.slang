import lib.region;

// Exposure as linear gain with highlight-preserving rolloff.
// `gain = 2^stops` multiplies in linear light.
// `preserve = 0.0` hard-clips at 1.0; `preserve = 1.0` applies full rolloff.
public void exposure_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float gain, float preserve)
{
    float4 c = input.read(idx);
    float3 boosted = c.rgb * gain;
    if (preserve > 0.0)
    {
        float3 soft  = boosted / (1.0 + boosted);
        float3 hard  = min(boosted, 1.0);
        c.rgb = lerp(hard, soft, preserve);
    }
    else
    {
        c.rgb = boosted;
    }
    output.write(idx, c);
}
