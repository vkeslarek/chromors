import lib.io;
import lib.region;

// Per-thread: for each pixel, accumulate value into histogram bin.
// Input: working-space float4 (linear light, from WorkingDecodeRegion sandwich).
// Output: HistogramOut with `bins` bins.
// Params: channel (0=R, 1=G, 2=B, 3=A, 4=luminance)
public void histogram_kernel<G: IRegion>(
    uint2 tid,
    G input,
    HistogramOut output,
    uint channel
) {
    float4 px = input.read(tid);
    float v;
    if (channel == 4u) {
        // luminance: BT.709 coefficients in linear light
        v = px.r * 0.2126 + px.g * 0.7152 + px.b * 0.0722;
    } else if (channel == 0u) { v = px.r; }
    else if (channel == 1u) { v = px.g; }
    else if (channel == 2u) { v = px.b; }
    else { v = px.a; }
    output.accumulate(v);
}
