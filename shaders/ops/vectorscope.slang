import lib.io;
import lib.region;

public void vectorscope_kernel<G: IRegion>(
    uint2        tid,
    G            input,
    HistogramOut output,   // bin_count == grid_size * grid_size
    uint         grid_size // typically 128
) {
    float4 px = input.read(tid);

    // The input is already converted to sRGB (via ConvertOp in tick.rs),
    // but the GPU pipeline might be evaluating in HDR (float16) producing values > 1.0.
    // The CPU VIPS path computes vectorscope on 8-bit clamped bytes, so we must
    // clamp here to ensure the preview vectorscope shape exactly matches the applied shape.
    float r = clamp(px.r, 0.0, 1.0);
    float g = clamp(px.g, 0.0, 1.0);
    float b = clamp(px.b, 0.0, 1.0);

    float cb = -0.168736 * r - 0.331264 * g + 0.500000 * b;
    float cr =  0.500000 * r - 0.418688 * g - 0.081312 * b;

    // CPU scales by (grid_size - 1). Example: (128 - 1) = 127.0
    float n = float(grid_size - 1);
    uint gx = uint(clamp((cb + 0.5) * n, 0.0, n));
    uint gy = uint(clamp((cr + 0.5) * n, 0.0, n));
    
    uint idx = gy * grid_size + gx;
    if (idx < output.bin_count) {
        InterlockedAdd(output.bins[idx], 1u);
    }
}
