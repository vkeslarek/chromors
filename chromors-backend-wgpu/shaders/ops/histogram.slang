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

// Per-thread: for each pixel, accumulate the first `bands` channels (<= 4)
// each into their own `bins`-wide histogram, packed into one `bins * bands`
// buffer (`out[b*bins+i]`, band-major).
public void histogram_multi_kernel<G: IRegion>(
    uint2 tid,
    G input,
    HistogramOut output,
    uint bins,
    uint bands
) {
    float4 px = input.read(tid);
    for (uint b = 0u; b < bands; b++) {
        float v = px[b];
        uint bin = uint(clamp(v, 0.0, 0.9999) * float(bins));
        InterlockedAdd(output.bins[b * bins + bin], 1u);
    }
}

// Bridges a staged `bins * bands` histogram (`HistogramKind`) into a
// `bins x 1 x bands` `ImageKind` (raw counts as F32, one bin per pixel,
// `res[b] = hist[b*bins+idx.x]`). `unused` is the original image input,
// carried only because `HistogramFind`'s DAG edge requires it — the kernel
// reads the staged histogram instead. dispatch domain = (bins, 1).
public void histogram_to_image_kernel<G: IRegion>(
    uint2 idx,
    G unused,
    HistogramIn hist,
    RWRegion output,
    uint bins,
    uint bands
) {
    float4 res = float4(0.0, 0.0, 0.0, 1.0);
    for (uint b = 0u; b < bands; b++) {
        res[b] = float(hist.read(b * bins + idx.x));
    }
    output.write(idx, res);
}

// Cumulative sum over a `bins x 1 x bands` histogram-image (raw F32 counts):
// `out[x][b] = sum(in[0..=x][b])`. dispatch domain = (bins, 1).
//
// `input` reads the *whole* histogram-image, every thread scanning all
// `bins` entries — fusing this with its producer (in the same dispatch,
// without a barrier) would race against the other threads' writes. So
// `hist` is a separately-materialized copy of `self.input`, injected via
// `GpuBuilder::extra_input`; `unused` is the fused-producer placeholder
// `self.input`'s mandatory DAG edge resolves to.
public void hist_image_cumulative_kernel<G: IRegion, H: IRegion>(
    uint2 idx,
    G unused,
    H hist,
    RWRegion output,
    uint bins,
    uint bands
) {
    float4 res = float4(0.0, 0.0, 0.0, 1.0);
    for (uint b = 0u; b < bands; b++) {
        float cum = 0.0;
        for (uint i = 0u; i <= idx.x; i++) {
            cum += hist.read(uint2(i, 0u))[b];
        }
        res[b] = cum;
    }
    output.write(idx, res);
}

// Normalize a `bins x 1 x bands` histogram-image (raw F32 counts) so each
// band's maximum bin value maps to `bins - 1`. dispatch domain = (bins, 1).
// See `hist_image_cumulative_kernel` for why `hist` is separately staged.
public void hist_image_normalize_kernel<G: IRegion, H: IRegion>(
    uint2 idx,
    G unused,
    H hist,
    RWRegion output,
    uint bins,
    uint bands
) {
    float4 res = float4(0.0, 0.0, 0.0, 1.0);
    for (uint b = 0u; b < bands; b++) {
        float max_val = 0.0;
        for (uint i = 0u; i < bins; i++) {
            float v = hist.read(uint2(i, 0u))[b];
            if (v > max_val) { max_val = v; }
        }
        float v = hist.read(idx)[b];
        res[b] = max_val > 0.0 ? round(v * float(bins - 1u) / max_val) : v;
    }
    output.write(idx, res);
}

// Renders a `width x 1 x bands` histogram-image (raw F32 counts) as a
// `width x width` bar chart, one column per bin, bottom-aligned. dispatch
// domain = (width, width). See `hist_image_cumulative_kernel` for why
// `hist` is separately staged.
public void hist_plot_kernel<G: IRegion, H: IRegion>(
    uint2 idx,
    G unused,
    H hist,
    RWRegion output,
    uint width,
    uint bands
) {
    float4 res = float4(0.0, 0.0, 0.0, 1.0);
    for (uint b = 0u; b < bands; b++) {
        float max_val = 0.0;
        for (uint i = 0u; i < width; i++) {
            float v = hist.read(uint2(i, 0u))[b];
            if (v > max_val) { max_val = v; }
        }
        float v = hist.read(uint2(idx.x, 0u))[b];
        float bar = max_val > 0.0 ? (v / max_val * float(width)) : 0.0;
        res[b] = (float(width - 1u - idx.y) < bar) ? 1.0 : 0.0;
    }
    output.write(idx, res);
}

// Cumulative distribution function -> LUT, per band. `hist` is a staged
// (already-accumulated) `bins * bands` histogram. dispatch domain = (bins, 1);
// idx.x = LUT entry index, shared across bands (packed into one float4).
// Output entries are normalized to [0, 1] (the working-space range `maplut`
// expects). If a band's total count is zero, falls back to an identity ramp.
public void equalize_lut_kernel(
    uint2 idx,
    HistogramIn hist,
    RWRegion output,
    uint bins,
    uint bands
) {
    float4 res = float4(0.0, 0.0, 0.0, 1.0);
    for (uint b = 0u; b < bands; b++) {
        uint total = 0u;
        uint cum = 0u;
        for (uint i = 0u; i < bins; i++) {
            uint count = hist.read(b * bins + i);
            // Exclusive prefix sum: bin 0 always maps to 0, matching vips'
            // hist_equal (so a constant image maps to all-zero, not all-max).
            if (i < idx.x) {
                cum += count;
            }
            total += count;
        }
        res[b] = total > 0u
            ? (float(cum) / float(total))
            : (float(idx.x) / float(max(bins, 2u) - 1u));
    }
    if (bands == 1u) {
        // Replicate the single band's curve to G/B so `maplut`'s default
        // per-channel mapping (band < 0) applies the same curve everywhere.
        res = float4(res.r, res.r, res.r, 1.0);
    }
    output.write(idx, res);
}

// Cumulative histogram: `out[b*bins+i] = sum(in[b*bins .. b*bins+i])`, per
// band. dispatch domain = (bins, 1); each thread sums its own prefix.
public void histogram_cumulative_kernel(
    uint2 idx,
    HistogramIn hist_in,
    HistogramOut hist_out,
    uint bins,
    uint bands
) {
    for (uint b = 0u; b < bands; b++) {
        uint cum = 0u;
        for (uint i = 0u; i <= idx.x; i++) {
            cum += hist_in.read(b * bins + i);
        }
        hist_out.write(b * bins + idx.x, cum);
    }
}

// Normalize: scale each band so its maximum bin value maps to `bins - 1`.
// dispatch domain = (bins, 1). Each thread first finds its band's max (over
// all `bins` entries), then scales its own entry.
public void histogram_normalize_kernel(
    uint2 idx,
    HistogramIn hist_in,
    HistogramOut hist_out,
    uint bins,
    uint bands
) {
    for (uint b = 0u; b < bands; b++) {
        uint max_val = 0u;
        for (uint i = 0u; i < bins; i++) {
            uint v = hist_in.read(b * bins + i);
            if (v > max_val) { max_val = v; }
        }
        uint v = hist_in.read(b * bins + idx.x);
        uint scaled = max_val > 0u
            ? uint(round(float(v) * float(bins - 1u) / float(max_val)))
            : v;
        hist_out.write(b * bins + idx.x, scaled);
    }
}
