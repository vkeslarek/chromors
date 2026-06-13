import lib.region;

// `radius` is precomputed in Rust to match vips_gaussmat's mask size (the
// largest x with exp(-x^2 / (2*sigma^2)) >= min_ampl=0.2), not sigma*3 --
// vips's default mask is much narrower than a 3-sigma window.
//
// Single-pass 2D blur (not separable H/V): a separable two-step fused pass
// would have the V step read NEIGHBOR threads' H output (`work_0`), but a
// single dispatch gives no ordering/visibility guarantee across workgroups
// for that kind of cross-thread temp read. Reading the SOURCE directly with
// a full (2*radius+1)^2 window avoids any cross-thread dependency: every
// thread only reads decoded source pixels, which are valid for the whole
// dispatch.
public void blur_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float sigma, int radius)
{
    if (radius == 0) {
        output.write(idx, input.read_clamped(int2(idx)));
        return;
    }
    float two_sigma2 = 2.0 * sigma * sigma;
    float4 acc = float4(0);
    float wsum = 0.0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float w = exp(-float(x * x + y * y) / two_sigma2);
            acc += w * input.read_clamped(int2(idx) + int2(x, y));
            wsum += w;
        }
    }
    output.write(idx, acc / wsum);
}

// ── Unsharp mask ───────────────────────────────────────────────────────────
//
// `result = orig + m1 * (orig - gaussian_blur(orig, sigma))`, clamped to
// [0,1]. A single-gain approximation of vips `sharpen` (which works in LAB on
// the L channel only with a piecewise-linear flat/jagged/edge gain curve);
// this applies a uniform gain to all working-space channels.

public void sharpen_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float sigma, float m1)
{
    float4 orig = input.read_clamped(int2(idx));
    int r = int(ceil(sigma * 3.0));
    if (r == 0) {
        output.write(idx, orig);
        return;
    }
    float two_sigma2 = 2.0 * sigma * sigma;
    float4 acc = float4(0);
    float wsum = 0.0;
    for (int y = -r; y <= r; y++) {
        for (int x = -r; x <= r; x++) {
            float w = exp(-float(x * x + y * y) / two_sigma2);
            acc += w * input.read_clamped(int2(idx) + int2(x, y));
            wsum += w;
        }
    }
    float4 blurred = acc / wsum;
    float4 result = orig + m1 * (orig - blurred);
    output.write(idx, clamp(result, float4(0.0), float4(1.0)));
}
