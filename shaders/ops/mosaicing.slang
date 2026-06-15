import lib.region;

// ── Merge ────────────────────────────────────────────────────────────────────
//
// Approximation of vips' merge/lrmerge/tbmerge: places `ref_img` at
// (r_left, r_top) and `sec_img` at (s_left, s_top) on the output canvas, and
// raised-cosine blends the two across their overlap. Differs from vips in
// that the blend window is the *whole* overlap (clamped to `max_blend`)
// rather than vips' per-scanline search for the first/last non-transparent
// pixel -- i.e. this assumes both inputs are fully opaque rectangles, which
// holds for the common "two photos, no irregular edges" mosaicing case.

public void merge_kernel<R1: IRegion, R2: IRegion>(
    uint2 idx, R1 ref_img, R2 sec_img, RWRegion output,
    uint direction, int r_left, int r_top, int s_left, int s_top,
    uint ref_w, uint ref_h, uint sec_w, uint sec_h,
    int first, float bwidth)
{
    int2 p = int2(idx);
    int2 rp = p - int2(r_left, r_top);
    int2 sp = p - int2(s_left, s_top);

    bool in_r = rp.x >= 0 && rp.x < int(ref_w) && rp.y >= 0 && rp.y < int(ref_h);
    bool in_s = sp.x >= 0 && sp.x < int(sec_w) && sp.y >= 0 && sp.y < int(sec_h);

    float4 ref_col = in_r ? ref_img.read(uint2(rp)) : float4(0);
    float4 sec_col = in_s ? sec_img.read(uint2(sp)) : float4(0);

    float4 result;
    if (in_r && in_s) {
        int axis = direction == 0u ? p.x : p.y;
        float w_ref;
        if (bwidth <= 0.0) {
            w_ref = float(axis) < float(first) ? 1.0 : 0.0;
        } else {
            float t = clamp((float(axis) - float(first)) / bwidth, 0.0, 1.0);
            w_ref = (cos(3.14159265358979 * t) + 1.0) * 0.5;
        }
        result = w_ref * ref_col + (1.0 - w_ref) * sec_col;
    } else if (in_r) {
        result = ref_col;
    } else if (in_s) {
        result = sec_col;
    } else {
        result = float4(0);
    }
    output.write(idx, result);
}
