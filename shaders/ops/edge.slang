import lib.region;

// ── Sobel gradient magnitude ──────────────────────────────────────────────────
// Gx = [-1,0,+1; -2,0,+2; -1,0,+1]   Gy = [-1,-2,-1; 0,0,0; +1,+2,+1]
public void sobel_kernel<R: IRegion>(uint2 idx, R input, RWRegion output) {
    int2 p = int2(idx);
    float4 tl = input.read_clamped(p + int2(-1,-1));
    float4 tc = input.read_clamped(p + int2( 0,-1));
    float4 tr = input.read_clamped(p + int2( 1,-1));
    float4 ml = input.read_clamped(p + int2(-1, 0));
    float4 mr = input.read_clamped(p + int2( 1, 0));
    float4 bl = input.read_clamped(p + int2(-1, 1));
    float4 bc = input.read_clamped(p + int2( 0, 1));
    float4 br = input.read_clamped(p + int2( 1, 1));
    float4 gx = -tl + tr - 2.0*ml + 2.0*mr - bl + br;
    float4 gy = -tl - 2.0*tc - tr + bl + 2.0*bc + br;
    output.write(idx, sqrt(gx*gx + gy*gy));
}

// ── Prewitt gradient magnitude ────────────────────────────────────────────────
// Gx = [-1,0,+1; -1,0,+1; -1,0,+1]   Gy = [-1,-1,-1; 0,0,0; +1,+1,+1]
public void prewitt_kernel<R: IRegion>(uint2 idx, R input, RWRegion output) {
    int2 p = int2(idx);
    float4 tl = input.read_clamped(p + int2(-1,-1));
    float4 tc = input.read_clamped(p + int2( 0,-1));
    float4 tr = input.read_clamped(p + int2( 1,-1));
    float4 ml = input.read_clamped(p + int2(-1, 0));
    float4 mr = input.read_clamped(p + int2( 1, 0));
    float4 bl = input.read_clamped(p + int2(-1, 1));
    float4 bc = input.read_clamped(p + int2( 0, 1));
    float4 br = input.read_clamped(p + int2( 1, 1));
    float4 gx = -tl + tr - ml + mr - bl + br;
    float4 gy = -tl - tc - tr + bl + bc + br;
    output.write(idx, sqrt(gx*gx + gy*gy));
}

// ── Scharr gradient magnitude ─────────────────────────────────────────────────
// Gx = [-3,0,+3; -10,0,+10; -3,0,+3]   Gy = [-3,-10,-3; 0,0,0; +3,+10,+3]
public void scharr_kernel<R: IRegion>(uint2 idx, R input, RWRegion output) {
    int2 p = int2(idx);
    float4 tl = input.read_clamped(p + int2(-1,-1));
    float4 tc = input.read_clamped(p + int2( 0,-1));
    float4 tr = input.read_clamped(p + int2( 1,-1));
    float4 ml = input.read_clamped(p + int2(-1, 0));
    float4 mr = input.read_clamped(p + int2( 1, 0));
    float4 bl = input.read_clamped(p + int2(-1, 1));
    float4 bc = input.read_clamped(p + int2( 0, 1));
    float4 br = input.read_clamped(p + int2( 1, 1));
    float4 gx = -3.0*tl + 3.0*tr - 10.0*ml + 10.0*mr - 3.0*bl + 3.0*br;
    float4 gy = -3.0*tl - 10.0*tc - 3.0*tr + 3.0*bl + 10.0*bc + 3.0*br;
    output.write(idx, sqrt(gx*gx + gy*gy));
}

public void invert_kernel<R: IRegion>(uint2 idx, R input, RWRegion output)
{
    // Negate every band (matches libvips `invert`, which inverts alpha too).
    float4 c = input.read(idx);
    output.write(idx, float4(1.0, 1.0, 1.0, 1.0) - c);
}

public void sign_kernel<R: IRegion>(uint2 idx, R input, RWRegion output) {
    float4 c = input.read(idx);
    output.write(idx, sign(c));
}

public void abs_kernel<R: IRegion>(uint2 idx, R input, RWRegion output) {
    float4 c = input.read(idx);
    output.write(idx, abs(c));
}

public void hypot_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    output.write(idx, float4(sqrt(a.rgb * a.rgb + b.rgb * b.rgb), a.a));
}

