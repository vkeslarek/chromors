import lib.region;

// ── Single-image linear ──────────────────────────────────────────────────────
public void linear_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float a, float b)
{
    float4 c = input.read(idx);
    c = c * a + float4(b, b, b, 0.0);
    output.write(idx, c);
}

// ── Single-image math (sin, cos, tan, ...) ───────────────────────────────────
public void math_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint math)
{
    float4 c = input.read(idx);
    float v = c.r;
    float r;
    switch (math) {
        case  0u: r = sin(v);                   break;
        case  1u: r = cos(v);                   break;
        case  2u: r = tan(v);                   break;
        case  3u: r = asin(clamp(v, -1.0, 1.0)); break;
        case  4u: r = acos(clamp(v, -1.0, 1.0)); break;
        case  5u: r = atan(v);                  break;
        case  6u: r = log(v);                   break;
        case  7u: r = log10(v);                 break;
        case  8u: r = exp(v);                   break;
        case  9u: r = exp(v * 2.302585093);     break;
        case 10u: r = sinh(v);                  break;
        case 11u: r = cosh(v);                  break;
        case 12u: r = tanh(v);                  break;
        case 13u: r = log(v + sqrt(v * v + 1.0)); break;
        case 14u: r = log(v + sqrt(v * v - 1.0)); break;
        case 15u: r = 0.5 * log((1.0 + v) / max(1.0 - v, 1e-10)); break;
        default:  r = v; break;
    }
    output.write(idx, float4(r, r, r, c.a));
}

// ── Single-image round (rint, ceil, floor) ───────────────────────────────────
//
// vips round/floor/ceil/rint operate on the *raw pixel value* (a no-op for
// integer formats), not the decoded [0,1] working value -- rescale by
// src_max before rounding and back after, matching morph_kernel's convention.
public void round_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint round_op, float src_max)
{
    float4 c = input.read(idx) * src_max;
    switch (round_op) {
        case 0u: c = round(c); break;
        case 1u: c = ceil(c);  break;
        case 2u: c = floor(c); break;
        default: break;
    }
    output.write(idx, c / src_max);
}

// ── Single-image math2_const (pow, wop, atan2 with constant) ─────────────────
public void math2_const_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint math2, uint n_consts, float c0, float c1, float c2, float c3, float c4, float c5, float c6, float c7, float c8, float c9)
{
    float4 c = input.read(idx);
    float3 rgb = c.rgb;
    switch (math2) {
        case 0u: rgb = pow(rgb, float3(c0, c0, c0));         break;
        case 1u: rgb = pow(float3(c0, c0, c0), rgb);         break;
        case 2u: rgb = atan2(rgb, float3(c0, c0, c0));       break;
        default: break;
    }
    output.write(idx, float4(rgb, c.a));
}

// ── Single-image boolean_const (and, or, eor, lshift, rshift) ────────────────
public void boolean_const_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint boolean_op, uint n_consts, float c0, float c1, float c2, float c3, float c4, float c5, float c6, float c7, float c8, float c9)
{
    float4 c = input.read(idx);
    int3 v = int3(c.rgb);
    int k = int(c0);
    int3 r;
    switch (boolean_op) {
        case 0u: r = v & int3(k, k, k); break;
        case 1u: r = v | int3(k, k, k); break;
        case 2u: r = v ^ int3(k, k, k); break;
        case 3u: r = v << k;            break;
        case 4u: r = v >> k;            break;
        default: r = v;                 break;
    }
    output.write(idx, float4(float3(r), c.a));
}

// ── Single-image relational_const (equal, notequal, less, ...) ───────────────
public void relational_const_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint relational, uint n_consts, float c0, float c1, float c2, float c3, float c4, float c5, float c6, float c7, float c8, float c9)
{
    float4 c = input.read(idx);
    float3 v = c.rgb;
    float3 r;
    switch (relational) {
        case 0u: r = float3(v == float3(c0, c0, c0) ? 1.0 : 0.0); break;
        case 1u: r = float3(v != float3(c0, c0, c0) ? 1.0 : 0.0); break;
        case 2u: r = float3(v < float3(c0, c0, c0) ? 1.0 : 0.0);  break;
        case 3u: r = float3(v <= float3(c0, c0, c0) ? 1.0 : 0.0); break;
        case 4u: r = float3(v > float3(c0, c0, c0) ? 1.0 : 0.0);  break;
        case 5u: r = float3(v >= float3(c0, c0, c0) ? 1.0 : 0.0); break;
        default: r = v; break;
    }
    output.write(idx, float4(r, c.a));
}

// ── Single-image remainder_const ─────────────────────────────────────────────
public void remainder_const_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint n_consts, float c0, float c1, float c2, float c3, float c4, float c5, float c6, float c7, float c8, float c9)
{
    float4 c = input.read(idx);
    float3 rgb = c.rgb;
    float d = max(abs(c0), 1e-10);
    rgb = rgb - d * floor(rgb / d);
    output.write(idx, float4(rgb, c.a));
}

// ── Two-image arithmetic ─────────────────────────────────────────────────────

public void add_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    output.write(idx, a + b);
}

public void subtract_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    output.write(idx, a - b);
}

public void multiply_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    output.write(idx, a * b);
}

public void divide_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    float4 eps = float4(1e-10, 1e-10, 1e-10, 1.0);
    output.write(idx, a / (b + eps));
}

public void max_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    output.write(idx, max(a, b));
}

public void min_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    output.write(idx, min(a, b));
}

// vips remainder is integer modulo on raw pixel values: rescale to the raw
// domain by src_max before dividing. A small epsilon on the quotient guards
// against GPU division using an approximate reciprocal (e.g. a/d landing at
// 0.99999994 instead of 1.0 when a == b) -- safe because the true fractional
// part of a raw-integer ratio is always >= 1/src_max, far larger than the
// epsilon.
public void remainder_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output, float src_max)
{
    float4 a_norm = left.read(idx);
    float4 a = a_norm * src_max;
    float4 b = right.read(idx) * src_max;
    float4 d = max(abs(b), float4(1e-10, 1e-10, 1e-10, 1.0));
    float4 q = floor(a / d + 1e-4);
    float4 r = (a - d * q) / src_max;
    // vips convention for unsigned formats: x % 0 == TYPE_MAX, i.e. 1.0 once
    // normalised to [0,1].
    if (b.x == 0.0) r.x = 1.0;
    if (b.y == 0.0) r.y = 1.0;
    if (b.z == 0.0) r.z = 1.0;
    if (b.w == 0.0) r.w = 1.0;
    r.a = a_norm.a;
    output.write(idx, r);
}

public void hypot_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    output.write(idx, float4(sqrt(a.rgb * a.rgb + b.rgb * b.rgb), a.a));
}

// ── Two-image + enum ─────────────────────────────────────────────────────────

public void math2_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output, uint math2)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    float3 ra;
    switch (math2) {
        case 0u: ra = pow(a.rgb, b.rgb);       break;
        case 1u: ra = pow(b.rgb, a.rgb);       break;
        case 2u: ra = atan2(a.rgb, b.rgb);    break;
        default: ra = a.rgb;                   break;
    }
    output.write(idx, float4(ra, a.a));
}

public void boolean_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output, uint boolean_op)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    int3 va = int3(a.rgb);
    int3 vb = int3(b.rgb);
    int3 r;
    switch (boolean_op) {
        case 0u: r = va & vb;  break;
        case 1u: r = va | vb;  break;
        case 2u: r = va ^ vb;  break;
        case 3u: r = va << vb; break;
        case 4u: r = va >> vb; break;
        default: r = va;      break;
    }
    output.write(idx, float4(float3(r), a.a));
}

public void relational_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output, uint relational)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);
    float3 r;
    switch (relational) {
        case 0u: r = float3(a.rgb == b.rgb ? 1.0 : 0.0); break;
        case 1u: r = float3(a.rgb != b.rgb ? 1.0 : 0.0); break;
        case 2u: r = float3(a.rgb <  b.rgb ? 1.0 : 0.0); break;
        case 3u: r = float3(a.rgb <= b.rgb ? 1.0 : 0.0); break;
        case 4u: r = float3(a.rgb >  b.rgb ? 1.0 : 0.0); break;
        case 5u: r = float3(a.rgb >= b.rgb ? 1.0 : 0.0); break;
        default: r = a.rgb; break;
    }
    output.write(idx, float4(r, a.a));
}

public void complexform_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 real_in, R2 imag_in, RWRegion output)
{
    float4 r = real_in.read(idx);
    float4 i = imag_in.read(idx);
    output.write(idx, float4(r.r, i.r, 0.0, r.a));
}

// ── Complex operations (float4 pairs: (r,g) = pair0, (b,a) = pair1) ──────────

public void complex_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint cmplx)
{
    float4 c = input.read(idx);
    float2 p0 = c.rg;
    float2 p1 = c.ba;
    float2 r0, r1;
    switch (cmplx) {
        case 0u:   // Polar: (real, imag) -> (mag, angle)
            r0 = float2(length(p0), atan2(p0.y, p0.x));
            r1 = float2(length(p1), atan2(p1.y, p1.x));
            break;
        case 1u:   // Rect: (mag, angle) -> (real, imag)
            r0 = float2(p0.x * cos(p0.y), p0.x * sin(p0.y));
            r1 = float2(p1.x * cos(p1.y), p1.x * sin(p1.y));
            break;
        case 2u:   // Conj: conjugate
            r0 = float2(p0.x, -p0.y);
            r1 = float2(p1.x, -p1.y);
            break;
        default:
            r0 = p0; r1 = p1; break;
    }
    output.write(idx, float4(r0.x, r0.y, r1.x, r1.y));
}

public void complexget_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint get)
{
    float4 c = input.read(idx);
    switch (get) {
        case 0u:   // Real — zero out imag channels
            c = float4(c.r, 0.0, c.b, 0.0);
            break;
        case 1u:   // Imag — zero out real channels
            c = float4(0.0, c.g, 0.0, c.a);
            break;
        default: break;
    }
    output.write(idx, c);
}

public void complex2_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 left, R2 right, RWRegion output, uint cmplx)
{
    float4 a = left.read(idx);
    float4 b = right.read(idx);

    float2 p0a = a.rg;
    float2 p1a = a.ba;
    float2 p0b = b.rg;
    float2 p1b = b.ba;
    float2 r0, r1;

    switch (cmplx) {
        case 0u:   // CrossPhase: a * conj(b)
            r0 = float2(p0a.x * p0b.x + p0a.y * p0b.y,
                        p0a.y * p0b.x - p0a.x * p0b.y);
            r1 = float2(p1a.x * p1b.x + p1a.y * p1b.y,
                        p1a.y * p1b.x - p1a.x * p1b.y);
            break;
        default:
            r0 = p0a; r1 = p1a; break;
    }
    output.write(idx, float4(r0.x, r0.y, r1.x, r1.y));
}
