import lib.region;

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

public void complexform_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 real_in, R2 imag_in, RWRegion output)
{
    float4 r = real_in.read(idx);
    float4 i = imag_in.read(idx);
    output.write(idx, float4(r.r, i.r, 0.0, r.a));
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



public void linear_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float a0, float a1, float a2, float a3, float b0, float b1, float b2, float b3)
{
    float4 c = input.read(idx);
    output.write(idx, c * float4(a0, a1, a2, a3) + float4(b0, b1, b2, b3));
}

public void math2_const_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint math2, float src_max, float c0, float c1, float c2, float c3)
{
    float4 a_norm = input.read(idx);
    float4 a = a_norm * src_max;
    float3 ra;
    switch (math2) {
        case 0u: ra = pow(a.rgb, float3(c0, c1, c2));       break;
        case 1u: ra = pow(float3(c0, c1, c2), a.rgb);       break;
        case 2u: ra = atan2(a.rgb, float3(c0, c1, c2));    break;
        default: ra = a.rgb;                   break;
    }
    output.write(idx, float4(ra / src_max, a_norm.a));
}

public void remainder_const_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, float src_max, float c0, float c1, float c2, float c3)
{
    float4 a_norm = input.read(idx);
    float4 a = a_norm * src_max;
    float4 b = float4(c0, c1, c2, c3);
    float4 d = max(abs(b), float4(1e-10, 1e-10, 1e-10, 1.0));
    float4 q = floor(a / d + 1e-4);
    float4 r = (a - d * q) / src_max;
    if (b.x == 0.0) r.x = 1.0;
    if (b.y == 0.0) r.y = 1.0;
    if (b.z == 0.0) r.z = 1.0;
    if (b.w == 0.0) r.w = 1.0;
    r.a = a_norm.a;
    output.write(idx, r);
}
