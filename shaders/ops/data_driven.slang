import lib.region;

public void maplut_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 input, R2 lut, RWRegion output, uint lut_width, int band) {
    float4 c = input.read(idx);
    float4 res;
    // For each channel, map it via LUT. LUT image has width = lut_width, height = 1.
    // The channel's float value is [0, 1] mapped to [0, lut_width - 1].
    for (int i = 0; i < 4; i++) {
        uint idx_lut = uint(clamp(c[i], 0.0, 1.0) * float(lut_width - 1));
        float4 lut_val = lut.read(uint2(idx_lut, 0));
        res[i] = lut_val[band >= 0 ? band : i];
    }
    output.write(idx, res);
}

public void recomb_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 input, R2 matrix, RWRegion output, uint n) {
    float4 c = input.read(idx);
    float4 res = float4(0.0);
    // matrix is n x n, stored as an image. i corresponds to the output channel.
    for (uint i = 0; i < n; i++) {
        float sum = 0.0;
        for (uint j = 0; j < n; j++) {
            float weight = matrix.read(uint2(j, i)).r; // 1-band image
            sum += c[j] * weight;
        }
        res[i] = sum;
    }
    // Any remaining channels > n pass through or default to 1.0 for alpha?
    // Let's pass through or leave at 0 (and alpha at 1)
    if (n < 4 && n >= 3) { res.a = c.a; } // Preserve alpha if RGB->RGB matrix
    output.write(idx, res);
}

public void ifthenelse_kernel<R1: IRegion, R2: IRegion, R3: IRegion>(uint2 idx, R1 cond, R2 if_true, R3 if_false, RWRegion output, uint blend) {
    float4 c_cond = cond.read(idx);
    float4 t = if_true.read(idx);
    float4 f = if_false.read(idx);
    float4 res;
    if (blend != 0) {
        res = lerp(f, t, c_cond); // Per channel blend
    } else {
        res = float4(
            c_cond.r > 0.5 ? t.r : f.r,
            c_cond.g > 0.5 ? t.g : f.g,
            c_cond.b > 0.5 ? t.b : f.b,
            c_cond.a > 0.5 ? t.a : f.a
        );
    }
    output.write(idx, res);
}

public void case1_kernel<R1: IRegion, R2: IRegion>(uint2 idx, R1 input, R2 c0, RWRegion output) {
    uint sel = uint(round(input.read(idx).r));
    output.write(idx, c0.read(idx)); // Only 0 is valid for 1 case
}

public void case2_kernel<R1: IRegion, R2: IRegion, R3: IRegion>(uint2 idx, R1 input, R2 c0, R3 c1, RWRegion output) {
    uint sel = uint(round(input.read(idx).r));
    float4 res;
    switch (sel) {
        case 0: res = c0.read(idx); break;
        case 1: res = c1.read(idx); break;
        default: res = float4(0.0); break;
    }
    output.write(idx, res);
}

public void case3_kernel<R1: IRegion, R2: IRegion, R3: IRegion, R4: IRegion>(uint2 idx, R1 input, R2 c0, R3 c1, R4 c2, RWRegion output) {
    uint sel = uint(round(input.read(idx).r));
    float4 res;
    switch (sel) {
        case 0: res = c0.read(idx); break;
        case 1: res = c1.read(idx); break;
        case 2: res = c2.read(idx); break;
        default: res = float4(0.0); break;
    }
    output.write(idx, res);
}

public void case4_kernel<R1: IRegion, R2: IRegion, R3: IRegion, R4: IRegion, R5: IRegion>(uint2 idx, R1 input, R2 c0, R3 c1, R4 c2, R5 c3, RWRegion output) {
    uint sel = uint(round(input.read(idx).r));
    float4 res;
    switch (sel) {
        case 0: res = c0.read(idx); break;
        case 1: res = c1.read(idx); break;
        case 2: res = c2.read(idx); break;
        case 3: res = c3.read(idx); break;
        default: res = float4(0.0); break;
    }
    output.write(idx, res);
}

public void case5_kernel<R1: IRegion, R2: IRegion, R3: IRegion, R4: IRegion, R5: IRegion, R6: IRegion>(uint2 idx, R1 input, R2 c0, R3 c1, R4 c2, R5 c3, R6 c4, RWRegion output) {
    uint sel = uint(round(input.read(idx).r));
    float4 res;
    switch (sel) {
        case 0: res = c0.read(idx); break;
        case 1: res = c1.read(idx); break;
        case 2: res = c2.read(idx); break;
        case 3: res = c3.read(idx); break;
        case 4: res = c4.read(idx); break;
        default: res = float4(0.0); break;
    }
    output.write(idx, res);
}
