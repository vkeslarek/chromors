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

// ── Invertlut ────────────────────────────────────────────────────────────────
//
// `lut` holds `height` rows of `(x, y0, y1, y2)` packed into float4 — column
// 0 is the target value (vips' "in" matrix column 0), columns 1.. are the
// real-value vertices per output band (vips' "in" matrix columns 1..bands).
// Rows are assumed pre-sorted by column 0 (vips sorts internally; the GPU
// port skips that step and requires sorted input).
//
// For each output band `bi` and output index `idx.x` (`k`), this mirrors
// `vips_invertlut`'s piecewise-linear inversion + head/tail extrapolation.

public void invertlut_kernel<R1: IRegion>(
    uint2 idx, R1 lut, RWRegion output,
    uint height, uint size, uint bands)
{
    int k = int(idx.x);
    int last_idx = int(size) - 1;
    float4 result = float4(0.0);

    for (uint bi = 0u; bi < bands; bi++)
    {
        float4 row0 = lut.read(uint2(0u, 0u));
        float4 rowN = lut.read(uint2(height - 1u, 0u));
        float y0 = row0[bi + 1u];
        float yN = rowN[bi + 1u];
        float x0 = row0.x;
        float xN = rowN.x;

        int first = int(y0 * float(last_idx));
        int last = int(yN * float(last_idx));

        float val;
        if (k >= first && k <= last)
        {
            float ki = float(k) / float(last_idx);

            int j = -1;
            for (int jj = int(height) - 1; jj >= 0; jj--)
            {
                if (lut.read(uint2(uint(jj), 0u))[bi + 1u] < ki)
                {
                    j = jj;
                    break;
                }
            }
            if (j == -1) j = 0;

            if (height > 1u)
            {
                int j1 = min(j + 1, int(height) - 1);
                float yj = lut.read(uint2(uint(j), 0u))[bi + 1u];
                float yj1 = lut.read(uint2(uint(j1), 0u))[bi + 1u];
                float xj = lut.read(uint2(uint(j), 0u)).x;
                float xj1 = lut.read(uint2(uint(j1), 0u)).x;
                float irange = yj1 - yj;
                float orange = xj1 - xj;
                val = (irange != 0.0) ? (xj + orange * (ki - yj) / irange) : xj;
            }
            else
            {
                val = x0;
            }
        }
        else if (k < first)
        {
            float fac = (first != 0) ? (x0 / float(first)) : 0.0;
            val = float(k) * fac;
        }
        else
        {
            float fac = (1.0 - xN) / float(last_idx - last);
            val = xN + float(k - last) * fac;
        }

        result[bi] = val;
    }

    output.write(idx, result);
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
