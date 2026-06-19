import lib.region;

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

void insertion_sort_121(inout float arr[121], int count)
{
    for (int i = 1; i < count; i++)
    {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

public void median_kernel<R1: IRegion>(
    uint2 idx, R1 input, RWRegion output,
    uint sz)
{
    int half = int(sz) >> 1;
    int count = 0;
    float r[121];
    float g[121];
    float b[121];
    float a[121];

    for (uint my = 0u; my < sz; my++)
    {
        for (uint mx = 0u; mx < sz; mx++)
        {
            int2 src_pos = int2(idx) + int2(int(mx) - half, int(my) - half);
            float4 src = input.read_clamped(src_pos);
            r[count] = src.r;
            g[count] = src.g;
            b[count] = src.b;
            a[count] = src.a;
            count++;
        }
    }

    insertion_sort_121(r, count);
    insertion_sort_121(g, count);
    insertion_sort_121(b, count);
    insertion_sort_121(a, count);

    int mid = count / 2;
    output.write(idx, float4(r[mid], g[mid], b[mid], a[mid]));
}

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

// ── Canny ─────────────────────────────────────────────────────────────────────
//
// Approximation of vips' Canny edge detector: gaussian-blur the input, take a
// 2x2 forward-difference gradient (Gx/Gy), then suppress non-maxima along the
// gradient direction. Differs from vips in two ways: (1) the non-max check
// snaps to the nearest of 8 octants instead of bilinearly interpolating
// between the two neighboring octants, and (2) there's no separate hysteresis
// pass (vips doesn't have one either -- "you will probably need to process
// the output further to eliminate weak edges").

/// Single-channel gaussian blur at `p`, same weighting as `blur_kernel`.
float canny_blur<G: IRegion>(G input, int2 p, int ch, float sigma, int radius) {
    float two_sigma2 = 2.0 * sigma * sigma;
    float acc = 0.0;
    float wsum = 0.0;
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float w = exp(-float(x * x + y * y) / two_sigma2);
            float4 c = input.read_clamped(p + int2(x, y));
            float v = ch == 0 ? c.r : (ch == 1 ? c.g : c.b);
            acc += w * v;
            wsum += w;
        }
    }
    return acc / wsum;
}

/// Gradient magnitude `G` and direction `theta_deg` at `p`, from a 2x2
/// forward-difference of the blurred channel `ch`.
float canny_g<G: IRegion>(G input, int2 p, int ch, float sigma, int radius, out float theta_deg) {
    float b00 = canny_blur(input, p + int2(-1, -1), ch, sigma, radius);
    float b10 = canny_blur(input, p + int2(0, -1), ch, sigma, radius);
    float b01 = canny_blur(input, p + int2(-1, 0), ch, sigma, radius);
    float b11 = canny_blur(input, p + int2(0, 0), ch, sigma, radius);
    float gx = (b10 + b11) - (b00 + b01);
    float gy = (b01 + b11) - (b00 + b10);
    theta_deg = degrees(atan2(gx, gy));
    return (gx * gx + gy * gy) * 0.5;
}

public void canny_kernel<G: IRegion>(uint2 idx, G input, RWRegion output, float sigma, int radius) {
    int2 p = int2(idx);
    int2 offsets[8] = {
        int2(0, -1), int2(-1, -1), int2(-1, 0), int2(-1, 1),
        int2(0, 1), int2(1, 1), int2(1, 0), int2(1, -1),
    };
    float4 result = float4(0.0, 0.0, 0.0, 1.0);
    for (int ch = 0; ch < 3; ch++) {
        float theta;
        float g = canny_g(input, p, ch, sigma, radius, theta);
        if (theta < 0.0) theta += 360.0;
        int dir = int(theta / 45.0 + 0.5) & 7;
        int opp = (dir + 4) & 7;

        float theta_n, theta_o;
        float gn = canny_g(input, p + offsets[dir], ch, sigma, radius, theta_n);
        float go = canny_g(input, p + offsets[opp], ch, sigma, radius, theta_o);

        // vips keeps G as a small float (its "+256" baseline puts a flat
        // region at ~0.5/255 of the uchar range); our U8-encoded output
        // would round nearly everything to 0 at that scale, so apply a
        // fixed gain to spread typical edge magnitudes across the u8 range.
        const float GAIN = 16.0;
        float v = (g <= gn || g < go) ? 0.0 : clamp(g * GAIN, 0.0, 1.0);
        if (ch == 0) result.r = v;
        else if (ch == 1) result.g = v;
        else result.b = v;
    }
    output.write(idx, result);
}

