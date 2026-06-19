public uint2 pcg2d(uint2 v) {
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    return v;
}

public uint2 key(int2 lod_coord, int lod, uint seed) {
    uint2 c = uint2(lod_coord);
    uint s1 = seed;
    uint s2 = seed * 747796405u + uint(lod) * 2891336453u;
    return pcg2d(c ^ uint2(s1, s2));
}

public float2 rand_f32_2(uint2 state) {
    uint2 r = pcg2d(state);
    float x = float(r.x >> 8) * (1.0 / 16777216.0);
    float y = float(r.y >> 8) * (1.0 / 16777216.0);
    return float2(x, y);
}

public float gauss(int2 lod_coord, int lod, uint seed) {
    uint2 k = key(lod_coord, lod, seed);
    float2 r1 = rand_f32_2(k);
    float2 r2 = rand_f32_2(pcg2d(k)); // next state
    
    // Box-Muller transform
    float u1 = max(r1.x, 1e-7);
    float u2 = r2.x;
    float r = sqrt(-2.0 * log(u1));
    float theta = 6.28318530718 * u2;
    return r * cos(theta);
}
