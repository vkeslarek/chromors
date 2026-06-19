pub fn pcg2d(mut v: [u32; 2]) -> [u32; 2] {
    v[0] = v[0].wrapping_mul(1664525).wrapping_add(1013904223);
    v[1] = v[1].wrapping_mul(1664525).wrapping_add(1013904223);

    v[0] = v[0].wrapping_add(v[1].wrapping_mul(1664525));
    v[1] = v[1].wrapping_add(v[0].wrapping_mul(1664525));

    v[0] ^= v[0] >> 16;
    v[1] ^= v[1] >> 16;

    v[0] = v[0].wrapping_add(v[1].wrapping_mul(1664525));
    v[1] = v[1].wrapping_add(v[0].wrapping_mul(1664525));

    v[0] ^= v[0] >> 16;
    v[1] ^= v[1] >> 16;

    v
}

pub fn key(seed: u32, x: u32, y: u32) -> [u32; 2] {
    let mut v = [x, y];
    v[0] = v[0].wrapping_add(seed);
    v[1] = v[1].wrapping_add(seed);
    pcg2d(v)
}

pub fn rand_f32(mut state: [u32; 2]) -> (f32, [u32; 2]) {
    state = pcg2d(state);
    let f = (state[0] >> 8) as f32 / 16777216.0;
    (f, state)
}

pub fn gauss(mut state: [u32; 2]) -> (f32, [u32; 2]) {
    let (u1, s1) = rand_f32(state);
    let (u2, s2) = rand_f32(s1);
    let r = (-2.0 * u1.max(1e-7).ln()).sqrt();
    let theta = std::f32::consts::TAU * u2;
    (r * theta.cos(), s2)
}
