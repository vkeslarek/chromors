with open("tests/cross_backend/filters.rs", "r") as f:
    code = f.read()

code = code.replace("""    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("morph(Dilate) RMS = {}", rms);""", """    let mut diffs = 0;
    for i in 0..vips_bytes.len() {
        if vips_bytes[i] != gpu_bytes[i] {
            if diffs < 10 {
                println!("diff at {}: vips={}, gpu={}, c={}", i, vips_bytes[i], gpu_bytes[i], i % 4);
            }
            diffs += 1;
        }
    }
    println!("Total diffs: {}", diffs);
    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("morph(Dilate) RMS = {}", rms);""")

with open("tests/cross_backend/filters.rs", "w") as f:
    f.write(code)
