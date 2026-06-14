use poc::*;

fn main() {
    let _g = tests::common::vips_serial();
    let ctx = tests::common::gpu_ctx();
    let vips_img = tests::common::rgb();
    let gpu_img = tests::common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.embed(
        20,
        20,
        240,
        240,
        Some(poc::operation::geometry::Extend::Background),
        Some([0.0, 0.0, 0.0]),
    );
    let gpu_res = gpu_img.embed(
        20,
        20,
        240,
        240,
        Some(poc::operation::geometry::Extend::Background),
        Some([0.0, 0.0, 0.0]),
    );

    let vips_bytes = tests::common::vips_materialize(&vips_res);
    let gpu_bytes = tests::common::poc_materialize(&gpu_res);

    println!("vips length: {}, gpu length: {}", vips_bytes.len(), gpu_bytes.len());
    
    let mut diff_count = 0;
    for i in (0..vips_bytes.len()).step_by(3) {
        if vips_bytes[i] != gpu_bytes[i] || vips_bytes[i+1] != gpu_bytes[i+1] || vips_bytes[i+2] != gpu_bytes[i+2] {
            if diff_count < 10 {
                let p = i / 3;
                let x = p % 240;
                let y = p / 240;
                println!("Diff at ({}, {}): vips={:?} gpu={:?}", x, y, &vips_bytes[i..i+3], &gpu_bytes[i..i+3]);
            }
            diff_count += 1;
        }
    }
    println!("Total diffs: {}", diff_count);
}
