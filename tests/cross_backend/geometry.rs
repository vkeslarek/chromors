use super::*;

#[test]
fn shrink_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.shrink(2.0, 2.0, None);
    let gpu_res = gpu_img.shrink(2.0, 2.0, None);

    assert_eq!(gpu_res.width(), vips_res.width());
    assert_eq!(gpu_res.height(), vips_res.height());

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("shrink RMS = {}", rms);
    assert!(rms < 10.0, "shrink diff too high: {}", rms);
}

#[test]
fn with_lod_matches_vips_shrink() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let lod = chromors::work_unit::Lod(2); // scale_factor = 4

    let vips_res = vips_img.shrink(4.0, 4.0, None);
    let gpu_res = gpu_img.with_lod(lod);

    assert_eq!(gpu_res.width(), vips_res.width());
    assert_eq!(gpu_res.height(), vips_res.height());
    assert_eq!(gpu_res.width(), vips_img.width() / 4);
    assert_eq!(gpu_res.height(), vips_img.height() / 4);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("with_lod RMS = {}", rms);
    assert!(rms < 10.0, "with_lod diff too high: {}", rms);
}

#[test]
fn with_lod_zero_is_identity() {
    let ctx = common::gpu_ctx();
    let _g = common::vips_serial();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let lod0 = gpu_img.with_lod(chromors::work_unit::Lod(0));
    assert_eq!(lod0.width(), gpu_img.width());
    assert_eq!(lod0.height(), gpu_img.height());
}

#[test]
fn crop_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.crop(10, 10, 50, 50);
    let gpu_res = gpu_img.crop(10, 10, 50, 50);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("crop RMS = {}", rms);
    assert!(rms < 5.0, "crop diff too high: {}", rms);
}

#[test]
fn embed_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.embed(
        20,
        20,
        240,
        240,
        Some(Extend::Background),
        Some([0.0, 0.0, 0.0]),
    );
    let gpu_res = gpu_img.embed(
        20,
        20,
        240,
        240,
        Some(Extend::Background),
        Some([0.0, 0.0, 0.0]),
    );

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let mut diff_count = 0;
    for i in (0..vips_bytes.len()).step_by(3) {
        if vips_bytes[i] != gpu_bytes[i]
            || vips_bytes[i + 1] != gpu_bytes[i + 1]
            || vips_bytes[i + 2] != gpu_bytes[i + 2]
        {
            if diff_count < 10 {
                let p = i / 3;
                let x = p % 240;
                let y = p / 240;
                println!(
                    "Diff at ({}, {}): vips={:?} gpu={:?}",
                    x,
                    y,
                    &vips_bytes[i..i + 3],
                    &gpu_bytes[i..i + 3]
                );
            }
            diff_count += 1;
        }
    }
    println!("Total diffs: {}", diff_count);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("embed RMS = {}", rms);
    assert!(rms < 5.0, "embed diff too high: {}", rms);
}

#[test]
fn flip_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.flip(Direction::Horizontal);
    let gpu_res = gpu_img.flip(Direction::Horizontal);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("flip RMS = {}", rms);
    assert!(rms < 1.0, "flip diff too high: {}", rms);
}

#[test]
fn rot90_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.rot90(Angle::D90);
    let gpu_res = gpu_img.rot90(Angle::D90);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("rot90 RMS = {}", rms);
    assert!(rms < 1.0, "rot90 diff too high: {}", rms);
}

#[test]
#[ignore = "BUG: Rot45 D45 output_spec reports an expanded canvas matching the GPU lower's actual buffer (282x282), but vips' rot45 is a same-size pixel permutation (199x199) — spec/semantics mismatch between backends. Separately, GPU Rot45 on odd Rgb8 dims hits a wgpu storage-buffer 4-byte-alignment validation error (w*h*3 not aligned to 4)."]
fn rot45_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb().crop(0, 0, 199, 199);
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.rot45(Angle45::D45);
    let gpu_res = gpu_img.rot45(Angle45::D45);

    let vw = vips_res.width() as usize;
    let vh = vips_res.height() as usize;
    let gw = gpu_res.width() as usize;
    let gh = gpu_res.height() as usize;
    assert_eq!((vw, vh), (gw, gh), "rot45 output dims mismatch");

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8_interior(&vips_bytes, &gpu_bytes, vw, vh, 3, 20);
    println!("rot45 RMS (interior) = {}", rms);
    assert!(rms < 20.0, "rot45 diff too high: {}", rms);
}

#[test]
#[ignore = "BUG: Rotate::output_spec keeps the input WxH (200x200), but vips' rotate naturally expands the canvas to fit the rotated image (~245x245 @ 15deg) while the GPU lower stays 200x200 — dims never match."]
fn rotate_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.rotate(15.0, Some([0.0, 0.0, 0.0]), None, None, None, None);
    let gpu_res = gpu_img.rotate(15.0, Some([0.0, 0.0, 0.0]), None, None, None, None);

    let vw = vips_res.width() as usize;
    let vh = vips_res.height() as usize;
    let gw = gpu_res.width() as usize;
    let gh = gpu_res.height() as usize;
    assert_eq!((vw, vh), (gw, gh), "rotate output dims mismatch");

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8_interior(&vips_bytes, &gpu_bytes, vw, vh, 3, 20);
    println!("rotate RMS (interior) = {}", rms);
    assert!(rms < 20.0, "rotate diff too high: {}", rms);
}

#[test]
fn resize_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.resize(0.5, None, None, None);
    let gpu_res = gpu_img.resize(0.5, None, None, None);

    let vw = vips_res.width() as usize;
    let vh = vips_res.height() as usize;
    let gw = gpu_res.width() as usize;
    let gh = gpu_res.height() as usize;
    assert_eq!((vw, vh), (gw, gh), "resize output dims mismatch");

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8_interior(&vips_bytes, &gpu_bytes, vw, vh, 3, 5);
    println!("resize RMS (interior) = {}", rms);
    assert!(rms < 15.0, "resize diff too high: {}", rms);
}

#[test]
fn zoom_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.zoom(2, 2);
    let gpu_res = gpu_img.zoom(2, 2);

    let vw = vips_res.width() as usize;
    let vh = vips_res.height() as usize;
    let gw = gpu_res.width() as usize;
    let gh = gpu_res.height() as usize;
    assert_eq!((vw, vh), (gw, gh), "zoom output dims mismatch");

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("zoom RMS = {}", rms);
    assert!(rms < 1.0, "zoom diff too high: {}", rms);
}

#[test]
fn replicate_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.replicate(2, 2);
    let gpu_res = gpu_img.replicate(2, 2);

    let vw = vips_res.width() as usize;
    let vh = vips_res.height() as usize;
    let gw = gpu_res.width() as usize;
    let gh = gpu_res.height() as usize;
    assert_eq!((vw, vh), (gw, gh), "replicate output dims mismatch");

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("replicate RMS = {}", rms);
    assert!(rms < 1.0, "replicate diff too high: {}", rms);
}

#[test]
fn subsample_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.subsample(2, 2, Some(true));
    let gpu_res = gpu_img.subsample(2, 2, Some(true));

    let vw = vips_res.width() as usize;
    let vh = vips_res.height() as usize;
    let gw = gpu_res.width() as usize;
    let gh = gpu_res.height() as usize;
    assert_eq!((vw, vh), (gw, gh), "subsample output dims mismatch");

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("subsample RMS = {}", rms);
    assert!(rms < 1.0, "subsample diff too high: {}", rms);
}

#[test]
fn reduce_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.reduce(2.0, 2.0, None, None);
    let gpu_res = gpu_img.reduce(2.0, 2.0, None, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("reduce RMS = {}", rms);
    assert!(rms < 10.0, "reduce diff too high: {}", rms);
}

#[test]
fn reduce_horizontal_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.reduce_horizontal(2.0, None, None);
    let gpu_res = gpu_img.reduce_horizontal(2.0, None, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("reduce_horizontal RMS = {}", rms);
    assert!(rms < 10.0, "reduce_horizontal diff too high: {}", rms);
}

#[test]
fn reduce_vertical_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.reduce_vertical(2.0, None, None);
    let gpu_res = gpu_img.reduce_vertical(2.0, None, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("reduce_vertical RMS = {}", rms);
    assert!(rms < 10.0, "reduce_vertical diff too high: {}", rms);
}

#[test]
fn shrink_horizontal_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.shrink_horizontal(2, None);
    let gpu_res = gpu_img.shrink_horizontal(2, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("shrink_horizontal RMS = {}", rms);
    assert!(rms < 10.0, "shrink_horizontal diff too high: {}", rms);
}

#[test]
fn shrink_vertical_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.shrink_vertical(2, None);
    let gpu_res = gpu_img.shrink_vertical(2, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("shrink_vertical RMS = {}", rms);
    assert!(rms < 10.0, "shrink_vertical diff too high: {}", rms);
}

#[test]
fn gravity_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let new_w = vips_img.width() + 40;
    let new_h = vips_img.height() + 40;

    let vips_res = vips_img.gravity(
        CompassDirection::Centre,
        new_w,
        new_h,
        Some(Extend::Background),
        Some([0.0, 0.0, 0.0]),
    );
    let gpu_res = gpu_img.gravity(
        CompassDirection::Centre,
        new_w,
        new_h,
        Some(Extend::Background),
        Some([0.0, 0.0, 0.0]),
    );

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("gravity RMS = {}", rms);
    assert!(rms < 10.0, "gravity diff too high: {}", rms);
}

#[test]
fn thumbnail_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.thumbnail(
        100, None, None, None, None, None, None, None, None, None, None,
    );
    let gpu_res = gpu_img.thumbnail(
        100, None, None, None, None, None, None, None, None, None, None,
    );

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("thumbnail RMS = {}", rms);
    assert!(rms < 15.0, "thumbnail diff too high: {}", rms);
}

#[test]
fn extract_area_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.extract_area(5, 5, 40, 40);
    let gpu_res = gpu_img.extract_area(5, 5, 40, 40);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("extract_area RMS = {}", rms);
    assert!(rms < 5.0, "extract_area diff too high: {}", rms);
}

// ── Data-driven ops ──────────────────────────────────────────────────────────

#[test]
fn grid_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    // 200x200 input -> 4 strips of 50px, re-tiled into a 2x2 grid -> 400x100.
    let vips_res = vips_img.grid(50, 2, 2);
    let gpu_res = gpu_img.grid(50, 2, 2);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("grid RMS = {}", rms);
    assert!(rms < 1.0, "grid diff too high: {}", rms);
}

#[test]
fn with_lod_tile_offset_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let lod = chromors::work_unit::Lod(2); // scale_factor = 4, 200x200 -> 50x50

    // Reference: full vips shrink, then crop the right portion (x=24..50).
    let vips_shrunk = vips_img.shrink(4.0, 4.0, None);
    let vips_ref = vips_shrunk.crop(24, 0, 26, 50);

    let gpu_res = gpu_img.with_lod(lod);
    use chromors::io::Target;
    let target = chromors::data::image::RamImageTarget;
    let gpu_bytes = gpu_res
        .pull(
            &target,
            chromors::work_unit::Region {
                x: 24,
                y: 0,
                w: 26,
                h: 50,
                lod: chromors::work_unit::Lod(0),
            },
        )
        .unwrap();

    let vips_bytes = common::vips_materialize(&vips_ref);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("with_lod tile-offset RMS = {}", rms);
    assert!(rms < 5.0, "tile-offset diff too high: {}", rms);
}

/// 25x50 RGB8 = 3750 bytes, not a multiple of 4 — regression test for the
/// wgpu "Effective buffer binding size ... expected to align to 4" panic
/// that non-4-aligned output buffers used to trigger (e.g. odd-sized atlas
/// tiles at higher mip levels).
#[test]
fn with_lod_unaligned_region_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let lod = chromors::work_unit::Lod(2); // scale_factor = 4, 200x200 -> 50x50

    let vips_shrunk = vips_img.shrink(4.0, 4.0, None);
    let vips_ref = vips_shrunk.crop(25, 0, 25, 50);

    let gpu_res = gpu_img.with_lod(lod);
    use chromors::io::Target;
    let target = chromors::data::image::RamImageTarget;
    let gpu_bytes = gpu_res
        .pull(
            &target,
            chromors::work_unit::Region {
                x: 25,
                y: 0,
                w: 25,
                h: 50,
                lod: chromors::work_unit::Lod(0),
            },
        )
        .unwrap();

    let vips_bytes = common::vips_materialize(&vips_ref);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("with_lod unaligned-region RMS = {}", rms);
    assert!(rms < 5.0, "unaligned-region diff too high: {}", rms);
}

// ── Smartcrop ───────────────────────────────────────────────────────────────

#[test]
fn smartcrop_centre_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let new_w = vips_img.width() - 40;
    let new_h = vips_img.height() - 40;

    let vips_res = vips_img.smartcrop(new_w, new_h, Some(Interesting::Centre));
    let gpu_res = gpu_img.smartcrop(new_w, new_h, Some(Interesting::Centre));

    assert_eq!(gpu_res.width(), vips_res.width());
    assert_eq!(gpu_res.height(), vips_res.height());

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("smartcrop_centre RMS = {}", rms);
    assert!(rms < 5.0, "smartcrop_centre diff too high: {}", rms);
}

#[test]
fn smartcrop_attention_runs_and_is_sane() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let new_w = vips_img.width() - 40;
    let new_h = vips_img.height() - 40;

    // GPU smartcrop score is an approximation of vips' ENTROPY/ATTENTION
    // heuristic — not bit-exact. Just check dims + that it ran cleanly.
    let gpu_res = gpu_img.smartcrop(new_w, new_h, Some(Interesting::Attention));

    assert_eq!(gpu_res.width(), new_w);
    assert_eq!(gpu_res.height(), new_h);

    let gpu_bytes = common::poc_materialize(&gpu_res);
    assert!(!gpu_bytes.is_empty());
    assert_eq!(gpu_bytes.len() % (new_w * new_h) as usize, 0);
}
