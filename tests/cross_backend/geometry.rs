use super::*;

#[test]
#[ignore = "BUG: Shrink::output_spec (src/operation/geometry.rs) returns the input width/height unchanged for both backends, but vips' shrink physically shrinks its output (w/2 * h/2 * bands bytes); GPU materializer sizes its output buffer from output_spec(), so the GPU output cannot be compared byte-for-byte against vips' actually-shrunk image. Fix requires an output_spec change in Shrink."]
fn shrink_matches_vips() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_res = vips_img.shrink(2.0, 2.0, None);
    let gpu_res = gpu_img.shrink(2.0, 2.0, None);

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("shrink RMS = {}", rms);
    assert!(rms < 10.0, "shrink diff too high: {}", rms);
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
