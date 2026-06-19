use super::*;
use chromors::operation::mosaicing::Merge;

// GPU `merge` assumes both inputs are fully opaque rectangles: placement of
// `ref`/`sec` on the output canvas and the non-overlap regions match vips
// exactly, but the blend *inside* the overlap is a cosine across the whole
// overlap (clamped to `max_blend`/default 10) rather than vips' per-scanline
// search for the first/last non-transparent pixel -- so the seam itself can
// differ in exact position/shape.
#[test]
fn merge_horizontal_matches_vips_outside_overlap() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let reference = common::rgb();
    let secondary = common::rgb();

    let dx = reference.width() / 2;
    let dy = 0;

    let vips_res = reference.push(Merge {
        input: reference.as_input(),
        secondary: secondary.as_input(),
        direction: Direction::Horizontal,
        dx,
        dy,
        max_blend: None,
    });

    let gpu_reference = common::vips_to_gpu(&reference, &ctx);
    let gpu_secondary = common::vips_to_gpu(&secondary, &ctx);
    let gpu_res = gpu_reference.push(Merge {
        input: gpu_reference.as_input(),
        secondary: gpu_secondary.as_input(),
        direction: Direction::Horizontal,
        dx,
        dy,
        max_blend: None,
    });

    assert_eq!(gpu_res.width(), vips_res.width());
    assert_eq!(gpu_res.height(), vips_res.height());

    let ow = vips_res.width();
    let oh = vips_res.height();
    let bands = 3usize;

    let vips_bytes = common::vips_materialize(&vips_res);
    let gpu_bytes = common::poc_materialize(&gpu_res);

    // `sec` occupies x in [0, sec_w); `ref` occupies x in [dx, dx + ref_w).
    // The overlap [dx, sec_w) is where blending happens; everywhere else is
    // a direct copy from one source image and should match vips exactly.
    let overlap_left = dx;
    let overlap_right = secondary.width();

    let mut outside_bytes = 0usize;
    let mut outside_rms_acc = 0f64;
    for y in 0..oh {
        for x in 0..ow {
            if x >= overlap_left && x < overlap_right {
                continue;
            }
            let idx = ((y * ow + x) as usize) * bands;
            for b in 0..bands {
                let v = vips_bytes[idx + b];
                let g = gpu_bytes[idx + b];
                outside_rms_acc += ((v as f64) - (g as f64)).powi(2);
                outside_bytes += 1;
            }
        }
    }
    let outside_rms = (outside_rms_acc / outside_bytes as f64).sqrt();
    println!("merge_horizontal outside-overlap RMS = {}", outside_rms);
    assert_eq!(
        outside_rms, 0.0,
        "non-overlap region should match vips exactly"
    );
}

#[test]
fn merge_vertical_with_max_blend_runs_and_is_sane() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let reference = common::rgb();
    let secondary = common::rgb();

    let dx = 0;
    let dy = reference.height() / 2;

    let gpu_reference = common::vips_to_gpu(&reference, &ctx);
    let gpu_secondary = common::vips_to_gpu(&secondary, &ctx);
    let gpu_res = gpu_reference.push(Merge {
        input: gpu_reference.as_input(),
        secondary: gpu_secondary.as_input(),
        direction: Direction::Vertical,
        dx,
        dy,
        max_blend: Some(10),
    });

    assert_eq!(gpu_res.width(), reference.width());
    assert_eq!(gpu_res.height(), dy + secondary.height());

    let gpu_bytes = common::poc_materialize(&gpu_res);
    assert!(!gpu_bytes.is_empty());
}
