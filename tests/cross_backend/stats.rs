use super::*;

#[test]
fn histogram_extracts_channel() {
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let (w, h) = (gpu_img.width(), gpu_img.height());
    let hist = gpu_img.histogram(256, 0);

    use poc::data::histogram::RawTarget;
    use poc::io::Target;
    use poc::work_unit::Atomic;

    let bytes = hist.pull(&RawTarget, Atomic).unwrap();
    let counts: &[u32] = bytemuck::cast_slice(&bytes);
    let total: u64 = counts.iter().take(256).map(|&c| c as u64).sum();

    println!("histogram total = {} expected = {}", total, (w * h) as u64);
    assert_eq!(
        total,
        (w * h) as u64,
        "histogram bin total must equal pixel count"
    );
}

#[test]
fn histogram_capability_matches_gpu() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);
    let (w, h) = (vips_img.width(), vips_img.height());

    // Vips: hist_find on a single band -> 256x1 uint image, sum of all bins.
    use poc::io::Target;
    use poc::work_unit::{Lod, Region};
    let vips_hist = vips_img.histogram_find(Some(0));
    let target = poc::data::image::RamImageTarget;
    let raw = vips_hist
        .pull(
            &target,
            Region {
                x: 0,
                y: 0,
                w: vips_hist.width(),
                h: vips_hist.height(),
                lod: Lod(0),
            },
        )
        .unwrap();
    let vips_counts: &[u32] = bytemuck::cast_slice(&raw);
    let vips_total: u64 = vips_counts.iter().map(|&v| v as u64).sum();
    println!(
        "vips hist_find total = {} expected = {}",
        vips_total,
        (w * h) as u64
    );

    // GPU: histogram over channel 0, 256 bins.
    let gpu_hist = gpu_img.histogram(256, 0);
    use poc::data::histogram::RawTarget;
    use poc::work_unit::Atomic;
    let gpu_bytes = gpu_hist.pull(&RawTarget, Atomic).unwrap();
    let gpu_counts: &[u32] = bytemuck::cast_slice(&gpu_bytes);
    let gpu_total: u64 = gpu_counts.iter().take(256).map(|&c| c as u64).sum();

    println!(
        "gpu histogram total = {} expected pixels = {}",
        gpu_total,
        (w * h) as u64
    );
    assert_eq!(
        vips_total,
        (w * h) as u64,
        "vips hist_find total must equal pixel count"
    );
    assert_eq!(
        gpu_total,
        (w * h) as u64,
        "GPU histogram total must equal pixel count"
    );
}

#[test]
fn histogram_gpu_capability_counts_pixels() {
    let ctx = common::gpu_ctx();
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);
    let (w, h) = (gpu_img.width(), gpu_img.height());

    use poc::data::histogram::RawTarget;
    use poc::io::Target;
    use poc::work_unit::Atomic;

    for channel in [0u32, 1, 2] {
        let hist = gpu_img.histogram(256, channel);
        let bytes = hist.pull(&RawTarget, Atomic).unwrap();
        let counts: &[u32] = bytemuck::cast_slice(&bytes);
        let total: u64 = counts.iter().take(256).map(|&c| c as u64).sum();
        println!(
            "channel {} histogram total = {} expected = {}",
            channel,
            total,
            (w * h) as u64
        );
        assert_eq!(
            total,
            (w * h) as u64,
            "channel {} histogram total must equal pixel count",
            channel
        );
    }
}

// ── Band / channel operations ─────────────────────────────────────────────────
