use super::*;
use chromors::data::histogram::GpuImageExt;

#[test]
fn histogram_extracts_channel() {
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let (w, h) = (gpu_img.width(), gpu_img.height());
    let hist = gpu_img.histogram(256, 0);

    use chromors::data::histogram::RawTarget;
    use chromors::io::Target;
    use chromors::work_unit::Atomic;

    let bytes: Vec<u8> = hist.pull(&RawTarget, Atomic).unwrap();
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
    use chromors::data::histogram::GpuImageExt;
    use chromors::io::Target;
    use chromors::work_unit::{Lod, Region};
    let vips_hist = vips_img.histogram_find(Some(0));
    let target = chromors::data::image::RamImageTarget;
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
    use chromors::data::histogram::RawTarget;
    use chromors::work_unit::Atomic;
    let gpu_bytes: Vec<u8> = gpu_hist.pull(&RawTarget, Atomic).unwrap();
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

    use chromors::data::histogram::RawTarget;
    use chromors::io::Target;
    use chromors::work_unit::Atomic;

    for channel in [0u32, 1, 2] {
        let hist = gpu_img.histogram(256, channel);
        let bytes: Vec<u8> = hist.pull(&RawTarget, Atomic).unwrap();
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

// ── HistogramFind: HistogramKind -> ImageKind bridge (GPU) ──────────────────────

#[test]
fn histogram_find_matches_vips_gray() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    // `gray.jpg` is 1-band; vips_hist_find with an explicit `band` on a
    // single-band image degenerates to 1x1 (a vips quirk), so use the
    // 3-band fixture and select band 0 — both backends agree on `256x1x1`.
    let vips_img = common::rgb();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);
    let (w, h) = (vips_img.width(), vips_img.height());

    use chromors::data::image::RamImageTarget;
    use chromors::io::Target;
    use chromors::work_unit::{Lod, Region};

    let vips_hist = vips_img.histogram_find(Some(0));
    let vips_bytes = vips_hist
        .pull(
            &RamImageTarget,
            Region {
                x: 0,
                y: 0,
                w: vips_hist.width(),
                h: vips_hist.height(),
                lod: Lod(0),
            },
        )
        .unwrap();
    let vips_counts: &[u32] = bytemuck::cast_slice(&vips_bytes);

    let gpu_hist = gpu_img.histogram_find(Some(0));
    assert_eq!(gpu_hist.width(), 256);
    assert_eq!(gpu_hist.height(), 1);
    let gpu_bytes = gpu_hist
        .pull(
            &RamImageTarget,
            Region {
                x: 0,
                y: 0,
                w: gpu_hist.width(),
                h: gpu_hist.height(),
                lod: Lod(0),
            },
        )
        .unwrap();
    let gpu_floats: &[f32] = bytemuck::cast_slice(&gpu_bytes);
    println!(
        "vips hist bytes = {} ({} u32), gpu hist bytes = {} ({} f32)",
        vips_bytes.len(),
        vips_counts.len(),
        gpu_bytes.len(),
        gpu_floats.len()
    );

    let stride = gpu_floats.len() / vips_counts.len();
    let total_vips: u64 = vips_counts.iter().map(|&v| v as u64).sum();
    assert_eq!(total_vips, (w * h) as u64);

    let total_gpu: f64 = (0..vips_counts.len())
        .map(|i| gpu_floats[i * stride] as f64)
        .sum();
    println!("histogram_find: vips total = {total_vips}, gpu total = {total_gpu}");
    assert!((total_gpu - total_vips as f64).abs() < 1.0);

    for i in 0..vips_counts.len() {
        let v = vips_counts[i] as f64;
        let g = gpu_floats[i * stride] as f64;
        assert!((v - g).abs() < 1.0, "bin {i}: vips={v} gpu={g}");
    }
}

// ── HistogramEqualize: ImageKind -> ImageKind (GPU) ─────────────────────────────

#[test]
fn histogram_equalize_matches_vips_gray() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    let vips_eq = vips_img.histogram_equalize(Some(0));
    let gpu_eq = gpu_img.histogram_equalize(Some(0));

    let vips_bytes = common::vips_materialize(&vips_eq);
    let gpu_bytes = common::poc_materialize(&gpu_eq);

    println!(
        "histogram_equalize: vips bytes = {}, gpu bytes = {}",
        vips_bytes.len(),
        gpu_bytes.len()
    );

    let rms = common::rms_u8(&vips_bytes, &gpu_bytes);
    println!("histogram_equalize RMS = {rms}");
    assert!(rms < 5.0, "RMS too high: {rms}");
}

// ── HistogramCumulative / HistogramNormalize / HistogramPlot smoke (GPU) ────────

#[test]
fn histogram_cumulative_normalize_plot_gpu_smoke() {
    let _g = common::vips_serial();
    let ctx = common::gpu_ctx();
    let vips_img = common::gray();
    let gpu_img = common::vips_to_gpu(&vips_img, &ctx);

    use chromors::data::image::RamImageTarget;
    use chromors::io::Target;
    use chromors::work_unit::{Lod, Region};

    let hist = gpu_img.histogram_find(Some(0));

    let cum = hist.histogram_cumulative();
    assert_eq!(cum.width(), 256);
    assert_eq!(cum.height(), 1);
    let cum_bytes: Vec<u8> = cum
        .pull(
            &RamImageTarget,
            Region {
                x: 0,
                y: 0,
                w: cum.width(),
                h: cum.height(),
                lod: Lod(0),
            },
        )
        .unwrap();
    let cum_floats: &[f32] = bytemuck::cast_slice(&cum_bytes);
    let stride = cum_floats.len() / 256;
    let last = cum_floats[255 * stride] as f64;
    let total = ((gpu_img.width() * gpu_img.height()) as u64) as f64;
    println!("histogram_cumulative: last = {last}, total = {total}");
    assert!((last - total).abs() < 1.0);
    for i in 1..256 {
        assert!(cum_floats[i * stride] + 1e-3 >= cum_floats[(i - 1) * stride]);
    }

    let norm = hist.histogram_normalize();
    assert_eq!(norm.width(), 256);
    assert_eq!(norm.height(), 1);
    let norm_bytes: Vec<u8> = norm
        .pull(
            &RamImageTarget,
            Region {
                x: 0,
                y: 0,
                w: norm.width(),
                h: norm.height(),
                lod: Lod(0),
            },
        )
        .unwrap();
    let norm_floats: &[f32] = bytemuck::cast_slice(&norm_bytes);
    let max = (0..256)
        .map(|i| norm_floats[i * stride])
        .fold(0.0f32, f32::max);
    println!("histogram_normalize: max = {max}");
    assert!((max - 255.0).abs() < 1.0);

    let plot = hist.histogram_plot();
    assert_eq!(plot.width(), 256);
    assert_eq!(plot.height(), 256);
    let plot_bytes: Vec<u8> = plot
        .pull(
            &RamImageTarget,
            Region {
                x: 0,
                y: 0,
                w: plot.width(),
                h: plot.height(),
                lod: Lod(0),
            },
        )
        .unwrap();
    assert!(!plot_bytes.is_empty());
}

// ── Band / channel operations ─────────────────────────────────────────────────

