use crate::common::rgb;
use pixors_engine::data::histogram::HistogramResult;
use pixors_engine::operation::custom_ops::HistogramSink;
use pixors_engine::target::HistogramTarget;

fn pull_histogram(
    hist: &pixors_engine::data::histogram::Histogram<pixors_engine::backend::vips::VipsBackend>,
) -> pixors_engine::target::MaterializedHistogram<pixors_engine::backend::vips::VipsBackend> {
    HistogramTarget::new(hist.clone()).pull().unwrap()
}

#[test]
fn histogram_counts_every_pixel() {
    let img = rgb();
    let (w, h) = (img.width() as u64, img.height() as u64);
    let bands = img.bands() as u64;

    let hist = img.histogram().unwrap();
    let mat = pull_histogram(&hist);
    let result = HistogramResult::from_bytes(&mat.buffer);
    assert_eq!(result.total_pixels, w * h * bands);
}

#[test]
fn histogram_matches_sink() {
    let img = rgb();
    let bands = img.bands() as usize;

    let hist = img.histogram().unwrap();
    let mat = pull_histogram(&hist);
    let cap_result = HistogramResult::from_bytes(&mat.buffer);

    let sink_hist = img.sink(HistogramSink).unwrap();
    assert_eq!(sink_hist.bins.len(), bands);

    let sink_bytes: Vec<u8> = sink_hist
        .bins
        .iter()
        .flat_map(|band| band.iter().copied())
        .flat_map(|x| x.to_le_bytes().to_vec())
        .collect();
    let sink_result = HistogramResult::from_bytes(&sink_bytes);

    assert_eq!(cap_result.total_pixels, sink_result.total_pixels);
    assert_eq!(mat.bins as usize * bands * 4, mat.buffer.len());
}

#[test]
fn histogram_buffer_length() {
    let img = rgb();
    let bands = img.bands() as usize;

    let hist = img.histogram().unwrap();
    let mat = pull_histogram(&hist);
    let bins = mat.bins as usize;
    assert_eq!(bins, 256, "uchar input should produce 256 bins");
    assert_eq!(mat.buffer.len(), bins * bands * 4);
}

#[test]
fn histogram_chainable() {
    let img = rgb();
    let hist = img.histogram().unwrap();
    let mat = pull_histogram(&hist);
    let result = HistogramResult::from_bytes(&mat.buffer);
    assert!(result.total_pixels > 0);
}
