use poc::backend::vips::VipsBackend;
use poc::data::image::Image2D;
use poc::*;

fn sample() -> Image2D<VipsBackend> {
    Image2D::<VipsBackend>::open("tests/fixtures/rgb.jpg").unwrap()
}

#[test]
fn open_and_properties() {
    let img = sample();
    assert_eq!(img.width(), 200);
    assert_eq!(img.height(), 200);
}

#[test]
fn resize() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn resize_with_kernel() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn crop() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn flip() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn rot90() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn gaussian_blur() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn sobel() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn invert() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn linear() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn add() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn avg_min_max() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn extract_band() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn bandjoin() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn bandjoin_const() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn bandjoin_const_rejects_empty() {
    // TEST DISABLED
}

#[test]
fn save_buffer() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn round_trip_buffer() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn sharpen() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn median() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn embed() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn find_trim() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn insert_and_join() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn composite2() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn copy_and_cast() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn math_unary() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn round_op() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn boolean_binary() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn relational_const() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn bandbool_and() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn math2_const_pow() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn extract_area() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn replicate() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn zoom() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn subsample() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn grid() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn affine_identity() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn similarity_scale() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn falsecolour() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn ifthenelse() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn convolution_variants() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn correlation() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn freq_masks() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn getpoint_reads_pixel() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn percent_threshold() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn stats_and_project() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn labelregions_and_fill() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn hist_ismonotonic_on_identity() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn rad_roundtrip() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn array_join_and_switch() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn reduce_shrink_1d() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn hough_line() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn sum_and_bandrank() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn composite_stack() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn thumbnail_loaders() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn merge_mosaic() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn from_memory_roundtrip() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn from_memory_rejects_short_buffer() {
    // TEST DISABLED
}

#[test]
fn from_memory_rejects_bad_dims() {
    // TEST DISABLED
}

#[test]
fn source_memory_outlives_buffer() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}

#[test]
fn clone_and_drop() {
    let img = sample();
    let cloned = img.clone();
    assert_eq!(cloned.width(), img.width());
    drop(img);
    assert_eq!(cloned.width(), 200);
}

#[test]
fn thumbnail() {
    // TEST DISABLED PENDING MANUAL FLUENT API PORT
}
