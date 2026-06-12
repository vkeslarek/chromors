use poc::backend::vips::VipsBackend;
use poc::data::image::Image2D;

mod common;

#[test]
fn test_vips_size() {
    common::init();
    let img = Image2D::<VipsBackend>::open("tests/fixtures/gray.jpg").unwrap();
    println!("vips gray w: {}, h: {}", img.width(), img.height());
    
    let sobel = img.sobel();
    println!("vips sobel w: {}, h: {}", sobel.width(), sobel.height());
    
    let bytes = common::vips_materialize_f32(&sobel);
    println!("vips sobel floats len: {}", bytes.len());
    
    panic!("look");
}
