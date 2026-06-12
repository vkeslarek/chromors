use poc::backend::vips::gobject::VipsGObject;
use poc::backend::vips::VipsBackend;
use poc::data::image::Image2D;
use poc::ffi;
mod common;

#[test]
fn test_vips_native_format() {
    common::init();
    let img = Image2D::<VipsBackend>::open("tests/fixtures/gray.jpg").unwrap();
    let mut op = VipsGObject::new(b"sobel\0").unwrap();
    
    // We hack the materialize to get the ptr
    let bytes = poc::io::Target::pull(&poc::data::image::RamImageTarget, &img, poc::node::Region { x: 0, y: 0, w: 200, h: 200, lod: poc::node::Lod(0) }).unwrap();
    
    // Actually we can just run VipsGObject directly
    let mut src_op = VipsGObject::new(b"jpegload\0").unwrap();
    src_op.set_string("filename", "tests/fixtures/gray.jpg");
    let src_handle = src_op.run().unwrap();
    
    op.set_image("in", src_handle.ptr());
    let out_handle = op.run().unwrap();
    
    unsafe {
        let fmt = ffi::vips_image_get_format(out_handle.ptr());
        println!("native vips sobel format: {}", fmt);
    }
    panic!("look");
}
