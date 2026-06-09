use pixors_engine::backend::vips::VipsBackend;
use pixors_engine::data::image::Image;
use pixors_engine::pixel::PixelFormat;

fn main() {
    pixors_engine::init();
    let data = vec![0u8; 100 * 100 * 6];
    let img = Image::<VipsBackend>::from_memory(&data, 100, 100, 3, PixelFormat::Rgb16).unwrap();
    println!("Original format: {:?}", img.pixel_format());

    use pixors_engine::operation::arithmetic::LinearOperation;
    let op = LinearOperation {
        a: 1.5,
        b: 0.0,
        uchar: None,
    };
    let out = img.execute(&op).unwrap();
    println!("After linear (a=1.5): {:?}", out.pixel_format());

    let op2 = LinearOperation {
        a: 1.0,
        b: 0.0,
        uchar: None,
    };
    let out2 = img.execute(&op2).unwrap();
    println!("After linear (a=1.0): {:?}", out2.pixel_format());
}
