#[test]
fn debug_sizes() {
    let _g = chromors::tests::common::vips_serial();
    let ctx = chromors::tests::common::gpu_ctx();
    let vips_img = chromors::tests::common::rgba();
    let gpu_img = chromors::tests::common::vips_to_gpu(&vips_img, &ctx);
    
    println!("GPU img format: {:?}", gpu_img.spec.format);
    let gpu_res = gpu_img.linear(vec![1.0, 1.0, 1.0, 0.5], vec![0.0]);
    println!("GPU res format: {:?}", gpu_res.spec.format);
    
    let gpu_bytes = chromors::tests::common::poc_materialize(&gpu_res);
    println!("GPU bytes len: {}", gpu_bytes.len());
}
