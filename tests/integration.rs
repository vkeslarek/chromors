use pixors_engine::backend::vips::VipsBackend;
use pixors_engine::data::image::Image;
use pixors_engine::*;

fn init() {
    std::sync::Once::new().call_once(pixors_engine::init);
}

fn sample() -> Image<VipsBackend> {
    init();
    Image::<VipsBackend>::open("tests/fixtures/rgb.jpg").unwrap()
}

#[test]
fn open_and_properties() {
    let img = sample();
    assert_eq!(img.width(), 200);
    assert_eq!(img.height(), 200);
    assert_eq!(img.bands(), 3);
}

#[test]
fn resize() {
    let half = sample()
        .execute(&ResizeOperation {
            scale: 0.5,
            kernel: None,
            vertical_scale: None,
            gap: None,
        })
        .unwrap();
    assert_eq!(half.width(), 100);
}

#[test]
fn resize_with_kernel() {
    let half = sample()
        .execute(&ResizeOperation {
            scale: 0.5,
            kernel: Some(Kernel::Lanczos3),
            vertical_scale: None,
            gap: None,
        })
        .unwrap();
    assert_eq!(half.width(), 100);
}

#[test]
fn crop() {
    let cropped = sample()
        .execute(&CropOperation {
            left: 10,
            top: 10,
            width: 50,
            height: 50,
        })
        .unwrap();
    assert_eq!(cropped.width(), 50);
}

#[test]
fn flip() {
    let flipped = sample()
        .execute(&FlipOperation {
            direction: Direction::Horizontal,
        })
        .unwrap();
    assert_eq!(flipped.width(), 200);
}

#[test]
fn rot90() {
    let r = sample()
        .execute(&Rot90Operation { angle: Angle::D90 })
        .unwrap();
    assert_eq!(r.width(), 200);
}

#[test]
fn gaussian_blur() {
    let blurred = sample()
        .execute(&GaussianBlurOperation {
            sigma: 5.0,
            minimum_amplitude: None,
            precision: None,
        })
        .unwrap();
    assert_eq!(blurred.width(), 200);
}

#[test]
fn sobel() {
    let edges = sample().execute(&SobelOperation).unwrap();
    assert_eq!(edges.width(), 200);
}

#[test]
fn invert() {
    let inv = sample().execute(&InvertOperation).unwrap();
    assert_eq!(inv.width(), 200);
}

#[test]
fn linear() {
    let bright = sample()
        .execute(&LinearOperation {
            a: 2.0,
            b: 0.0,
            uchar: None,
        })
        .unwrap();
    assert_eq!(bright.width(), 200);
}

#[test]
fn add() {
    let img = sample();
    let result = img.execute(&AddOperation { right: img.clone() }).unwrap();
    assert_eq!(result.width(), 200);
}

#[test]
fn avg_min_max() {
    let img = sample();
    let avg: f64 = img.execute(&AverageOperation).unwrap();
    let min: f64 = img
        .execute(&MinimumOperation {
            size: None,
            x: None,
            y: None,
        })
        .unwrap();
    let max: f64 = img
        .execute(&MaximumOperation {
            size: None,
            x: None,
            y: None,
        })
        .unwrap();
    assert!(min <= avg && avg <= max);
}

#[test]
fn extract_band() {
    let r = sample()
        .execute(&ExtractBandOperation {
            band: 0,
            count: None,
        })
        .unwrap();
    assert_eq!(r.bands(), 1);
}

#[test]
fn bandjoin() {
    let r = sample()
        .execute(&ExtractBandOperation {
            band: 0,
            count: None,
        })
        .unwrap();
    let joined = r.bandjoin(&r).unwrap();
    assert_eq!(joined.bands(), 2);
}

#[test]
fn bandjoin_const() {
    let joined = sample().bandjoin_const(&[0.5]).unwrap();
    assert_eq!(joined.bands(), 4);
}

#[test]
fn bandjoin_const_rejects_empty() {
    assert!(sample().bandjoin_const(&[]).is_err());
}

#[test]
fn save_buffer() {
    let png = sample().write_to_buffer(".png").unwrap();
    assert!(!png.is_empty());
}

#[test]
fn round_trip_buffer() {
    let png = sample().write_to_buffer(".png").unwrap();
    let decoded = Image::<VipsBackend>::from_buffer(&png).unwrap();
    assert_eq!(decoded.width(), 200);
}

#[test]
fn sharpen() {
    let s = sample()
        .execute(&SharpenOperation {
            sigma: None,
            flat: None,
            jagged: None,
            edge: None,
            smooth: None,
            maximum: None,
        })
        .unwrap();
    assert_eq!(s.width(), 200);
}

#[test]
fn median() {
    let m = sample().execute(&MedianOperation { size: 3 }).unwrap();
    assert_eq!(m.width(), 200);
}

#[test]
fn embed() {
    let e = sample()
        .execute(&EmbedOperation {
            x: 10,
            y: 10,
            width: 300,
            height: 300,
            extend: None,
            background: None,
        })
        .unwrap();
    assert_eq!(e.width(), 300);
}

#[test]
fn find_trim() {
    let e = sample()
        .execute(&EmbedOperation {
            x: 10,
            y: 10,
            width: 300,
            height: 300,
            extend: None,
            background: None,
        })
        .unwrap();
    let bounds = e
        .execute(&FindTrimOperation {
            background: None,
            threshold: None,
            line_art: None,
        })
        .unwrap();
    assert!(bounds.width <= 300 && bounds.height <= 300);
}

#[test]
fn insert_and_join() {
    let a = sample();
    let small = a
        .execute(&ResizeOperation {
            scale: 0.25,
            kernel: None,
            vertical_scale: None,
            gap: None,
        })
        .unwrap();
    let ins = a
        .execute(&InsertOperation {
            sub: small.clone(),
            x: 50,
            y: 50,
            expand: None,
            background: None,
        })
        .unwrap();
    assert_eq!(ins.width(), 200);
    let joined = a
        .execute(&JoinOperation {
            right: a.clone(),
            direction: Direction::Horizontal,
            expand: None,
            shim: None,
            background: None,
            align: None,
        })
        .unwrap();
    assert_eq!(joined.width(), 400);
}

#[test]
fn composite2() {
    let a = sample();
    let half = a
        .execute(&ResizeOperation {
            scale: 0.5,
            kernel: None,
            vertical_scale: None,
            gap: None,
        })
        .unwrap();
    let c = a
        .execute(&Composite2Operation {
            overlay: half,
            mode: BlendMode::Over,
            x: None,
            y: None,
            compositing_space: None,
            premultiplied: None,
        })
        .unwrap();
    assert_eq!(c.width(), 200);
}

#[test]
fn copy_and_cast() {
    let img = sample();
    let cp = img
        .execute(&CopyOperation {
            width: None,
            height: None,
            bands: None,
            format: None,
            interpretation: None,
            horizontal_resolution: None,
            vertical_resolution: None,
            offset_x: None,
            offset_y: None,
        })
        .unwrap();
    assert_eq!(cp.width(), 200);

    let casted = img
        .execute(&CastOperation {
            format: PixelFormat::RgbF32,
            shift: None,
        })
        .unwrap();
    assert!(matches!(casted.pixel_format(), PixelFormat::RgbF32));
}

#[test]
fn math_unary() {
    let out = sample()
        .execute(&MathOperation {
            math: OperationMath::Sin,
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn round_op() {
    let out = sample()
        .execute(&RoundOperation {
            round: OperationRound::Floor,
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn boolean_binary() {
    let img = sample();
    let out = img
        .execute(&BooleanOperation {
            right: img.clone(),
            boolean: OperationBoolean::And,
        })
        .unwrap();
    assert_eq!(out.bands(), 3);
}

#[test]
fn relational_const() {
    let out = sample()
        .execute(&RelationalConstOperation {
            relational: OperationRelational::More,
            constants: vec![128.0],
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn bandbool_and() {
    let out = sample()
        .execute(&BandboolOperation {
            boolean: OperationBoolean::And,
            bands: 3,
        })
        .unwrap();
    assert_eq!(out.bands(), 1);
}

#[test]
fn math2_const_pow() {
    let out = sample()
        .execute(&Math2ConstOperation {
            math2: OperationMath2::Pow,
            constants: vec![2.0],
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn extract_area() {
    let out = sample()
        .execute(&ExtractAreaOperation {
            left: 10,
            top: 10,
            width: 50,
            height: 50,
        })
        .unwrap();
    assert_eq!(out.width(), 50);
}

#[test]
fn replicate() {
    let out = sample()
        .execute(&ReplicateOperation { across: 2, down: 3 })
        .unwrap();
    assert_eq!(out.width(), 400);
    assert_eq!(out.height(), 600);
}

#[test]
fn zoom() {
    let out = sample()
        .execute(&ZoomOperation {
            horizontal: 2,
            vertical: 2,
        })
        .unwrap();
    assert_eq!(out.width(), 400);
}

#[test]
fn subsample() {
    let out = sample()
        .execute(&SubsampleOperation {
            horizontal: 2,
            vertical: 2,
            point: None,
        })
        .unwrap();
    assert_eq!(out.width(), 100);
}

#[test]
fn grid() {
    let out = sample()
        .execute(&GridOperation {
            tile_height: 100,
            across: 2,
            down: 1,
        })
        .unwrap();
    assert_eq!(out.width(), 400);
}

#[test]
fn affine_identity() {
    let out = sample()
        .execute(&AffineOperation {
            matrix: vec![1.0, 0.0, 0.0, 1.0],
            interpolate: None,
            output_area: None,
            offset_input_x: None,
            offset_input_y: None,
            offset_output_x: None,
            offset_output_y: None,
            background: None,
            premultiplied: None,
            extend: None,
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn similarity_scale() {
    let out = sample()
        .execute(&SimilarityOperation {
            scale: Some(0.5),
            angle: None,
            interpolate: None,
            background: None,
            offset_input_x: None,
            offset_input_y: None,
            offset_output_x: None,
            offset_output_y: None,
        })
        .unwrap();
    assert_eq!(out.width(), 100);
}

#[test]
fn falsecolour() {
    let gray = sample()
        .execute(&ExtractBandOperation {
            band: 0,
            count: None,
        })
        .unwrap();
    let out = gray.execute(&FalsecolourOperation).unwrap();
    assert_eq!(out.bands(), 3);
}

#[test]
fn ifthenelse() {
    let a = sample();
    let mask = a
        .execute(&RelationalConstOperation {
            relational: OperationRelational::More,
            constants: vec![128.0],
        })
        .unwrap();
    let out = mask
        .execute(&IfthenelseOperation {
            if_true: &a,
            if_false: &a,
            blend: None,
        })
        .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn convolution_variants() {
    let img = sample();
    let mask = Image::<VipsBackend>::generate(&GaussMat {
        sigma: 1.0,
        minimum_amplitude: 0.2,
    })
    .unwrap();
    assert_eq!(
        img.execute(&ConvfOperation { mask: mask.clone() })
            .unwrap()
            .width(),
        200
    );
    assert_eq!(
        img.execute(&ConviOperation { mask: mask.clone() })
            .unwrap()
            .width(),
        200
    );

    // convsep needs a separable (1xN) mask.
    let row = mask
        .execute(&CropOperation {
            left: 0,
            top: 0,
            width: mask.width(),
            height: 1,
        })
        .unwrap();
    let sep = img
        .execute(&ConvsepOperation {
            mask: row.clone(),
            precision: Some(Precision::Float),
            layers: None,
            cluster: None,
        })
        .unwrap();
    assert_eq!(sep.width(), 200);
}

#[test]
fn correlation() {
    let img = sample();
    let reference = img
        .execute(&CropOperation {
            left: 0,
            top: 0,
            width: 20,
            height: 20,
        })
        .unwrap();
    let out = img
        .execute(&FastcorOperation {
            reference: reference.clone(),
        })
        .unwrap();
    assert!(out.width() > 0);
}

#[test]
fn freq_masks() {
    let ideal = Image::<VipsBackend>::generate(&MaskIdeal {
        width: 64,
        height: 64,
        frequency_cutoff: 0.5,
        uchar: None,
        nodc: None,
        reject: None,
        optical: None,
    })
    .unwrap();
    assert_eq!(ideal.width(), 64);
    let bw = Image::<VipsBackend>::generate(&MaskButterworth {
        width: 32,
        height: 32,
        order: 2.0,
        frequency_cutoff: 0.5,
        amplitude_cutoff: 0.5,
        uchar: None,
        nodc: None,
        reject: None,
        optical: None,
    })
    .unwrap();
    assert_eq!(bw.width(), 32);
}

#[test]
fn getpoint_reads_pixel() {
    let values = sample()
        .execute(&GetpointOperation {
            x: 0,
            y: 0,
            unpack_complex: None,
        })
        .unwrap();
    assert_eq!(values.0.len(), 3);
}

#[test]
fn percent_threshold() {
    let t = sample()
        .execute(&PercentOperation { percent: 50.0 })
        .unwrap();
    assert!(t.0 >= 0 && t.0 <= 255);
}

#[test]
fn stats_and_project() {
    let img = sample();
    let stats = img.execute(&StatsOperation).unwrap();
    assert!(stats.width() > 0);
    let proj = img.execute(&ProjectOperation).unwrap();
    assert_eq!(proj.columns.width(), 200);
    assert_eq!(proj.rows.height(), 200);
}

#[test]
fn labelregions_and_fill() {
    let img = sample();
    let labels = img.execute(&LabelregionsOperation).unwrap();
    assert!(labels.segments >= 0);
    assert_eq!(labels.mask.width(), 200);
    let filled = img.execute(&FillNearestOperation).unwrap();
    assert_eq!(filled.value.width(), 200);
    assert_eq!(filled.distance.width(), 200);
}

#[test]
fn hist_ismonotonic_on_identity() {
    let lut = Image::<VipsBackend>::generate(&Identity).unwrap();
    let m = lut.execute(&HistIsmonotonicOperation).unwrap();
    assert!(m.0);
}

#[test]
fn rad_roundtrip() {
    let f = sample().execute(&Float2radOperation).unwrap();
    let back = f.execute(&Rad2floatOperation).unwrap();
    assert_eq!(back.width(), 200);
}

#[test]
fn array_join_and_switch() {
    let a = sample();
    let joined = Image::<VipsBackend>::array_join(
        &[&a, &a],
        &ArrayJoinParams {
            across: Some(2),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(joined.width(), 400);

    let gray = a
        .execute(&ExtractBandOperation {
            band: 0,
            count: None,
        })
        .unwrap();
    let mask = gray
        .execute(&RelationalConstOperation {
            relational: OperationRelational::More,
            constants: vec![128.0],
        })
        .unwrap();
    let sw = Image::<VipsBackend>::switch(&[&mask]).unwrap();
    assert_eq!(sw.width(), 200);
}

#[test]
fn reduce_shrink_1d() {
    let img = sample();
    let rh = img
        .execute(&ReduceHorizontalOperation {
            shrink: 2.0,
            kernel: None,
            gap: None,
        })
        .unwrap();
    assert_eq!(rh.width(), 100);
    let sv = img
        .execute(&ShrinkVerticalOperation {
            shrink: 2,
            ceil: None,
        })
        .unwrap();
    assert_eq!(sv.height(), 100);
}

#[test]
fn hough_line() {
    let edges = sample()
        .execute(&ExtractBandOperation {
            band: 0,
            count: None,
        })
        .unwrap()
        .execute(&CannyOperation {
            sigma: None,
            precision: None,
        })
        .unwrap();
    let out = edges
        .execute(&HoughLineOperation {
            width: Some(128),
            height: Some(128),
        })
        .unwrap();
    assert_eq!(out.width(), 128);
}

#[test]
fn sum_and_bandrank() {
    let a = sample();
    let summed = Image::<VipsBackend>::sum(&[&a, &a]).unwrap();
    assert_eq!(summed.width(), 200);
    let ranked = Image::<VipsBackend>::band_rank(&[&a, &a, &a], 1).unwrap();
    assert_eq!(ranked.width(), 200);
}

#[test]
fn composite_stack() {
    let a = sample();
    let b = a
        .execute(&ResizeOperation {
            scale: 0.5,
            kernel: None,
            vertical_scale: None,
            gap: None,
        })
        .unwrap();
    let out =
        Image::<VipsBackend>::composite(&[&a, &b], &[BlendMode::Over], &CompositeParams::default())
            .unwrap();
    assert_eq!(out.width(), 200);
}

#[test]
fn thumbnail_loaders() {
    init();
    let t =
        Image::<VipsBackend>::thumbnail("tests/fixtures/rgb.jpg", 64, &ThumbnailParams::default())
            .unwrap();
    assert!(t.width() <= 64);

    let buf = sample().write_to_buffer(".png").unwrap();
    let tb = Image::<VipsBackend>::thumbnail_buffer(&buf, 32, &ThumbnailParams::default()).unwrap();
    assert!(tb.width() <= 32);
}

#[test]
fn merge_mosaic() {
    let a = sample();
    let merged = a
        .execute(&MergeOperation {
            secondary: &a,
            direction: Direction::Horizontal,
            dx: 200,
            dy: 0,
            max_blend: None,
        })
        .unwrap();
    assert!(merged.width() >= 200);
}

#[test]
fn from_memory_roundtrip() {
    init();
    // 2x2 RGB, 3 bytes/px = 12 bytes.
    let buf = vec![10u8, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
    let img = Image::<VipsBackend>::from_memory(&buf, 2, 2, 3, PixelFormat::Rgb8).unwrap();
    assert_eq!(img.width(), 2);
    assert_eq!(img.bands(), 3);
}

#[test]
fn from_memory_rejects_short_buffer() {
    init();
    let buf = vec![0u8; 5]; // need 12
    assert!(Image::<VipsBackend>::from_memory(&buf, 2, 2, 3, PixelFormat::Rgb8).is_err());
}

#[test]
fn from_memory_rejects_bad_dims() {
    init();
    let buf = vec![0u8; 12];
    assert!(Image::<VipsBackend>::from_memory(&buf, 0, 2, 3, PixelFormat::Rgb8).is_err());
}

#[test]
fn source_memory_outlives_buffer() {
    init();
    // Source must not dangle when the original buffer is dropped.
    let png = sample().write_to_buffer(".png").unwrap();
    let source = {
        let owned = png.clone();
        Source::new_from_memory(&owned).unwrap()
        // `owned` drops here; the source holds a vips-owned copy.
    };
    let img = Image::<VipsBackend>::new_from_source(&source).unwrap();
    assert_eq!(img.width(), 200);
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
    let thumb = sample()
        .execute(&ThumbnailOperation {
            width: 64,
            height: None,
            size: None,
            crop: None,
            linear: None,
            auto_rotate: None,
            no_rotate: None,
            import_profile: None,
            export_profile: None,
            intent: None,
            fail_on: None,
        })
        .unwrap();
    assert!(thumb.width() <= 64);
}
