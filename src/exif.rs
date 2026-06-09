use serde::{Deserialize, Serialize};

/// Typed image metadata, format-agnostic.
///
/// Variants are grouped by category. The [`Custom`] variant provides a
/// catch-all for unknown or non-standard tags.
///
/// [`Custom`]: Metadata::Custom
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Metadata {
    // ── Identity ──────────────────────────────────────────────────────────
    Make(String),
    Model(String),
    Software(String),
    ProcessingSoftware(String),
    HostComputer(String),

    // ── Author / Rights ──────────────────────────────────────────────────
    Artist(String),
    Copyright(String),
    Description(String),
    DocumentName(String),
    ImageTitle(String),
    Photographer(String),
    ImageEditor(String),
    Rating(u16),
    RatingPercent(u16),

    // ── Date / Time ──────────────────────────────────────────────────────
    DateTime(String),
    DateTimeOriginal(String),
    DateTimeDigitized(String),
    SubSecTime(String),
    SubSecTimeOriginal(String),
    SubSecTimeDigitized(String),
    OffsetTime(String),
    OffsetTimeOriginal(String),
    OffsetTimeDigitized(String),

    // ── Camera ───────────────────────────────────────────────────────────
    ExposureTime(String),
    ShutterSpeedValue(String),
    FNumber(String),
    ApertureValue(String),
    BrightnessValue(String),
    ExposureProgram(u16),
    ExposureMode(u16),
    ISOSpeedRatings(u32),
    SensitivityType(u16),
    StandardOutputSensitivity(u32),
    RecommendedExposureIndex(u32),
    ISOSpeed(u32),
    ISOSpeedLatitudeyyy(u32),
    ISOSpeedLatitudezzz(u32),
    FocalLength(String),
    FocalLengthIn35mm(u16),
    DigitalZoomRatio(f64),
    FocalPlaneXResolution(f64),
    FocalPlaneYResolution(f64),
    FocalPlaneResolutionUnit(u16),
    Flash(u16),
    FlashEnergy(f64),
    ExposureBias(String),
    ExposureIndex(String),
    MeteringMode(u16),
    SensingMethod(u16),
    FileSource(u8),
    SceneType(u8),
    SceneCaptureType(u16),
    CustomRendered(u16),
    GainControl(u16),
    Contrast(u16),
    Saturation(u16),
    Sharpness(u16),
    WhiteBalance(u16),
    LightSource(u16),
    SelfTimerMode(u16),
    SubjectArea(Vec<u16>),
    SubjectLocation([u16; 2]),
    SubjectDistance(String),
    SubjectDistanceRange(u16),

    // ── Lens / Body ──────────────────────────────────────────────────────
    LensMake(String),
    LensModel(String),
    LensSpecification(String),
    LensSerialNumber(String),
    MaxAperture(String),
    BodySerialNumber(String),
    CameraOwnerName(String),
    CameraFirmware(String),

    // ── Image ────────────────────────────────────────────────────────────
    ImageWidth(u32),
    ImageHeight(u32),
    PixelXDimension(u32),
    PixelYDimension(u32),
    SamplesPerPixel(u16),
    BitsPerSample(Vec<u16>),
    Dpi {
        x: f32,
        y: f32,
    },
    ResolutionUnit(u16),
    Orientation(u16),
    ImageNumber(u32),
    ImageUniqueId(String),
    ExifVersion(String),
    ColorSpaceTag(String),

    // ── Page / Tiling ────────────────────────────────────────────────────
    PageName(String),
    PageNumber(u16, u16),
    SubfileType(u32),
    XPosition(f64),
    YPosition(f64),

    // ── Comments / Maker Note ────────────────────────────────────────────
    UserComment(String),
    MakerNote(Vec<u8>),
    RelatedSoundFile(String),

    // ── Software chain ───────────────────────────────────────────────────
    RAWDevelopingSoftware(String),
    ImageEditingSoftware(String),
    MetadataEditingSoftware(String),
    ImageHistory(String),

    // ── Composite ────────────────────────────────────────────────────────
    CompositeImage(u16),
    CompositeImageCount([u16; 2]),

    // ── Correction flags ─────────────────────────────────────────────────
    DistortionCorrection(u16),
    ChromaticAberrationCorrection(u16),
    ShadingCorrection(u16),
    NoiseReduction(u16),
    LearningOptOutIn(Vec<u8>),

    // ── Color / HDR ─────────────────────────────────────────────────────
    WhitePoint([f64; 2]),
    PrimaryChromaticities {
        red: [f64; 2],
        green: [f64; 2],
        blue: [f64; 2],
    },
    TransferFunction(Vec<u16>),
    Gamma(f64),
    IccProfile(Vec<u8>),
    PhotometricInterpretation(u16),
    CFAPattern(Vec<u8>),
    ComponentsConfiguration(u32),
    CompressedBitsPerPixel(f64),

    // ── HDR (PNG) ───────────────────────────────────────────────────────
    MasteringDisplayLuminance {
        min: f64,
        max: f64,
    },
    ContentLightLevel {
        max_fall: f64,
        max_cll: f64,
    },

    // ── YCbCr ───────────────────────────────────────────────────────────
    YCbCrCoefficients([f64; 3]),
    YCbCrPositioning(u16),
    YCbCrSubSampling(Vec<u16>),
    ReferenceBlackWhite([f64; 6]),

    // ── Compression / Layout ────────────────────────────────────────────
    Compression(u16),
    PlanarConfiguration(u16),
    Predictor(u16),
    TileWidth(u32),
    TileLength(u32),

    // ── Environment ──────────────────────────────────────────────────────
    AmbientTemperature(f64),
    Humidity(f64),
    Pressure(f64),
    WaterDepth(f64),
    Acceleration(f64),
    CameraElevationAngle(f64),

    // ── GPS ─────────────────────────────────────────────────────────────
    GPSVersionID([u8; 4]),
    GpsLatitudeRef(String),
    GpsLatitude(Vec<f64>),
    GpsLongitudeRef(String),
    GpsLongitude(Vec<f64>),
    GpsAltitudeRef(u8),
    GpsAltitude(f64),
    GPSTimeStamp(Vec<f64>),
    GpsDateStamp(String),
    GPSSatellites(String),
    GPSStatus(String),
    GPSMeasureMode(String),
    GPSDOP(f64),
    GPSSpeedRef(String),
    GPSSpeed(f64),
    GPSTrackRef(String),
    GPSTrack(f64),
    GPSImgDirectionRef(String),
    GPSImgDirection(f64),
    GPSMapDatum(String),
    GPSDestLatitudeRef(String),
    GPSDestLatitude(Vec<f64>),
    GPSDestLongitudeRef(String),
    GPSDestLongitude(Vec<f64>),
    GPSDestBearingRef(String),
    GPSDestBearing(f64),
    GPSDestDistanceRef(String),
    GPSDestDistance(f64),
    GPSProcessingMethod(String),
    GPSAreaInformation(String),
    GPSHPositioningError(f64),

    // ── Interoperability ────────────────────────────────────────────────
    InteropIndex(String),
    InteropVersion(String),
    RelatedImageFileFormat(String),
    RelatedImageWidth(u16),
    RelatedImageHeight(u16),

    // ── Windows XP ──────────────────────────────────────────────────────
    XPTitle(Vec<u8>),
    XPComment(Vec<u8>),
    XPAuthor(Vec<u8>),
    XPKeywords(Vec<u8>),
    XPSubject(Vec<u8>),

    // ── Catch-all ────────────────────────────────────────────────────────
    Custom {
        key: String,
        value: String,
    },
}

// ── Extraction from VipsImage ────────────────────────────────────────────────

use crate::libvips_ffi as ffi;
use std::ffi::{CStr, CString};

fn get(vips: *mut ffi::VipsImage, key: &str) -> Option<String> {
    let c_key = CString::new(key).ok()?;
    unsafe {
        let mut out: *mut std::ffi::c_char = std::ptr::null_mut();
        if ffi::vips_image_get_as_string(vips, c_key.as_ptr(), &mut out) != 0 {
            return None;
        }
        if out.is_null() {
            return None;
        }
        // `out` is g_malloc'd by vips; copy it out then free.
        let s = strip_vips_annotation(CStr::from_ptr(out).to_string_lossy().as_ref()).to_string();
        ffi::g_free(out as *mut std::ffi::c_void);
        if s.is_empty() { None } else { Some(s) }
    }
}

pub fn extract(vips: *mut ffi::VipsImage) -> Vec<Metadata> {
    let mut meta = Vec::new();

    if let Some(icc) = get(vips, "icc-profile-data") {
        meta.push(Metadata::IccProfile(icc.into_bytes()));
    }

    let unit = get(vips, "exif-ifd0-ResolutionUnit").and_then(|v| v.parse::<u16>().ok());
    let x_res_str = get(vips, "exif-ifd0-XResolution");
    let y_res_str = get(vips, "exif-ifd0-YResolution");
    if let (Some(xs), Some(ys)) = (x_res_str, y_res_str)
        && let (Ok(x), Ok(y)) = (parse_rational_f32(&xs), parse_rational_f32(&ys))
    {
        let mult = match unit {
            Some(3) => 2.54,
            _ => 1.0,
        };
        meta.push(Metadata::Dpi {
            x: x * mult,
            y: y * mult,
        });
    }

    for key in &["xmp-data", "iptc-data", "exif-data"] {
        if let Some(val) = get(vips, key) {
            let short = key.strip_suffix("-data").unwrap_or(key);
            meta.push(Metadata::Custom {
                key: short.into(),
                value: val,
            });
        }
    }

    let string_keys: &[(&str, fn(String) -> Metadata)] = &[
        ("exif-ifd0-Make", Metadata::Make),
        ("exif-ifd0-Model", Metadata::Model),
        ("exif-ifd0-Software", Metadata::Software),
        ("exif-ifd0-ProcessingSoftware", Metadata::ProcessingSoftware),
        ("exif-ifd0-Artist", Metadata::Artist),
        ("exif-ifd0-Copyright", Metadata::Copyright),
        ("exif-ifd0-ImageDescription", Metadata::Description),
        ("exif-ifd0-DateTime", Metadata::DateTime),
        ("exif-ifd2-DateTimeOriginal", Metadata::DateTimeOriginal),
        ("exif-ifd2-DateTimeDigitized", Metadata::DateTimeDigitized),
        ("exif-ifd2-SubSecTime", Metadata::SubSecTime),
        ("exif-ifd2-SubSecTimeOriginal", Metadata::SubSecTimeOriginal),
        (
            "exif-ifd2-SubSecTimeDigitized",
            Metadata::SubSecTimeDigitized,
        ),
        ("exif-ifd2-OffsetTime", Metadata::OffsetTime),
        ("exif-ifd2-OffsetTimeOriginal", Metadata::OffsetTimeOriginal),
        (
            "exif-ifd2-OffsetTimeDigitized",
            Metadata::OffsetTimeDigitized,
        ),
        ("exif-ifd2-ExposureTime", Metadata::ExposureTime),
        ("exif-ifd2-ShutterSpeedValue", Metadata::ShutterSpeedValue),
        ("exif-ifd2-FNumber", Metadata::FNumber),
        ("exif-ifd2-ApertureValue", Metadata::ApertureValue),
        ("exif-ifd2-BrightnessValue", Metadata::BrightnessValue),
        ("exif-ifd2-FocalLength", Metadata::FocalLength),
        ("exif-ifd2-ExposureBiasValue", Metadata::ExposureBias),
        ("exif-ifd2-ExposureIndex", Metadata::ExposureIndex),
        ("exif-ifd2-LensModel", Metadata::LensModel),
        ("exif-ifd2-LensMake", Metadata::LensMake),
        ("exif-ifd2-LensSpecification", Metadata::LensSpecification),
        ("exif-ifd2-LensSerialNumber", Metadata::LensSerialNumber),
        ("exif-ifd2-MaxApertureValue", Metadata::MaxAperture),
        ("exif-ifd2-BodySerialNumber", Metadata::BodySerialNumber),
        ("exif-ifd2-CameraOwnerName", Metadata::CameraOwnerName),
        ("exif-ifd2-CameraFirmware", Metadata::CameraFirmware),
        ("exif-ifd2-ImageUniqueID", Metadata::ImageUniqueId),
        ("exif-ifd2-ExifVersion", Metadata::ExifVersion),
        ("exif-ifd2-ColorSpace", Metadata::ColorSpaceTag),
        ("exif-ifd2-UserComment", Metadata::UserComment),
        ("exif-ifd2-RelatedSoundFile", Metadata::RelatedSoundFile),
        ("exif-ifd0-DocumentName", Metadata::DocumentName),
        ("exif-ifd0-HostComputer", Metadata::HostComputer),
        ("exif-ifd0-PageName", Metadata::PageName),
        ("exif-ifd2-ImageTitle", Metadata::ImageTitle),
        ("exif-ifd2-Photographer", Metadata::Photographer),
        ("exif-ifd2-ImageEditor", Metadata::ImageEditor),
        (
            "exif-ifd2-RAWDevelopingSoftware",
            Metadata::RAWDevelopingSoftware,
        ),
        (
            "exif-ifd2-ImageEditingSoftware",
            Metadata::ImageEditingSoftware,
        ),
        (
            "exif-ifd2-MetadataEditingSoftware",
            Metadata::MetadataEditingSoftware,
        ),
        ("exif-ifd2-ImageHistory", Metadata::ImageHistory),
        ("exif-ifd2-SubjectDistance", Metadata::SubjectDistance),
        ("exif-gps-GPSLatitudeRef", Metadata::GpsLatitudeRef),
        ("exif-gps-GPSLongitudeRef", Metadata::GpsLongitudeRef),
        ("exif-gps-GPSDateStamp", Metadata::GpsDateStamp),
        ("exif-gps-GPSSatellites", Metadata::GPSSatellites),
        ("exif-gps-GPSStatus", Metadata::GPSStatus),
        ("exif-gps-GPSMeasureMode", Metadata::GPSMeasureMode),
        ("exif-gps-GPSSpeedRef", Metadata::GPSSpeedRef),
        ("exif-gps-GPSTrackRef", Metadata::GPSTrackRef),
        ("exif-gps-GPSImgDirectionRef", Metadata::GPSImgDirectionRef),
        ("exif-gps-GPSMapDatum", Metadata::GPSMapDatum),
        ("exif-gps-GPSDestLatitudeRef", Metadata::GPSDestLatitudeRef),
        (
            "exif-gps-GPSDestLongitudeRef",
            Metadata::GPSDestLongitudeRef,
        ),
        ("exif-gps-GPSDestBearingRef", Metadata::GPSDestBearingRef),
        ("exif-gps-GPSDestDistanceRef", Metadata::GPSDestDistanceRef),
        (
            "exif-gps-GPSProcessingMethod",
            Metadata::GPSProcessingMethod,
        ),
        ("exif-gps-GPSAreaInformation", Metadata::GPSAreaInformation),
        ("exif-ifd2-InteropIndex", Metadata::InteropIndex),
        ("exif-ifd2-InteropVersion", Metadata::InteropVersion),
        (
            "exif-ifd2-RelatedImageFileFormat",
            Metadata::RelatedImageFileFormat,
        ),
    ];
    for (key, ctor) in string_keys {
        if let Some(val) = get(vips, key) {
            meta.push(ctor(val));
        }
    }

    let u16_keys: &[(&str, fn(u16) -> Metadata)] = &[
        ("exif-ifd2-ExposureProgram", Metadata::ExposureProgram),
        ("exif-ifd2-ExposureMode", Metadata::ExposureMode),
        (
            "exif-ifd2-FocalLengthIn35mmFilm",
            Metadata::FocalLengthIn35mm,
        ),
        ("exif-ifd2-Flash", Metadata::Flash),
        ("exif-ifd2-SelfTimerMode", Metadata::SelfTimerMode),
        ("exif-ifd2-SensitivityType", Metadata::SensitivityType),
        ("exif-ifd0-Orientation", Metadata::Orientation),
        ("exif-ifd2-MeteringMode", Metadata::MeteringMode),
        ("exif-ifd2-SensingMethod", Metadata::SensingMethod),
        ("exif-ifd2-WhiteBalance", Metadata::WhiteBalance),
        ("exif-ifd2-LightSource", Metadata::LightSource),
        ("exif-ifd2-SceneCaptureType", Metadata::SceneCaptureType),
        ("exif-ifd2-CustomRendered", Metadata::CustomRendered),
        ("exif-ifd2-GainControl", Metadata::GainControl),
        ("exif-ifd2-Contrast", Metadata::Contrast),
        ("exif-ifd2-Saturation", Metadata::Saturation),
        ("exif-ifd2-Sharpness", Metadata::Sharpness),
        (
            "exif-ifd2-SubjectDistanceRange",
            Metadata::SubjectDistanceRange,
        ),
        (
            "exif-ifd2-DistortionCorrection",
            Metadata::DistortionCorrection,
        ),
        (
            "exif-ifd2-ChromaticAberrationCorrection",
            Metadata::ChromaticAberrationCorrection,
        ),
        ("exif-ifd2-ShadingCorrection", Metadata::ShadingCorrection),
        ("exif-ifd2-NoiseReduction", Metadata::NoiseReduction),
        (
            "exif-ifd0-PhotometricInterpretation",
            Metadata::PhotometricInterpretation,
        ),
        ("exif-ifd0-Compression", Metadata::Compression),
        (
            "exif-ifd0-PlanarConfiguration",
            Metadata::PlanarConfiguration,
        ),
        ("exif-ifd0-YCbCrPositioning", Metadata::YCbCrPositioning),
        ("exif-ifd0-Rating", Metadata::Rating),
        ("exif-ifd0-RatingPercent", Metadata::RatingPercent),
        ("exif-ifd0-SamplesPerPixel", Metadata::SamplesPerPixel),
        ("exif-ifd2-CompositeImage", Metadata::CompositeImage),
        (
            "exif-ifd2-FocalPlaneResolutionUnit",
            Metadata::FocalPlaneResolutionUnit,
        ),
        ("exif-ifd2-RelatedImageWidth", Metadata::RelatedImageWidth),
        ("exif-ifd2-RelatedImageHeight", Metadata::RelatedImageHeight),
    ];
    for (key, ctor) in u16_keys {
        if let Some(val) = get(vips, key)
            && let Ok(n) = val.parse()
        {
            meta.push(ctor(n));
        }
    }

    let u32_keys: &[(&str, fn(u32) -> Metadata)] = &[
        ("exif-ifd2-ISOSpeedRatings", Metadata::ISOSpeedRatings),
        (
            "exif-ifd2-StandardOutputSensitivity",
            Metadata::StandardOutputSensitivity,
        ),
        (
            "exif-ifd2-RecommendedExposureIndex",
            Metadata::RecommendedExposureIndex,
        ),
        ("exif-ifd2-ISOSpeed", Metadata::ISOSpeed),
        (
            "exif-ifd2-ISOSpeedLatitudeyyy",
            Metadata::ISOSpeedLatitudeyyy,
        ),
        (
            "exif-ifd2-ISOSpeedLatitudezzz",
            Metadata::ISOSpeedLatitudezzz,
        ),
        ("exif-ifd0-ImageWidth", Metadata::ImageWidth),
        ("exif-ifd0-ImageLength", Metadata::ImageHeight),
        ("exif-ifd2-PixelXDimension", Metadata::PixelXDimension),
        ("exif-ifd2-PixelYDimension", Metadata::PixelYDimension),
        ("exif-ifd2-ImageNumber", Metadata::ImageNumber),
        ("exif-ifd0-SubfileType", Metadata::SubfileType),
        ("exif-ifd0-TileWidth", Metadata::TileWidth),
        ("exif-ifd0-TileLength", Metadata::TileLength),
        (
            "exif-ifd2-ComponentsConfiguration",
            Metadata::ComponentsConfiguration,
        ),
    ];
    for (key, ctor) in u32_keys {
        if let Some(val) = get(vips, key)
            && let Ok(n) = val.parse()
        {
            meta.push(ctor(n));
        }
    }

    if let Some(val) = get(vips, "exif-ifd2-FileSource")
        && let Ok(n) = val.parse()
    {
        meta.push(Metadata::FileSource(n));
    }
    if let Some(val) = get(vips, "exif-ifd2-SceneType")
        && let Ok(n) = val.parse()
    {
        meta.push(Metadata::SceneType(n));
    }

    let f64_keys: &[(&str, fn(f64) -> Metadata)] = &[
        ("exif-ifd2-DigitalZoomRatio", Metadata::DigitalZoomRatio),
        ("exif-ifd2-FlashEnergy", Metadata::FlashEnergy),
        (
            "exif-ifd2-FocalPlaneXResolution",
            Metadata::FocalPlaneXResolution,
        ),
        (
            "exif-ifd2-FocalPlaneYResolution",
            Metadata::FocalPlaneYResolution,
        ),
        (
            "exif-ifd2-CompressedBitsPerPixel",
            Metadata::CompressedBitsPerPixel,
        ),
        ("exif-ifd2-AmbientTemperature", Metadata::AmbientTemperature),
        ("exif-ifd2-Humidity", Metadata::Humidity),
        ("exif-ifd2-Pressure", Metadata::Pressure),
        ("exif-ifd2-WaterDepth", Metadata::WaterDepth),
        ("exif-ifd2-Acceleration", Metadata::Acceleration),
        (
            "exif-ifd2-CameraElevationAngle",
            Metadata::CameraElevationAngle,
        ),
        ("exif-gps-GPSDOP", Metadata::GPSDOP),
        ("exif-gps-GPSSpeed", Metadata::GPSSpeed),
        ("exif-gps-GPSTrack", Metadata::GPSTrack),
        ("exif-gps-GPSImgDirection", Metadata::GPSImgDirection),
        ("exif-gps-GPSDestBearing", Metadata::GPSDestBearing),
        ("exif-gps-GPSDestDistance", Metadata::GPSDestDistance),
        (
            "exif-gps-GPSHPositioningError",
            Metadata::GPSHPositioningError,
        ),
    ];
    for (key, ctor) in f64_keys {
        if let Some(val) = get(vips, key)
            && let Ok(n) = parse_rational_f64(&val)
        {
            meta.push(ctor(n));
        }
    }

    if let Some(val) = get(vips, "exif-ifd0-XPosition")
        && let Ok(n) = parse_rational_f64(&val)
    {
        meta.push(Metadata::XPosition(n));
    }
    if let Some(val) = get(vips, "exif-ifd0-YPosition")
        && let Ok(n) = parse_rational_f64(&val)
    {
        meta.push(Metadata::YPosition(n));
    }

    if let Some(val) = get(vips, "exif-ifd0-PageNumber") {
        let parts: Vec<u16> = val
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if parts.len() >= 2 {
            meta.push(Metadata::PageNumber(parts[0], parts[1]));
        }
    }
    if let Some(val) = get(vips, "exif-ifd0-Predictor")
        && let Ok(n) = val.parse::<u16>()
    {
        meta.push(Metadata::Predictor(n));
    }
    if let Some(val) = get(vips, "exif-ifd0-YCbCrCoefficients") {
        let parts: Vec<f64> = val
            .split_whitespace()
            .filter_map(|s| parse_rational_f64(s).ok())
            .collect();
        if parts.len() == 3 {
            meta.push(Metadata::YCbCrCoefficients([parts[0], parts[1], parts[2]]));
        }
    }
    if let Some(val) = get(vips, "exif-ifd0-ReferenceBlackWhite") {
        let parts: Vec<f64> = val
            .split_whitespace()
            .filter_map(|s| parse_rational_f64(s).ok())
            .collect();
        if parts.len() == 6 {
            meta.push(Metadata::ReferenceBlackWhite([
                parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
            ]));
        }
    }
    if let Some(val) = get(vips, "exif-ifd0-ResolutionUnit")
        && let Ok(n) = val.parse::<u16>()
    {
        meta.push(Metadata::ResolutionUnit(n));
    }
    if let Some(val) = get(vips, "exif-gps-GPSAltitudeRef")
        && let Ok(n) = val.parse()
    {
        meta.push(Metadata::GpsAltitudeRef(n));
    }
    if let Some(val) = get(vips, "exif-gps-GPSAltitude")
        && let Ok(n) = parse_rational_f64(&val)
    {
        meta.push(Metadata::GpsAltitude(n));
    }
    if let Some(val) = get(vips, "exif-gps-GPSLatitude")
        && let Some(v) = parse_gps_coord(&val)
    {
        meta.push(Metadata::GpsLatitude(v));
    }
    if let Some(val) = get(vips, "exif-gps-GPSLongitude")
        && let Some(v) = parse_gps_coord(&val)
    {
        meta.push(Metadata::GpsLongitude(v));
    }
    if let Some(val) = get(vips, "exif-gps-GPSDestLatitude")
        && let Some(v) = parse_gps_coord(&val)
    {
        meta.push(Metadata::GPSDestLatitude(v));
    }
    if let Some(val) = get(vips, "exif-gps-GPSDestLongitude")
        && let Some(v) = parse_gps_coord(&val)
    {
        meta.push(Metadata::GPSDestLongitude(v));
    }
    if let Some(val) = get(vips, "exif-gps-GPSTimeStamp") {
        let parts: Vec<f64> = val
            .split_whitespace()
            .filter_map(|t| parse_rational_f64(t).ok())
            .collect();
        if parts.len() == 3 {
            meta.push(Metadata::GPSTimeStamp(parts));
        }
    }
    if let Some(val) = get(vips, "exif-gps-GPSVersionID") {
        let parts: Vec<u8> = val
            .split_whitespace()
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        if parts.len() == 4 {
            meta.push(Metadata::GPSVersionID([
                parts[0], parts[1], parts[2], parts[3],
            ]));
        }
    }
    if let Some(val) = get(vips, "exif-ifd2-SubjectLocation") {
        let parts: Vec<u16> = val
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if parts.len() >= 2 {
            meta.push(Metadata::SubjectLocation([parts[0], parts[1]]));
        }
    }
    if let Some(val) = get(vips, "exif-ifd2-CompositeImageCount") {
        let parts: Vec<u16> = val
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if parts.len() >= 2 {
            meta.push(Metadata::CompositeImageCount([parts[0], parts[1]]));
        }
    }
    if let Some(val) = get(vips, "exif-ifd0-TransferFunction") {
        let v: Vec<u16> = val
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if !v.is_empty() {
            meta.push(Metadata::TransferFunction(v));
        }
    }
    if let Some(val) = get(vips, "exif-ifd2-LearningOptOutIn") {
        meta.push(Metadata::LearningOptOutIn(val.into_bytes()));
    }
    for (key, ctor) in &[
        (
            "exif-ifd0-XPTitle",
            Metadata::XPTitle as fn(Vec<u8>) -> Metadata,
        ),
        ("exif-ifd0-XPComment", Metadata::XPComment),
        ("exif-ifd0-XPAuthor", Metadata::XPAuthor),
        ("exif-ifd0-XPKeywords", Metadata::XPKeywords),
        ("exif-ifd0-XPSubject", Metadata::XPSubject),
    ] {
        if let Some(val) = get(vips, key) {
            meta.push(ctor(val.into_bytes()));
        }
    }
    if let Some(val) = get(vips, "exif-maker-MakerNote") {
        meta.push(Metadata::MakerNote(val.into_bytes()));
    }
    if let Some(val) = get(vips, "exif-ifd0-MakerNote") {
        meta.push(Metadata::MakerNote(val.into_bytes()));
    }
    if let Some(val) = get(vips, "exif-ifd0-YCbCrSubSampling") {
        let v: Vec<u16> = val
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if !v.is_empty() {
            meta.push(Metadata::YCbCrSubSampling(v));
        }
    }
    if let Some(val) = get(vips, "exif-ifd2-SubjectArea") {
        let v: Vec<u16> = val
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if !v.is_empty() {
            meta.push(Metadata::SubjectArea(v));
        }
    }
    if let Some(val) = get(vips, "exif-ifd2-CFAPattern") {
        meta.push(Metadata::CFAPattern(val.into_bytes()));
    }
    if let Some(val) = get(vips, "exif-ifd0-BitsPerSample") {
        let v: Vec<u16> = val
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if !v.is_empty() {
            meta.push(Metadata::BitsPerSample(v));
        }
    }
    if let Some(val) = get(vips, "exif-ifd0-Gamma")
        && let Ok(g) = parse_rational_f64(&val)
    {
        meta.push(Metadata::Gamma(g));
    }
    if let Some(val) = get(vips, "exif-ifd0-WhitePoint") {
        let parts: Vec<f64> = val
            .split_whitespace()
            .filter_map(|s| parse_rational_f64(s).ok())
            .collect();
        if parts.len() == 2 {
            meta.push(Metadata::WhitePoint([parts[0], parts[1]]));
        }
    }
    if let Some(val) = get(vips, "exif-ifd0-PrimaryChromaticities") {
        let parts: Vec<f64> = val
            .split_whitespace()
            .filter_map(|s| parse_rational_f64(s).ok())
            .collect();
        if parts.len() == 6 {
            meta.push(Metadata::PrimaryChromaticities {
                red: [parts[0], parts[1]],
                green: [parts[2], parts[3]],
                blue: [parts[4], parts[5]],
            });
        }
    }

    meta
}

fn strip_vips_annotation(s: &str) -> &str {
    s.trim_end()
        .strip_suffix(')')
        .and_then(|rest| rest.rsplit_once(" ("))
        .map(|(val, _)| val)
        .unwrap_or(s)
}

fn parse_rational_f32(s: &str) -> Result<f32, &'static str> {
    let s = strip_vips_annotation(s).trim();
    if let Some((num, den)) = s.split_once('/') {
        let n: f32 = num.trim().parse().map_err(|_| "invalid numerator")?;
        let d: f32 = den.trim().parse().map_err(|_| "invalid denominator")?;
        if d == 0.0 {
            return Err("zero denominator");
        }
        Ok(n / d)
    } else {
        s.parse().map_err(|_| "invalid float")
    }
}

fn parse_rational_f64(s: &str) -> Result<f64, &'static str> {
    let s = strip_vips_annotation(s).trim();
    if let Some((num, den)) = s.split_once('/') {
        let n: f64 = num.trim().parse().map_err(|_| "invalid numerator")?;
        let d: f64 = den.trim().parse().map_err(|_| "invalid denominator")?;
        if d == 0.0 {
            return Err("zero denominator");
        }
        Ok(n / d)
    } else {
        s.parse().map_err(|_| "invalid float")
    }
}

fn parse_gps_coord(s: &str) -> Option<Vec<f64>> {
    let parts: Vec<f64> = s
        .split_whitespace()
        .filter_map(|t| parse_rational_f64(t).ok())
        .collect();
    if parts.len() == 3 { Some(parts) } else { None }
}

// ── Display / Decode ─────────────────────────────────────────────────────────

impl Metadata {
    pub fn label(&self) -> &str {
        match self {
            Metadata::Make(_) => "Make",
            Metadata::Model(_) => "Model",
            Metadata::Software(_) => "Software",
            Metadata::ProcessingSoftware(_) => "Processing SW",
            Metadata::HostComputer(_) => "Host Computer",
            Metadata::Artist(_) => "Artist",
            Metadata::Copyright(_) => "Copyright",
            Metadata::Description(_) => "Description",
            Metadata::DocumentName(_) => "Document Name",
            Metadata::ImageTitle(_) => "Title",
            Metadata::Photographer(_) => "Photographer",
            Metadata::ImageEditor(_) => "Editor",
            Metadata::Rating(_) => "Rating",
            Metadata::RatingPercent(_) => "Rating %",
            Metadata::DateTime(_) => "Date/Time",
            Metadata::DateTimeOriginal(_) => "Original Date",
            Metadata::DateTimeDigitized(_) => "Digitized Date",
            Metadata::SubSecTime(_) => "SubSec (Modify)",
            Metadata::SubSecTimeOriginal(_) => "SubSec (Original)",
            Metadata::SubSecTimeDigitized(_) => "SubSec (Digitized)",
            Metadata::OffsetTime(_) => "TZ Offset",
            Metadata::OffsetTimeOriginal(_) => "TZ Offset Orig.",
            Metadata::OffsetTimeDigitized(_) => "TZ Offset Digit.",
            Metadata::ExposureTime(_) => "Exposure",
            Metadata::ShutterSpeedValue(_) => "Shutter Speed",
            Metadata::FNumber(_) => "Aperture",
            Metadata::ApertureValue(_) => "Aperture (APEX)",
            Metadata::BrightnessValue(_) => "Brightness",
            Metadata::ExposureProgram(_) => "Exposure Program",
            Metadata::ExposureMode(_) => "Exposure Mode",
            Metadata::ISOSpeedRatings(_) => "ISO",
            Metadata::SensitivityType(_) => "Sensitivity Type",
            Metadata::StandardOutputSensitivity(_) => "ISO (Std Out)",
            Metadata::RecommendedExposureIndex(_) => "ISO (Rec. Exp.)",
            Metadata::ISOSpeed(_) => "ISO Speed",
            Metadata::ISOSpeedLatitudeyyy(_) => "ISO Lat. yyy",
            Metadata::ISOSpeedLatitudezzz(_) => "ISO Lat. zzz",
            Metadata::FocalLength(_) => "Focal Length",
            Metadata::FocalLengthIn35mm(_) => "Focal (35mm eq.)",
            Metadata::DigitalZoomRatio(_) => "Digital Zoom",
            Metadata::FocalPlaneXResolution(_) => "FocalPlane Res X",
            Metadata::FocalPlaneYResolution(_) => "FocalPlane Res Y",
            Metadata::FocalPlaneResolutionUnit(_) => "FocalPlane Unit",
            Metadata::Flash(_) => "Flash",
            Metadata::FlashEnergy(_) => "Flash Energy",
            Metadata::ExposureBias(_) => "Exposure Bias",
            Metadata::ExposureIndex(_) => "Exposure Index",
            Metadata::MeteringMode(_) => "Metering",
            Metadata::SensingMethod(_) => "Sensing Method",
            Metadata::FileSource(_) => "File Source",
            Metadata::SceneType(_) => "Scene Type",
            Metadata::SceneCaptureType(_) => "Scene Capture",
            Metadata::CustomRendered(_) => "Custom Rendered",
            Metadata::GainControl(_) => "Gain Control",
            Metadata::Contrast(_) => "Contrast",
            Metadata::Saturation(_) => "Saturation",
            Metadata::Sharpness(_) => "Sharpness",
            Metadata::WhiteBalance(_) => "White Balance",
            Metadata::LightSource(_) => "Light Source",
            Metadata::SelfTimerMode(_) => "Self Timer",
            Metadata::SubjectArea(_) => "Subject Area",
            Metadata::SubjectLocation(_) => "Subject Loc.",
            Metadata::SubjectDistance(_) => "Subject Dist.",
            Metadata::SubjectDistanceRange(_) => "Subject Dist. Range",
            Metadata::LensMake(_) => "Lens Make",
            Metadata::LensModel(_) => "Lens",
            Metadata::LensSpecification(_) => "Lens Spec",
            Metadata::LensSerialNumber(_) => "Lens S/N",
            Metadata::MaxAperture(_) => "Max Aperture",
            Metadata::BodySerialNumber(_) => "Body S/N",
            Metadata::CameraOwnerName(_) => "Owner",
            Metadata::CameraFirmware(_) => "Firmware",
            Metadata::ImageWidth(_) => "Width",
            Metadata::ImageHeight(_) => "Height",
            Metadata::PixelXDimension(_) => "Valid Width",
            Metadata::PixelYDimension(_) => "Valid Height",
            Metadata::SamplesPerPixel(_) => "Samples/Pixel",
            Metadata::BitsPerSample(_) => "Bits/Sample",
            Metadata::Dpi { .. } => "DPI",
            Metadata::ResolutionUnit(_) => "Resolution Unit",
            Metadata::Orientation(_) => "Orientation",
            Metadata::ImageNumber(_) => "Image #",
            Metadata::ImageUniqueId(_) => "Image UID",
            Metadata::ExifVersion(_) => "EXIF Version",
            Metadata::ColorSpaceTag(_) => "Color Space",
            Metadata::PageName(_) => "Page Name",
            Metadata::PageNumber(_, _) => "Page #",
            Metadata::SubfileType(_) => "Subfile Type",
            Metadata::XPosition(_) => "X Position",
            Metadata::YPosition(_) => "Y Position",
            Metadata::UserComment(_) => "Comment",
            Metadata::MakerNote(_) => "Maker Note",
            Metadata::RelatedSoundFile(_) => "Related Audio",
            Metadata::RAWDevelopingSoftware(_) => "RAW Developer",
            Metadata::ImageEditingSoftware(_) => "Image S/W",
            Metadata::MetadataEditingSoftware(_) => "Meta S/W",
            Metadata::ImageHistory(_) => "Image History",
            Metadata::CompositeImage(_) => "Composite",
            Metadata::CompositeImageCount(_) => "Composite Count",
            Metadata::DistortionCorrection(_) => "Distortion Corr.",
            Metadata::ChromaticAberrationCorrection(_) => "CA Correction",
            Metadata::ShadingCorrection(_) => "Shading Corr.",
            Metadata::NoiseReduction(_) => "Noise Reduction",
            Metadata::LearningOptOutIn(_) => "ML Opt-out",
            Metadata::WhitePoint(_) => "White Point",
            Metadata::PrimaryChromaticities { .. } => "Primaries",
            Metadata::TransferFunction(_) => "Transfer Func.",
            Metadata::Gamma(_) => "Gamma",
            Metadata::IccProfile(_) => "ICC Profile",
            Metadata::PhotometricInterpretation(_) => "Photometric",
            Metadata::CFAPattern(_) => "CFA Pattern",
            Metadata::ComponentsConfiguration(_) => "Components",
            Metadata::CompressedBitsPerPixel(_) => "Compressed BPP",
            Metadata::MasteringDisplayLuminance { .. } => "HDR Display Lum.",
            Metadata::ContentLightLevel { .. } => "HDR Light Level",
            Metadata::YCbCrCoefficients(_) => "YCbCr Coeffs",
            Metadata::YCbCrPositioning(_) => "YCbCr Pos.",
            Metadata::YCbCrSubSampling(_) => "YCbCr SubSampling",
            Metadata::ReferenceBlackWhite(_) => "Ref. Black/White",
            Metadata::Compression(_) => "Compression",
            Metadata::PlanarConfiguration(_) => "Planar Config",
            Metadata::Predictor(_) => "Predictor",
            Metadata::TileWidth(_) => "Tile Width",
            Metadata::TileLength(_) => "Tile Height",
            Metadata::AmbientTemperature(_) => "Temperature",
            Metadata::Humidity(_) => "Humidity",
            Metadata::Pressure(_) => "Pressure",
            Metadata::WaterDepth(_) => "Water Depth",
            Metadata::Acceleration(_) => "Acceleration",
            Metadata::CameraElevationAngle(_) => "Elevation Angle",
            Metadata::GPSVersionID(_) => "GPS Version",
            Metadata::GpsLatitudeRef(_) => "GPS Lat Ref",
            Metadata::GpsLatitude(_) => "GPS Latitude",
            Metadata::GpsLongitudeRef(_) => "GPS Long Ref",
            Metadata::GpsLongitude(_) => "GPS Longitude",
            Metadata::GpsAltitudeRef(_) => "GPS Alt. Ref",
            Metadata::GpsAltitude(_) => "GPS Altitude",
            Metadata::GPSTimeStamp(_) => "GPS Time",
            Metadata::GpsDateStamp(_) => "GPS Date",
            Metadata::GPSSatellites(_) => "GPS Satellites",
            Metadata::GPSStatus(_) => "GPS Status",
            Metadata::GPSMeasureMode(_) => "GPS Measure",
            Metadata::GPSDOP(_) => "GPS DOP",
            Metadata::GPSSpeedRef(_) => "GPS Speed Ref",
            Metadata::GPSSpeed(_) => "GPS Speed",
            Metadata::GPSTrackRef(_) => "GPS Track Ref",
            Metadata::GPSTrack(_) => "GPS Track",
            Metadata::GPSImgDirectionRef(_) => "GPS Dir. Ref",
            Metadata::GPSImgDirection(_) => "GPS Direction",
            Metadata::GPSMapDatum(_) => "GPS Datum",
            Metadata::GPSDestLatitudeRef(_) => "GPS Dest Lat Ref",
            Metadata::GPSDestLatitude(_) => "GPS Dest Lat",
            Metadata::GPSDestLongitudeRef(_) => "GPS Dest Long Ref",
            Metadata::GPSDestLongitude(_) => "GPS Dest Long",
            Metadata::GPSDestBearingRef(_) => "GPS Dest Bear Ref",
            Metadata::GPSDestBearing(_) => "GPS Dest Bearing",
            Metadata::GPSDestDistanceRef(_) => "GPS Dest Dist Ref",
            Metadata::GPSDestDistance(_) => "GPS Dest Dist",
            Metadata::GPSProcessingMethod(_) => "GPS Method",
            Metadata::GPSAreaInformation(_) => "GPS Area",
            Metadata::GPSHPositioningError(_) => "GPS Error",
            Metadata::InteropIndex(_) => "Interop Index",
            Metadata::InteropVersion(_) => "Interop Version",
            Metadata::RelatedImageFileFormat(_) => "Related Format",
            Metadata::RelatedImageWidth(_) => "Related Width",
            Metadata::RelatedImageHeight(_) => "Related Height",
            Metadata::XPTitle(_) => "XP Title",
            Metadata::XPComment(_) => "XP Comment",
            Metadata::XPAuthor(_) => "XP Author",
            Metadata::XPKeywords(_) => "XP Keywords",
            Metadata::XPSubject(_) => "XP Subject",
            Metadata::Custom { key, .. } => key,
        }
    }

    pub fn value_str(&self) -> String {
        match self {
            Metadata::Make(v)
            | Metadata::Model(v)
            | Metadata::Software(v)
            | Metadata::ProcessingSoftware(v)
            | Metadata::HostComputer(v)
            | Metadata::Artist(v)
            | Metadata::Copyright(v)
            | Metadata::Description(v)
            | Metadata::DocumentName(v)
            | Metadata::ImageTitle(v)
            | Metadata::Photographer(v)
            | Metadata::ImageEditor(v)
            | Metadata::DateTime(v)
            | Metadata::DateTimeOriginal(v)
            | Metadata::DateTimeDigitized(v)
            | Metadata::SubSecTime(v)
            | Metadata::SubSecTimeOriginal(v)
            | Metadata::SubSecTimeDigitized(v)
            | Metadata::OffsetTime(v)
            | Metadata::OffsetTimeOriginal(v)
            | Metadata::OffsetTimeDigitized(v)
            | Metadata::ExposureTime(v)
            | Metadata::ShutterSpeedValue(v)
            | Metadata::FNumber(v)
            | Metadata::ApertureValue(v)
            | Metadata::BrightnessValue(v)
            | Metadata::FocalLength(v)
            | Metadata::ExposureBias(v)
            | Metadata::ExposureIndex(v)
            | Metadata::LensMake(v)
            | Metadata::LensModel(v)
            | Metadata::LensSpecification(v)
            | Metadata::LensSerialNumber(v)
            | Metadata::MaxAperture(v)
            | Metadata::BodySerialNumber(v)
            | Metadata::CameraOwnerName(v)
            | Metadata::CameraFirmware(v)
            | Metadata::ImageUniqueId(v)
            | Metadata::ExifVersion(v)
            | Metadata::ColorSpaceTag(v)
            | Metadata::UserComment(v)
            | Metadata::RelatedSoundFile(v)
            | Metadata::PageName(v)
            | Metadata::SubjectDistance(v)
            | Metadata::ImageHistory(v)
            | Metadata::RAWDevelopingSoftware(v)
            | Metadata::ImageEditingSoftware(v)
            | Metadata::MetadataEditingSoftware(v)
            | Metadata::GPSProcessingMethod(v)
            | Metadata::GPSAreaInformation(v)
            | Metadata::GpsLatitudeRef(v)
            | Metadata::GpsLongitudeRef(v)
            | Metadata::GpsDateStamp(v)
            | Metadata::GPSStatus(v)
            | Metadata::GPSMeasureMode(v)
            | Metadata::GPSSatellites(v)
            | Metadata::GPSSpeedRef(v)
            | Metadata::GPSTrackRef(v)
            | Metadata::GPSImgDirectionRef(v)
            | Metadata::GPSMapDatum(v)
            | Metadata::GPSDestLatitudeRef(v)
            | Metadata::GPSDestLongitudeRef(v)
            | Metadata::GPSDestBearingRef(v)
            | Metadata::GPSDestDistanceRef(v)
            | Metadata::InteropIndex(v)
            | Metadata::InteropVersion(v)
            | Metadata::RelatedImageFileFormat(v) => v.clone(),
            Metadata::Rating(v)
            | Metadata::RatingPercent(v)
            | Metadata::ExposureProgram(v)
            | Metadata::ExposureMode(v)
            | Metadata::FocalLengthIn35mm(v)
            | Metadata::Flash(v)
            | Metadata::MeteringMode(v)
            | Metadata::SensingMethod(v)
            | Metadata::WhiteBalance(v)
            | Metadata::LightSource(v)
            | Metadata::SceneCaptureType(v)
            | Metadata::CustomRendered(v)
            | Metadata::GainControl(v)
            | Metadata::Contrast(v)
            | Metadata::Saturation(v)
            | Metadata::Sharpness(v)
            | Metadata::Orientation(v)
            | Metadata::SubjectDistanceRange(v)
            | Metadata::PhotometricInterpretation(v)
            | Metadata::YCbCrPositioning(v)
            | Metadata::Compression(v)
            | Metadata::PlanarConfiguration(v)
            | Metadata::Predictor(v)
            | Metadata::FocalPlaneResolutionUnit(v)
            | Metadata::SelfTimerMode(v)
            | Metadata::SensitivityType(v)
            | Metadata::SamplesPerPixel(v)
            | Metadata::CompositeImage(v)
            | Metadata::DistortionCorrection(v)
            | Metadata::ChromaticAberrationCorrection(v)
            | Metadata::ShadingCorrection(v)
            | Metadata::NoiseReduction(v)
            | Metadata::ResolutionUnit(v)
            | Metadata::RelatedImageWidth(v)
            | Metadata::RelatedImageHeight(v) => format!("{v}"),
            Metadata::FileSource(v) | Metadata::SceneType(v) | Metadata::GpsAltitudeRef(v) => {
                format!("{v}")
            }
            Metadata::ISOSpeedRatings(v)
            | Metadata::StandardOutputSensitivity(v)
            | Metadata::RecommendedExposureIndex(v)
            | Metadata::ISOSpeed(v)
            | Metadata::ISOSpeedLatitudeyyy(v)
            | Metadata::ISOSpeedLatitudezzz(v)
            | Metadata::ImageWidth(v)
            | Metadata::ImageHeight(v)
            | Metadata::PixelXDimension(v)
            | Metadata::PixelYDimension(v)
            | Metadata::ImageNumber(v)
            | Metadata::TileWidth(v)
            | Metadata::TileLength(v)
            | Metadata::SubfileType(v)
            | Metadata::ComponentsConfiguration(v) => format!("{v}"),
            Metadata::DigitalZoomRatio(v) => format!("{:.2}×", v),
            Metadata::FlashEnergy(v) => format!("{:.2} BCPS", v),
            Metadata::FocalPlaneXResolution(v) | Metadata::FocalPlaneYResolution(v) => {
                format!("{:.2}", v)
            }
            Metadata::AmbientTemperature(v) => format!("{:.1} °C", v),
            Metadata::Humidity(v) => format!("{:.1} %", v),
            Metadata::Pressure(v) => format!("{:.1} hPa", v),
            Metadata::WaterDepth(v) => format!("{:.1} m", v),
            Metadata::Acceleration(v) => format!("{:.2} mGal", v),
            Metadata::CameraElevationAngle(v) => format!("{:.2} °", v),
            Metadata::CompressedBitsPerPixel(v) => format!("{:.2}", v),
            Metadata::GPSDOP(v) => format!("{:.3}", v),
            Metadata::GPSSpeed(v) => format!("{:.1}", v),
            Metadata::GPSTrack(v) | Metadata::GPSImgDirection(v) | Metadata::GPSDestBearing(v) => {
                format!("{:.2} °", v)
            }
            Metadata::GPSDestDistance(v) => format!("{:.0} m", v),
            Metadata::GPSHPositioningError(v) => format!("{:.1} m", v),
            Metadata::GpsAltitude(v) => format!("{:.1} m", v),
            Metadata::Gamma(v) => format!("{:.4}", v),
            Metadata::XPosition(v) | Metadata::YPosition(v) => format!("{:.2}", v),
            Metadata::BitsPerSample(v)
            | Metadata::TransferFunction(v)
            | Metadata::SubjectArea(v)
            | Metadata::YCbCrSubSampling(v) => format!("{:?}", v),
            Metadata::Dpi { x, y } => format!("{:.0}×{:.0}", x, y),
            Metadata::WhitePoint(v) => format!("({:.4}, {:.4})", v[0], v[1]),
            Metadata::PrimaryChromaticities { red, green, blue } => format!(
                "R({:.4},{:.4}) G({:.4},{:.4}) B({:.4},{:.4})",
                red[0], red[1], green[0], green[1], blue[0], blue[1]
            ),
            Metadata::IccProfile(v)
            | Metadata::MakerNote(v)
            | Metadata::CFAPattern(v)
            | Metadata::LearningOptOutIn(v)
            | Metadata::XPTitle(v)
            | Metadata::XPComment(v)
            | Metadata::XPAuthor(v)
            | Metadata::XPKeywords(v)
            | Metadata::XPSubject(v) => format!("{} bytes", v.len()),
            Metadata::PageNumber(cur, total) => format!("{cur}/{total}"),
            Metadata::SubjectLocation(v) => format!("({}, {})", v[0], v[1]),
            Metadata::CompositeImageCount(v) => format!("src={} used={}", v[0], v[1]),
            Metadata::GPSVersionID(v) => format!("{}.{}.{}.{}", v[0], v[1], v[2], v[3]),
            Metadata::YCbCrCoefficients(v) => format!("[{:.4}, {:.4}, {:.4}]", v[0], v[1], v[2]),
            Metadata::ReferenceBlackWhite(v) => format!(
                "R0={:.1} R1={:.1} G0={:.1} G1={:.1} B0={:.1} B1={:.1}",
                v[0], v[1], v[2], v[3], v[4], v[5]
            ),
            Metadata::MasteringDisplayLuminance { min, max } => {
                format!("min={:.2} max={:.2} cd/m²", min, max)
            }
            Metadata::ContentLightLevel { max_fall, max_cll } => {
                format!("MaxFALL={:.2} MaxCLL={:.2}", max_fall, max_cll)
            }
            Metadata::GpsLatitude(v)
            | Metadata::GpsLongitude(v)
            | Metadata::GPSTimeStamp(v)
            | Metadata::GPSDestLatitude(v)
            | Metadata::GPSDestLongitude(v) => format!("{:?}", v),
            Metadata::Custom { key: _, value } => value.clone(),
        }
    }

    pub fn decode(&self) -> Option<&'static str> {
        match self {
            Metadata::Orientation(v) => Some(match v {
                1 => "Normal",
                2 => "Mirror H",
                3 => "Rotate 180°",
                4 => "Mirror V",
                5 => "Mirror H + Rotate 270°",
                6 => "Rotate 90° CW",
                7 => "Mirror H + Rotate 90°",
                8 => "Rotate 270° CW",
                _ => return None,
            }),
            Metadata::ExposureProgram(v) => Some(match v {
                0 => "Not defined",
                1 => "Manual",
                2 => "Program AE",
                3 => "Aperture-priority AE",
                4 => "Shutter-priority AE",
                5 => "Creative (Slow)",
                6 => "Action (High)",
                7 => "Portrait",
                8 => "Landscape",
                9 => "Bulb",
                _ => return None,
            }),
            Metadata::ExposureMode(v) => Some(match v {
                0 => "Auto",
                1 => "Manual",
                2 => "Auto bracket",
                _ => return None,
            }),
            Metadata::MeteringMode(v) => Some(match v {
                0 => "Unknown",
                1 => "Average",
                2 => "Center-weighted",
                3 => "Spot",
                4 => "Multi-spot",
                5 => "Multi-segment",
                6 => "Partial",
                255 => "Other",
                _ => return None,
            }),
            Metadata::Flash(v) => {
                let fired = v & 0x01 != 0;
                let ret = v & 0x40 != 0;
                let mode = (v >> 3) & 0x03;
                Some(match (fired, ret, mode) {
                    (false, _, _) => "No Flash",
                    (true, true, _) => "Fired, Return Detected",
                    (true, false, 0) => "Fired, No Return",
                    (true, false, 1) => "Fired, Compulsory",
                    (true, false, 2) => "Fired, Auto",
                    _ => return None,
                })
            }
            Metadata::WhiteBalance(v) => Some(match v {
                0 => "Auto",
                1 => "Manual",
                _ => return None,
            }),
            Metadata::LightSource(v) => Some(match v {
                0 => "Unknown",
                1 => "Daylight",
                2 => "Fluorescent",
                3 => "Tungsten",
                4 => "Flash",
                9 => "Fine Weather",
                10 => "Cloudy",
                11 => "Shade",
                12 => "Daylight Fluor.",
                13 => "Day White Fluor.",
                14 => "Cool White Fluor.",
                15 => "White Fluorescent",
                17 => "Std Light A",
                18 => "Std Light B",
                19 => "Std Light C",
                20 => "D55",
                21 => "D65",
                22 => "D75",
                23 => "D50",
                24 => "ISO Studio Tungsten",
                255 => "Other",
                _ => return None,
            }),
            Metadata::SensingMethod(v) => Some(match v {
                1 => "Not defined",
                2 => "One-chip color",
                3 => "Two-chip color",
                4 => "Three-chip color",
                5 => "Color sequential",
                7 => "Trilinear",
                8 => "Color seq. linear",
                _ => return None,
            }),
            Metadata::FileSource(v) => Some(match v {
                1 => "Film Scanner",
                2 => "Reflection Scanner",
                3 => "Digital Camera",
                _ => return None,
            }),
            Metadata::SceneType(v) => Some(match v {
                1 => "Directly Photographed",
                _ => return None,
            }),
            Metadata::SceneCaptureType(v) => Some(match v {
                0 => "Standard",
                1 => "Landscape",
                2 => "Portrait",
                3 => "Night Scene",
                _ => return None,
            }),
            Metadata::CustomRendered(v) => Some(match v {
                0 => "Normal",
                1 => "Custom",
                _ => return None,
            }),
            Metadata::GainControl(v) => Some(match v {
                0 => "None",
                1 => "Low Gain Up",
                2 => "High Gain Up",
                3 => "Low Gain Down",
                4 => "High Gain Down",
                _ => return None,
            }),
            Metadata::Contrast(v)
            | Metadata::Saturation(v)
            | Metadata::Sharpness(v)
            | Metadata::DistortionCorrection(v)
            | Metadata::ChromaticAberrationCorrection(v)
            | Metadata::ShadingCorrection(v)
            | Metadata::NoiseReduction(v) => Some(match v {
                0 => "Off",
                1 => "On",
                _ => return None,
            }),
            Metadata::SubjectDistanceRange(v) => Some(match v {
                0 => "Unknown",
                1 => "Macro",
                2 => "Close",
                3 => "Distant",
                _ => return None,
            }),
            Metadata::YCbCrPositioning(v) => Some(match v {
                1 => "Centered",
                2 => "Co-sited",
                _ => return None,
            }),
            Metadata::PhotometricInterpretation(v) => Some(match v {
                0 => "WhiteIsZero",
                1 => "BlackIsZero",
                2 => "RGB",
                3 => "Palette",
                5 => "CMYK",
                6 => "YCbCr",
                8 => "CIELab",
                32803 => "CFA",
                34892 => "Linear Raw",
                _ => return None,
            }),
            Metadata::PlanarConfiguration(v) => Some(match v {
                1 => "Chunky",
                2 => "Planar",
                _ => return None,
            }),
            Metadata::Predictor(v) => Some(match v {
                1 => "None",
                2 => "Horizontal Diff",
                3 => "Float",
                _ => return None,
            }),
            Metadata::ResolutionUnit(v) => Some(match v {
                1 => "None",
                2 => "inches",
                3 => "cm",
                _ => return None,
            }),
            Metadata::CompositeImage(v) => Some(match v {
                0 => "Unknown",
                1 => "Not Composite",
                2 => "General Composite",
                3 => "Shot While Capturing",
                _ => return None,
            }),
            Metadata::SelfTimerMode(v) => Some(match v {
                0 => "Off",
                1 => "10s",
                2 => "2s",
                _ => return None,
            }),
            Metadata::SensitivityType(v) => Some(match v {
                0 => "Unknown",
                1 => "Std Output Sens.",
                2 => "Rec. Exposure Index",
                3 => "ISO Speed",
                4 => "SOS + REI",
                5 => "SOS + ISO",
                6 => "REI + ISO",
                7 => "SOS + REI + ISO",
                _ => return None,
            }),
            Metadata::GPSStatus(v) => Some(match v.as_str() {
                "A" => "Active",
                "V" => "Void",
                _ => return None,
            }),
            Metadata::GPSMeasureMode(v) => Some(match v.as_str() {
                "2" => "2D",
                "3" => "3D",
                _ => return None,
            }),
            Metadata::SubfileType(v) => Some(match v {
                0 => "Full-res",
                1 => "Reduced-res",
                2 => "Single Page",
                3 => "Reduced Page",
                4 => "Transparency Mask",
                _ => return None,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_orientation() {
        assert_eq!(Metadata::Orientation(6).decode(), Some("Rotate 90° CW"));
    }
    #[test]
    fn decode_flash() {
        assert_eq!(Metadata::Flash(0).decode(), Some("No Flash"));
        assert_eq!(Metadata::Flash(1).decode(), Some("Fired, No Return"));
    }
    #[test]
    fn decode_metering() {
        assert_eq!(Metadata::MeteringMode(3).decode(), Some("Spot"));
        assert_eq!(Metadata::MeteringMode(5).decode(), Some("Multi-segment"));
    }
    #[test]
    fn decode_sensitivity_type() {
        assert_eq!(Metadata::SensitivityType(3).decode(), Some("ISO Speed"));
        assert_eq!(
            Metadata::SensitivityType(7).decode(),
            Some("SOS + REI + ISO")
        );
    }
    #[test]
    fn new_tags_roundtrip() {
        assert_eq!(Metadata::PageNumber(1, 5).value_str(), "1/5");
        assert_eq!(Metadata::AmbientTemperature(23.5).value_str(), "23.5 °C");
        assert_eq!(
            Metadata::CompositeImageCount([4, 2]).value_str(),
            "src=4 used=2"
        );
        assert_eq!(Metadata::GPSVersionID([2, 2, 0, 0]).value_str(), "2.2.0.0");
        assert_eq!(Metadata::GpsAltitudeRef(0).label(), "GPS Alt. Ref");
    }
    #[test]
    fn custom_value_passthrough() {
        let m = Metadata::Custom {
            key: "foo".into(),
            value: "bar".into(),
        };
        assert_eq!(m.label(), "foo");
        assert_eq!(m.value_str(), "bar");
    }
}
