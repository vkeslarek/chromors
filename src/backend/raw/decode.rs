use std::ffi::{CString, c_char};
use std::sync::Arc;

use crate::error::Error;
use crate::libraw_ffi as raw;

use super::handle::{GpsInfo, LensInfo, RawHandle, RawMeta, RawPixels, RawSource, RawThumb};
use super::params::{ProcessWarnings, RawDecodeParams, WhiteBalanceSource};

// ── Public entry points ────────────────────────────────────────────────────────

/// Open a RAW source (file or buffer), unpack it, extract metadata and
/// thumbnail — but do NOT run demosaic. Call `materialize()` for pixels.
pub(crate) fn open_raw_source(
    source: Arc<RawSource>,
    params: RawDecodeParams,
) -> Result<RawHandle, Error> {
    unsafe {
        let ptr = source.open_ptr()?;

        let _ = raw::libraw_unpack_thumb(ptr);

        let rc = raw::libraw_unpack(ptr);
        if rc != 0 {
            raw::libraw_close(ptr);
            return Err(libraw_error(rc));
        }

        let meta = Arc::new(extract_meta(ptr));
        let thumb = extract_thumb(ptr).map(Arc::new);

        Ok(RawHandle {
            source,
            params,
            meta,
            ptr,
            thumb,
            pixels: None,
        })
    }
}

/// Run the demosaic + colour pipeline.  Idempotent.  On a clone (ptr = null),
/// re-opens from `handle.source` before processing.
///
/// Returns a shared `Arc<RawFrame>` that the caller can inspect and hold
/// independently.  Subsequent calls with the same params return the same Arc.
/// Changing params via an operation clears `handle.pixels`, so the next call
/// re-decodes.
pub(crate) fn materialize(handle: &mut RawHandle) -> Result<Arc<super::handle::RawFrame>, Error> {
    if let Some(existing) = &handle.pixels {
        return Ok(Arc::clone(existing));
    }

    let params = handle.params.clone();

    let frame = unsafe {
        let ptr = if !handle.ptr.is_null() {
            handle.ptr
        } else {
            let ptr = handle.source.open_ptr()?;
            let rc = raw::libraw_unpack(ptr);
            if rc != 0 {
                raw::libraw_close(ptr);
                return Err(libraw_error(rc));
            }
            handle.ptr = ptr;
            ptr
        };

        configure_and_process(ptr, &params)?;

        let mut err = 0i32;
        let processed = raw::libraw_dcraw_make_mem_image(ptr, &mut err);
        if processed.is_null() {
            let msg = if err != 0 {
                libraw_error(err).to_string()
            } else {
                "libraw_dcraw_make_mem_image returned null".into()
            };
            return Err(Error::Raw(msg));
        }

        let proc = &*processed;
        let warnings = ProcessWarnings((*ptr).process_warnings);

        let gamma_is_linear =
            (params.gamma_power - 1.0).abs() < 0.01 && (params.gamma_slope - 1.0).abs() < 0.01;

        let pixels = Arc::new(RawPixels {
            ptr: processed,
            width: proc.width as u32,
            height: proc.height as u32,
            colors: proc.colors,
            bits: proc.bits,
            warnings,
        });

        Arc::new(super::handle::RawFrame {
            width: pixels.width,
            height: pixels.height,
            colors: pixels.colors,
            bits: pixels.bits,
            warnings: pixels.warnings,
            pixels,
            output_color: params.output_color,
            gamma_is_linear,
            meta: Arc::clone(&handle.meta),
        })
    };

    unsafe {
        raw::libraw_close(handle.ptr);
        handle.ptr = std::ptr::null_mut();
    }

    handle.pixels = Some(Arc::clone(&frame));
    Ok(frame)
}

// ── Metadata extraction ────────────────────────────────────────────────────────

/// # Safety
/// `ptr` must be valid and `libraw_unpack` must have completed.
unsafe fn extract_meta(ptr: *mut raw::libraw_data_t) -> RawMeta {
    // SAFETY: caller guarantees ptr is valid after libraw_unpack.
    let (idata, sizes, other, color, lens) = unsafe {
        (
            &(*ptr).idata,
            &(*ptr).sizes,
            &(*ptr).other,
            &(*ptr).color,
            &(*ptr).lens,
        )
    };

    let make = cstr_arr(idata.make.as_ref());
    let model = cstr_arr(idata.model.as_ref());
    let normalized_make = cstr_arr(idata.normalized_make.as_ref());
    let normalized_model = cstr_arr(idata.normalized_model.as_ref());
    let software = cstr_arr(idata.software.as_ref());
    let cdesc = cstr_arr(idata.cdesc.as_ref());
    let description = cstr_arr(other.desc.as_ref());
    let artist = cstr_arr(other.artist.as_ref());

    let xmp = unsafe {
        if !idata.xmpdata.is_null() && idata.xmplen > 0 {
            let bytes =
                std::slice::from_raw_parts(idata.xmpdata as *const u8, idata.xmplen as usize);
            Some(bytes.to_vec())
        } else {
            None
        }
    };

    let lens_info = extract_lens(lens);
    let gps = extract_gps(&other.parsed_gps);
    let exif = unsafe { build_exif(ptr, &make, &model, &description, &artist, other) };

    RawMeta {
        make,
        model,
        normalized_make,
        normalized_model,
        software,
        raw_count: idata.raw_count,
        dng_version: idata.dng_version,
        is_foveon: idata.is_foveon != 0,
        filters: idata.filters,
        cdesc,
        raw_width: sizes.raw_width as u32,
        raw_height: sizes.raw_height as u32,
        raw_pitch: sizes.raw_pitch,
        pixel_aspect: sizes.pixel_aspect,
        flip: sizes.flip,
        iso: other.iso_speed,
        shutter: other.shutter,
        aperture: other.aperture,
        focal_len: other.focal_len,
        timestamp: other.timestamp,
        shot_order: other.shot_order,
        description,
        artist,
        analog_balance: other.analogbalance,
        cam_mul: color.cam_mul,
        pre_mul: color.pre_mul,
        black: color.black,
        maximum: color.maximum,
        raw_bps: color.raw_bps,
        cam_xyz: color.cam_xyz.map(|r| [r[0], r[1], r[2]]),
        rgb_cam: color.rgb_cam,
        lens: lens_info,
        gps,
        exif,
        xmp,
    }
}

/// All fields accessed via a reference — fully safe.
fn extract_lens(lens: &raw::libraw_lensinfo_t) -> LensInfo {
    LensInfo {
        lens_make: cstr_arr(lens.LensMake.as_ref()),
        lens: cstr_arr(lens.Lens.as_ref()),
        lens_serial: cstr_arr(lens.LensSerial.as_ref()),
        focal_length_35mm: lens.FocalLengthIn35mmFormat,
        min_focal: lens.MinFocal,
        max_focal: lens.MaxFocal,
        max_ap4_min_focal: lens.MaxAp4MinFocal,
        max_ap4_max_focal: lens.MaxAp4MaxFocal,
        max_ap: lens.makernotes.MaxAp,
        min_ap: lens.makernotes.MinAp,
        cur_focal: lens.makernotes.CurFocal,
        cur_ap: lens.makernotes.CurAp,
    }
}

fn extract_gps(gps: &raw::libraw_gps_info_t) -> Option<GpsInfo> {
    if gps.gpsparsed == 0 {
        return None;
    }
    Some(GpsInfo {
        latitude: gps.latitude,
        lat_ref: if gps.latref != 0 {
            gps.latref as u8 as char
        } else {
            'N'
        },
        longitude: gps.longitude,
        lon_ref: if gps.longref != 0 {
            gps.longref as u8 as char
        } else {
            'E'
        },
        altitude: gps.altitude,
        utc_time: gps.gpstimestamp,
    })
}

/// # Safety
/// `ptr` must be valid.
unsafe fn build_exif(
    ptr: *mut raw::libraw_data_t,
    make: &str,
    model: &str,
    description: &str,
    artist: &str,
    other: &raw::libraw_imgother_t,
) -> Vec<(String, String)> {
    let (idata, sizes) = unsafe { (&(*ptr).idata, &(*ptr).sizes) };
    let mut e = Vec::new();
    push_exif(&mut e, "exif-ifd0-Make", make);
    push_exif(&mut e, "exif-ifd0-Model", model);
    push_exif(&mut e, "exif-ifd0-ImageDescription", description);
    push_exif(&mut e, "exif-ifd0-Artist", artist);
    push_exif(
        &mut e,
        "exif-ifd0-Software",
        &cstr_arr(idata.software.as_ref()),
    );
    push_exif(
        &mut e,
        "exif-ifd2-PixelXDimension",
        &sizes.width.to_string(),
    );
    push_exif(
        &mut e,
        "exif-ifd2-PixelYDimension",
        &sizes.height.to_string(),
    );
    if other.iso_speed > 0.0 {
        push_exif(
            &mut e,
            "exif-ifd2-ISOSpeedRatings",
            &(other.iso_speed as i32).to_string(),
        );
    }
    if other.shutter > 0.0 {
        push_exif(
            &mut e,
            "exif-ifd2-ExposureTime",
            &fmt_shutter(other.shutter),
        );
    }
    if other.aperture > 0.0 {
        push_exif(
            &mut e,
            "exif-ifd2-FNumber",
            &format!("{:.1}", other.aperture),
        );
    }
    if other.focal_len > 0.0 {
        push_exif(
            &mut e,
            "exif-ifd2-FocalLength",
            &format!("{:.1}", other.focal_len),
        );
    }
    if other.timestamp > 0 {
        push_exif(
            &mut e,
            "exif-ifd2-DateTimeOriginal",
            &fmt_timestamp(other.timestamp),
        );
    }
    if let Some(gps) = extract_gps(&other.parsed_gps) {
        let (lat, lon) = (&gps.latitude, &gps.longitude);
        push_exif(
            &mut e,
            "exif-gps-GPSLatitude",
            &format!(
                "{} deg {}' {:.2}\" {}",
                lat[0] as i32, lat[1] as i32, lat[2], gps.lat_ref
            ),
        );
        push_exif(
            &mut e,
            "exif-gps-GPSLongitude",
            &format!(
                "{} deg {}' {:.2}\" {}",
                lon[0] as i32, lon[1] as i32, lon[2], gps.lon_ref
            ),
        );
        if gps.altitude.abs() > 0.0 {
            push_exif(
                &mut e,
                "exif-gps-GPSAltitude",
                &format!("{:.1}", gps.altitude),
            );
        }
        let ts = &gps.utc_time;
        if ts[0] > 0.0 || ts[1] > 0.0 || ts[2] > 0.0 {
            push_exif(
                &mut e,
                "exif-gps-GPSTimeStamp",
                &format!("{}:{}:{:.2}", ts[0] as i32, ts[1] as i32, ts[2]),
            );
        }
    }
    e
}

// ── Thumbnail extraction ───────────────────────────────────────────────────────

/// # Safety
/// `ptr` must be valid and `libraw_unpack_thumb` must have been called.
unsafe fn extract_thumb(ptr: *mut raw::libraw_data_t) -> Option<RawThumb> {
    let (thumb_ptr, t) = unsafe {
        let mut err = 0i32;
        let thumb_ptr = raw::libraw_dcraw_make_mem_thumb(ptr, &mut err);
        if thumb_ptr.is_null() || err != 0 {
            return None;
        }
        let t = &*thumb_ptr;
        (thumb_ptr, t)
    };
    if t.type_ != raw::LibRaw_image_formats_LIBRAW_IMAGE_JPEG || t.data_size == 0 {
        unsafe {
            raw::libraw_dcraw_clear_mem(thumb_ptr);
        }
        return None;
    }
    Some(RawThumb { ptr: thumb_ptr })
}

// ── Parameter configuration + process ─────────────────────────────────────────

/// Configure all output parameters AND run `libraw_dcraw_process` in a single
/// stack frame so CStrings for ICC/bad-pixel paths remain alive throughout.
///
/// # Safety
/// `ptr` must be a valid, unpacked `libraw_data_t`.
unsafe fn configure_and_process(
    ptr: *mut raw::libraw_data_t,
    params: &RawDecodeParams,
) -> Result<(), Error> {
    unsafe {
        configure_params(ptr, params);
    }

    // Build CStrings HERE so they remain alive through dcraw_process.
    let _op = params
        .output_profile
        .as_deref()
        .and_then(|s| CString::new(s).ok());
    let _cp = params
        .camera_profile
        .as_deref()
        .and_then(|s| CString::new(s).ok());
    let _bp = params
        .bad_pixels
        .as_deref()
        .and_then(|s| CString::new(s).ok());
    let _df = params
        .dark_frame
        .as_deref()
        .and_then(|s| CString::new(s).ok());

    unsafe {
        let p = &mut (*ptr).params;
        if let Some(cs) = &_op {
            p.output_profile = cs.as_ptr() as *mut _;
        }
        if let Some(cs) = &_cp {
            p.camera_profile = cs.as_ptr() as *mut _;
        }
        if let Some(cs) = &_bp {
            p.bad_pixels = cs.as_ptr() as *mut _;
        }
        if let Some(cs) = &_df {
            p.dark_frame = cs.as_ptr() as *mut _;
        }

        let rc = raw::libraw_dcraw_process(ptr);
        // CStrings still alive here.
        if rc != 0 {
            return Err(libraw_error(rc));
        }
    }
    Ok(())
}

/// Set all `libraw_output_params_t` fields from `RawDecodeParams`.
/// ICC profile paths are NOT set here — `configure_and_process` handles their
/// lifetimes.
///
/// # Safety
/// `ptr` must be a valid `libraw_data_t`.
unsafe fn configure_params(ptr: *mut raw::libraw_data_t, params: &RawDecodeParams) {
    unsafe {
        raw::libraw_set_output_color(ptr, params.output_color as i32);
        raw::libraw_set_output_bps(ptr, params.output_bps as i32);
        raw::libraw_set_output_tif(ptr, params.output_tiff as i32);
        raw::libraw_set_demosaic(ptr, params.interpolation as i32);
        raw::libraw_set_fbdd_noiserd(ptr, params.fbdd_noiserd as i32);
        raw::libraw_set_highlight(ptr, params.highlights as i32);
        raw::libraw_set_no_auto_bright(ptr, params.no_auto_bright as i32);
        raw::libraw_set_bright(ptr, params.bright);
        raw::libraw_set_adjust_maximum_thr(ptr, params.adjust_maximum_thr);

        if let Some(curve) = &params.gamma_curve {
            for (i, &v) in curve.iter().enumerate().take(6) {
                raw::libraw_set_gamma(ptr, i as i32, v as f32);
            }
        } else {
            raw::libraw_set_gamma(ptr, 0, params.gamma_power as f32);
            raw::libraw_set_gamma(ptr, 1, params.gamma_slope as f32);
        }

        let p = &mut (*ptr).params;

        match params.white_balance {
            WhiteBalanceSource::Camera => {
                p.use_camera_wb = 1;
                p.use_auto_wb = 0;
            }
            WhiteBalanceSource::Auto => {
                p.use_camera_wb = 0;
                p.use_auto_wb = 1;
            }
            WhiteBalanceSource::Custom(mul) => {
                p.use_camera_wb = 0;
                p.use_auto_wb = 0;
                for (i, &v) in mul.iter().enumerate().take(4) {
                    raw::libraw_set_user_mul(ptr, i as i32, v);
                }
            }
            WhiteBalanceSource::None => {
                p.use_camera_wb = 0;
                p.use_auto_wb = 0;
            }
        }

        p.use_camera_matrix = params.camera_matrix as i32;
        p.user_sat = params.user_sat;
        p.half_size = params.half_size as i32;
        p.four_color_rgb = params.four_color_rgb as i32;
        p.green_matching = params.green_matching as i32;
        p.dcb_iterations = params.dcb_iterations;
        p.dcb_enhance_fl = params.dcb_enhance as i32;
        p.auto_bright_thr = params.auto_bright_thr;
        p.threshold = params.threshold;
        p.med_passes = params.med_passes;
        p.output_flags = params.output_flags;

        if params.exp_correc {
            p.exp_correc = 1;
            p.exp_shift = params.exp_shift;
            p.exp_preser = params.exp_preser;
        } else {
            p.exp_correc = 0;
        }

        p.user_flip = params.user_flip;
        p.use_fuji_rotate = if params.use_fuji_rotate { -1 } else { 0 };
        p.no_auto_scale = params.no_auto_scale as i32;
        p.no_interpolation = params.no_interpolation as i32;
        p.user_black = params.user_black;
        p.user_cblack = params.user_cblack;

        if let WhiteBalanceSource::Custom(mul) = params.white_balance {
            p.user_mul = mul;
        }
        if let Some(grey) = &params.greybox {
            p.greybox = *grey;
        }
        if let Some(crop) = &params.cropbox {
            p.cropbox = *crop;
        }
        if let Some(aber) = &params.aber {
            p.aber = *aber;
        }

        let rp = &mut (*ptr).rawparams;
        rp.shot_select = params.shot_select;
        rp.max_raw_memory_mb = params.max_raw_memory_mb;
        rp.sony_arw2_posterization_thr = params.sony_arw2_posterization_thr;
        rp.use_rawspeed = params.use_rawspeed;
        rp.use_dngsdk = params.use_dngsdk;
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

pub(crate) fn libraw_error(code: i32) -> Error {
    unsafe {
        let p = raw::libraw_strerror(code);
        let msg = if p.is_null() {
            format!("libraw error {code}")
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        };
        Error::Raw(msg)
    }
}

/// Safe conversion: stops at the first null byte even if the slice continues.
pub(crate) fn cstr_arr(arr: &[c_char]) -> String {
    let bytes: Vec<u8> = arr
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

pub(crate) fn push_exif(entries: &mut Vec<(String, String)>, key: &str, value: &str) {
    if !value.is_empty() {
        entries.push((key.to_string(), value.to_string()));
    }
}

fn fmt_shutter(shutter: f32) -> String {
    if shutter >= 1.0 {
        format!("{:.1}s", shutter)
    } else {
        format!("1/{}s", (1.0 / shutter).round() as i32)
    }
}

fn fmt_timestamp(ts: std::os::raw::c_long) -> String {
    if ts <= 0 {
        return String::new();
    }
    let total_days = ts / 86400;
    let time_secs = ts % 86400;
    let (h, m, s) = (time_secs / 3600, (time_secs % 3600) / 60, time_secs % 60);
    let mut year = 1970i64;
    let mut rem = total_days;
    loop {
        let dy = if is_leap(year) { 366 } else { 365 };
        if rem < dy {
            break;
        }
        rem -= dy;
        year += 1;
    }
    let mdays = if is_leap(year) {
        [31i64, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut month = 1usize;
    for (i, &md) in mdays.iter().enumerate() {
        if rem < md {
            month = i + 1;
            break;
        }
        rem -= md;
        if i == 11 {
            month = 12;
        }
    }
    format!("{year:04}:{month:02}:{:02} {h:02}:{m:02}:{s:02}", rem + 1)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}
