//! Embedded custom vips operations.
//!
//! A [`VipsCustomOperation`] is a Rust pixel operation that runs **inside** the
//! libvips demand-driven pipeline via `vips_image_generate`: vips asks for
//! output regions on demand, we prepare the matching input region and call the
//! Rust `generate` callback. No full-image download — work happens region by
//! region, lazily, like any native vips operation.
//!
//! ```ignore
//! struct AddConst { k: f32 }
//! impl VipsCustomOperation for AddConst {
//!     fn generate(&self, out: &mut CustomRegion, input: &CustomRegion) -> Result<(), Error> {
//!         let (_, top, _, h) = out.rect();
//!         for y in top..top + h {
//!             let src = input.row(y);
//!             let dst = out.row_mut(y);
//!             for (d, s) in dst.iter_mut().zip(src) { *d = s.saturating_add(self.k as u8); }
//!         }
//!         Ok(())
//!     }
//! }
//! let out = img.custom(AddConst { k: 10.0 })?;
//! ```

use std::ffi::{c_char, c_int, c_void};
use std::slice;

use super::FromVipsBandFormat;
use crate::error::Error;
use crate::ffi;
use crate::pixel::Storage;

// glib's data-with-destructor attach — not in the generated bindings, declared
// here (links against the already-linked glib).
unsafe extern "C" {
    fn g_object_set_data_full(
        object: *mut ffi::GObject,
        key: *const c_char,
        data: *mut c_void,
        destroy: Option<unsafe extern "C" fn(*mut c_void)>,
    );
}

/// A region of an image, exposed to a custom op as raw rows. Coordinates are
/// absolute image pixels; `rect()` gives the valid window this region covers.
pub struct CustomRegion {
    ptr: *mut ffi::VipsRegion,
    psize: usize,
    storage: Storage,
    bands: i32,
}

impl CustomRegion {
    unsafe fn new(ptr: *mut ffi::VipsRegion) -> Self {
        let im = unsafe { (*ptr).im };
        let bands = unsafe { ffi::vips_image_get_bands(im) };
        let storage =
            Storage::from_vips_band_format(unsafe { ffi::vips_image_get_format(im) }, bands);
        CustomRegion {
            ptr,
            psize: storage.bytes_per_sample() * bands as usize,
            storage,
            bands,
        }
    }

    /// Valid window as `(left, top, width, height)` in image pixels.
    pub fn rect(&self) -> (i32, i32, i32, i32) {
        let v = unsafe { (*self.ptr).valid };
        (v.left, v.top, v.width, v.height)
    }

    /// Sample quantization of this region.
    pub fn storage(&self) -> Storage {
        self.storage
    }

    /// Band (channel) count of this region.
    pub fn bands(&self) -> i32 {
        self.bands
    }

    /// Bytes per pixel for this region's storage/band count.
    pub fn pixel_bytes(&self) -> usize {
        self.psize
    }

    fn row_ptr(&self, y: i32) -> (*mut u8, usize) {
        let v = unsafe { (*self.ptr).valid };
        debug_assert!(y >= v.top && y < v.top + v.height, "row {y} outside valid");
        let bpl = unsafe { (*self.ptr).bpl } as usize;
        let data = unsafe { (*self.ptr).data };
        let off = (y - v.top) as usize * bpl;
        let len = v.width as usize * self.psize;
        (unsafe { data.add(off) }, len)
    }

    /// Read-only row `y` (absolute), `width * pixel_bytes` bytes.
    pub fn row(&self, y: i32) -> &[u8] {
        let (p, len) = self.row_ptr(y);
        unsafe { slice::from_raw_parts(p, len) }
    }

    /// Mutable row `y` (absolute) — only valid for the output region.
    pub fn row_mut(&mut self, y: i32) -> &mut [u8] {
        let (p, len) = self.row_ptr(y);
        unsafe { slice::from_raw_parts_mut(p, len) }
    }

    /// Read-only row `y` as typed pixel slice `&[P]`.
    pub fn pixels<P: bytemuck::Pod>(&self, y: i32) -> &[P] {
        bytemuck::cast_slice(self.row(y))
    }

    /// Mutable row `y` as typed pixel slice `&mut [P]`.
    pub fn pixels_mut<P: bytemuck::NoUninit + bytemuck::AnyBitPattern>(
        &mut self,
        y: i32,
    ) -> &mut [P] {
        bytemuck::cast_slice_mut(self.row_mut(y))
    }
}

/// A custom pixel operation embedded in the vips pipeline.
///
/// The output has the same geometry/format as the input (a `copy`-style
/// pipeline). `generate` is called per output region; `input` is prepared to
/// exactly the output's valid rect.
pub trait VipsCustomOperation: Send + Sync + 'static {
    fn generate(&self, out: &mut CustomRegion, input: &CustomRegion) -> Result<(), Error>;
}

/// Wires a [`VipsCustomOperation`] into the vips pipeline: creates a new
/// output image with the same geometry/format/demand hint as `input` and
/// hooks `op.generate` up as its region generator via `vips_image_generate`.
/// Lazy and region-driven, like any native vips operation — no full-image
/// download.
pub fn run_custom<O: VipsCustomOperation>(
    input: &super::VipsHandle,
    op: O,
) -> Result<super::VipsHandle, Error> {
    unsafe {
        let out = ffi::vips_image_new();
        if out.is_null() {
            return Err(Error::Backend(super::vips_error()));
        }
        if ffi::vips_image_pipelinev(
            out,
            ffi::VipsDemandStyle_VIPS_DEMAND_STYLE_THINSTRIP,
            input.ptr,
            std::ptr::null_mut::<ffi::VipsImage>(),
        ) != 0
        {
            ffi::g_object_unref(out as ffi::gpointer);
            return Err(Error::Backend(super::vips_error()));
        }

        ffi::g_object_ref(input.ptr as ffi::gpointer);
        let holder = Box::new(CustomHolder {
            op: Box::new(op),
            input: input.ptr,
        });
        let holder_ptr = Box::into_raw(holder) as *mut c_void;
        g_object_set_data_full(
            out as *mut ffi::GObject,
            c"chromors-custom-op".as_ptr(),
            holder_ptr,
            Some(drop_holder),
        );

        if ffi::vips_image_generate(
            out,
            Some(ffi::vips_start_one),
            Some(generate_tramp),
            Some(ffi::vips_stop_one),
            input.ptr as *mut c_void,
            holder_ptr,
        ) != 0
        {
            ffi::g_object_unref(out as ffi::gpointer);
            return Err(Error::Backend(super::vips_error()));
        }

        Ok(super::VipsHandle { ptr: out })
    }
}

/// Keeps the boxed op and a ref to the input image alive for the lifetime of
/// the output image. Freed by the glib destroy-notify when `out` is dropped.
struct CustomHolder {
    op: Box<dyn VipsCustomOperation>,
    input: *mut ffi::VipsImage,
}

unsafe extern "C" fn drop_holder(data: *mut c_void) {
    let holder = unsafe { Box::from_raw(data as *mut CustomHolder) };
    unsafe { ffi::g_object_unref(holder.input as ffi::gpointer) };
    drop(holder);
}

unsafe extern "C" fn generate_tramp(
    out: *mut ffi::VipsRegion,
    seq: *mut c_void,
    _a: *mut c_void,
    b: *mut c_void,
    _stop: *mut c_int,
) -> c_int {
    let in_region = seq as *mut ffi::VipsRegion;
    let holder = unsafe { &*(b as *const CustomHolder) };

    // Demand the same rect from the input, then run the Rust callback.
    let valid = unsafe { (*out).valid };
    if unsafe { ffi::vips_region_prepare(in_region, &valid as *const _ as *mut _) } != 0 {
        return -1;
    }
    let mut out_reg = unsafe { CustomRegion::new(out) };
    let in_reg = unsafe { CustomRegion::new(in_region) };
    match holder.op.generate(&mut out_reg, &in_reg) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}

/// A custom **reduction** over an image: scan it region by region and produce
/// an arbitrary Rust value — not an `Image2D` or any other vips object.
///
/// This is how vips' own `avg`/`min`/`stats` work. vips runs regions across its
/// threadpool, so each thread folds into its own [`Acc`](VipsCustomSink::Acc);
/// the per-thread accumulators are `merge`d at the end, then `finish` produces
/// the result. No full-image download — work is region-local.
pub trait VipsCustomSink: Send + Sync + 'static {
    /// The value produced (e.g. `Vec<KeyPoint>`, a histogram, a struct).
    type Output;
    /// Per-thread accumulator. `Default` is the empty/identity state.
    type Acc: Default + Send + 'static;

    /// Fold one region into a thread-local accumulator.
    fn fold(&self, acc: &mut Self::Acc, region: &CustomRegion);
    /// Merge a finished thread accumulator `part` into `total`.
    fn merge(&self, total: &mut Self::Acc, part: Self::Acc);
    /// Reduce the merged accumulator to the final value.
    fn finish(&self, acc: Self::Acc) -> Self::Output;
}

struct SinkState<S: VipsCustomSink> {
    sink: S,
    global: std::sync::Mutex<S::Acc>,
}

unsafe extern "C" fn sink_start<S: VipsCustomSink>(
    _out: *mut ffi::VipsImage,
    _a: *mut c_void,
    _b: *mut c_void,
) -> *mut c_void {
    Box::into_raw(Box::new(S::Acc::default())) as *mut c_void
}

unsafe extern "C" fn sink_generate<S: VipsCustomSink>(
    region: *mut ffi::VipsRegion,
    seq: *mut c_void,
    _a: *mut c_void,
    b: *mut c_void,
    _stop: *mut c_int,
) -> c_int {
    let state = unsafe { &*(b as *const SinkState<S>) };
    let acc = unsafe { &mut *(seq as *mut S::Acc) };
    let valid = unsafe { (*region).valid };
    if unsafe { ffi::vips_region_prepare(region, &valid as *const _ as *mut _) } != 0 {
        return -1;
    }
    let reg = unsafe { CustomRegion::new(region) };
    state.sink.fold(acc, &reg);
    0
}

unsafe extern "C" fn sink_stop<S: VipsCustomSink>(
    seq: *mut c_void,
    _a: *mut c_void,
    b: *mut c_void,
) -> c_int {
    let state = unsafe { &*(b as *const SinkState<S>) };
    let acc = *unsafe { Box::from_raw(seq as *mut S::Acc) };
    let mut g = state.global.lock().unwrap_or_else(|e| e.into_inner());
    let total = std::mem::take(&mut *g);
    let mut total = total;
    state.sink.merge(&mut total, acc);
    *g = total;
    0
}
