//! Capability trait for backends usable by AI inference models.
//!
//! All AI models follow the same I/O contract:
//!   1. `.convert(layout)` — normalize to a fixed pixel format
//!   2. `.pull(&RamImageTarget, wu)` — extract as raw bytes
//!   3. run ONNX inference on those bytes
//!   4. reconstruct `Image2D<B>` / `Mask2D<B>` from output bytes
//!
//! Any backend implementing `AiBackend` supports steps 1–2 generically
//! (they rely on existing core traits). Steps 4's constructors (`image_from_bytes`,
//! `mask_from_values`) are the only backend-specific additions.

use std::sync::Arc;
use chromors::data::image::Image2D;
use chromors::data::mask2d::Mask2D;
use chromors::pixel::PixelLayout;
use chromors::error::Error;
use chromors::Backend;

/// A backend that can round-trip image and mask data through raw bytes —
/// the minimum required by all chromors-ai inference models.
pub trait AiBackend: Backend + Sized + 'static {
    /// Construct an `Image2D` from raw packed bytes in `layout` format.
    fn image_from_bytes(bytes: Vec<u8>, w: i32, h: i32, layout: PixelLayout) -> Image2D<Self>;

    /// Construct a `Mask2D` from normalized f32 values (0..1).
    fn mask_from_values(values: &[f32], w: i32, h: i32) -> Mask2D<Self>;
}

// ── VipsBackend impl ─────────────────────────────────────────────────────────

use chromors::backend::vips::VipsBackend;
use chromors::VipsImageExt;
use chromors::VipsMask2DExt;

impl AiBackend for VipsBackend {
    fn image_from_bytes(bytes: Vec<u8>, w: i32, h: i32, layout: PixelLayout) -> Image2D<Self> {
        Image2D::<VipsBackend>::from_bytes(bytes, w, h, layout)
    }

    fn mask_from_values(values: &[f32], w: i32, h: i32) -> Mask2D<Self> {
        Mask2D::<VipsBackend>::from_values(w, h, values)
    }
}
