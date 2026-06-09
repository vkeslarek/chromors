//! Typed value vocabulary for the GPU computation graph.
//!
//! * [`ValueKind`] — the shape tag stored on each [`super::graph::GraphNode`].
//!   Describes what kind of data the node produces, without carrying the payload.
//!   Replaces the old `NodeOutputKind` type.
//!
//! * [`GraphValue`] — the runtime payload produced by materializing a node.
//!   Replaces the old `MaterializedBuffer` type.

use std::sync::Arc;

use crate::geometry::Rect;
use crate::pixel::PixelFormat;

use super::buffer::ImageBuffer;

// ── ValueKind ─────────────────────────────────────────────────────────────────

/// Shape tag for a graph node's output.
///
/// Carries only the structural information needed to allocate buffers and
/// emit shader code — no runtime payload.
#[derive(Clone, Debug, PartialEq)]
pub enum ValueKind {
    /// A 2-D pixel image. Gray images are represented here with a Gray
    /// [`crate::pixel::PixelFormat`] — there is no separate Gray variant.
    Image,
    /// Fixed-size histogram accumulator. `bins` × u32 atomic counters.
    Histogram { bins: u32 },
    /// Atomic-append coordinate list. Counter at offset 0, then (x, y) pairs.
    PointList { capacity: u32 },
    /// Single float scalar output.
    Scalar,
    /// Multi-channel feature map. Storage = `width × height × ceil(channels/4) × 16` bytes.
    Features { channels: u32 },
    /// 1-D Mask (e.g. for separable convolution)
    Mask1D { length: u32 },
    /// 2-D Mask (e.g. for morph/compass)
    Mask2D { width: u32, height: u32 },
    /// 1-D FFT result (frequency domain)
    Fft1D { length: u32 },
    /// 2-D FFT result (frequency domain image)
    Fft2D,
}

impl ValueKind {
    /// Byte size of the GPU output buffer for a node of this kind.
    ///
    /// `w` and `h` are the output rect dimensions in pixels.
    /// `image_format` is only consulted for the [`ValueKind::Image`] variant.
    pub fn output_byte_size(&self, w: u32, h: u32, image_format: PixelFormat) -> u64 {
        match self {
            ValueKind::Image => {
                let bpp = image_format.bytes_per_pixel() as u64;
                (w as u64 * h as u64 * bpp).max(64)
            }
            ValueKind::Histogram { bins } => (*bins as u64 * 4).max(64),
            ValueKind::PointList { capacity } => (4 + *capacity as u64 * 8).max(64),
            ValueKind::Scalar => 64,
            ValueKind::Features { channels } => {
                (w as u64 * h as u64 * channels.div_ceil(4) as u64 * 16).max(64)
            }
            ValueKind::Mask1D { length } => (*length as u64 * 4).max(64), // Assuming f32 masks
            ValueKind::Mask2D { width, height } => (*width as u64 * *height as u64 * 4).max(64),
            ValueKind::Fft1D { length } => (*length as u64 * 8).max(64), // Assuming complex f32 (8 bytes)
            ValueKind::Fft2D => {
                // Complex f32 (8 bytes per pixel)
                (w as u64 * h as u64 * 8).max(64)
            }
        }
    }

    /// Returns `true` if this node kind needs a float4 `RWRegion` intermediate
    /// temp buffer in the fused shader.  Non-image outputs (histograms, scalars,
    /// …) write directly to their target and do not get a temp.
    #[inline]
    pub fn needs_fused_temp(&self) -> bool {
        matches!(self, ValueKind::Image)
    }

    /// How the emitter wraps writes to a buffer of this kind.
    ///
    /// Single source of truth for the "is this an atomic-accumulator output"
    /// question — the emitter previously re-derived it at four call sites via
    /// `matches!(kind, ValueKind::Histogram { .. })`. New atomic-accumulate
    /// kinds (e.g. a future reduction-shaped `Scalar`/`PointList` use) plug in
    /// here once, instead of adding a fifth scattered match arm.
    pub fn write_mode(&self) -> WriteMode {
        match self {
            ValueKind::Histogram { bins } => WriteMode::AtomicAccumulate { count: *bins },
            _ => WriteMode::Positional,
        }
    }
}

/// How a kernel writes its result into a target buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteMode {
    /// One write per dispatched thread, addressed by `region_index` —
    /// `target.write(idx, value)` / `from_working` + `codec::encode`.
    /// The default for spatial outputs (Image, FeatureMap, masks, FFTs).
    Positional,
    /// Scattered atomic increments into a fixed-size counter buffer —
    /// `HistogramOut { target, bin_count }`. No per-thread positional write,
    /// no `region_target` descriptor; `count` sizes the `bin_count` param.
    AtomicAccumulate { count: u32 },
}

// ── GraphValue ────────────────────────────────────────────────────────────────

/// Runtime payload produced when a graph node is materialised.
///
/// Replaces the old `MaterializedBuffer { Image | Raw }` binary.  For now the
/// non-image variants still carry raw bytes (same as the old `Raw` arm); typed
/// variants (`Histogram { data: Vec<u32> }` etc.) will be introduced in Phase B.
#[derive(Clone)]
pub enum GraphValue {
    /// Pixel data in VRAM, with two coordinate frames.
    ///
    /// * `buffer_rect` — where valid pixels sit *inside* `buffer` (buffer-local).
    /// * `source_rect` — which rect of the full image those pixels represent.
    ///
    /// Invariant: dimensions of the two rects are always equal.
    Image {
        buffer: Arc<ImageBuffer>,
        buffer_rect: Rect,
        source_rect: Rect,
    },
    /// Raw host-side bytes for any non-image output (histogram, scalar, …).
    /// Typed variants will replace this in Phase B.
    Raw {
        bytes: Vec<u8>,
        kind: ValueKind,
        source_rect: Rect,
    },
}

impl GraphValue {
    // ── Constructors (mirrors the old MaterializedBuffer helpers) ─────────────

    pub fn image(buffer: Arc<ImageBuffer>, source_rect: Rect) -> Self {
        let buffer_rect = Rect::new(0, 0, source_rect.width, source_rect.height);
        GraphValue::Image {
            buffer,
            buffer_rect,
            source_rect,
        }
    }

    pub fn raw(bytes: Vec<u8>, kind: ValueKind, source_rect: Rect) -> Self {
        GraphValue::Raw {
            bytes,
            kind,
            source_rect,
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn buffer_coords(&self, image_rect: Rect) -> Rect {
        match self {
            GraphValue::Image {
                buffer_rect,
                source_rect,
                ..
            } => Rect::new(
                image_rect.x - source_rect.x + buffer_rect.x,
                image_rect.y - source_rect.y + buffer_rect.y,
                image_rect.width,
                image_rect.height,
            ),
            GraphValue::Raw { source_rect, .. } => Rect::new(
                image_rect.x - source_rect.x,
                image_rect.y - source_rect.y,
                image_rect.width,
                image_rect.height,
            ),
        }
    }

    pub fn byte_size(&self) -> u64 {
        match self {
            GraphValue::Image { buffer, .. } => buffer.total_bytes(),
            GraphValue::Raw { bytes, .. } => bytes.len() as u64,
        }
    }

    pub fn read_bytes(
        &self,
        ctx: &crate::backend::gpu::context::GpuContext,
    ) -> Result<Vec<u8>, crate::error::Error> {
        match self {
            GraphValue::Image { buffer, .. } => buffer.read_to_cpu(ctx),
            GraphValue::Raw { bytes, .. } => Ok(bytes.clone()),
        }
    }
}
