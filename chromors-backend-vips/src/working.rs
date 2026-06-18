//! CPU working-space region wrappers.
//!
//! A zero-copy wrapper over [`CustomRegion`] that fetches and decodes pixels
//! on the fly into any [`Pixel`] type, with SIMD support.
//!
//! Example:
//! ```ignore
//! struct MyFilter { amount: f32 }
//!
//! impl RegionProcessor for MyFilter {
//!     fn process<P: Pixel>(&self, src: &RegionView<P>, dst: &mut RegionViewMut<P>) {
//!         let (_, _, w, h) = src.rect();
//!         for y in 0..h {
//!             for x in 0..w {
//!                 let mut pixel: Hsv = src.get(x, y);
//!                 pixel.v *= self.amount;
//!                 dst.set(x, y, pixel); // Preserves alpha automatically if P has alpha!
//!             }
//!         }
//!     }
//! }
//!
//! impl VipsCustomOperation for MyFilter {
//!     fn generate(&self, out: &mut CustomRegion, input: &CustomRegion) -> Result<(), Error> {
//!         execute_processor(input, out, AlphaPolicy::Straight, self)
//!     }
//! }
//! ```

use std::marker::PhantomData;

use super::custom::CustomRegion;
use crate::Error;
use crate::{AlphaPolicy, Pixel};

/// A zero-copy read-only wrapper over a `CustomRegion` for a specific physical
/// storage format `P`. It can decode pixels on the fly into any working pixel `W`.
pub struct RegionView<'a, P: Pixel> {
    pub region: &'a CustomRegion,
    pub policy: AlphaPolicy,
    _marker: PhantomData<P>,
}

impl<'a, P: Pixel> RegionView<'a, P> {
    pub fn new(region: &'a CustomRegion, policy: AlphaPolicy) -> Self {
        Self {
            region,
            policy,
            _marker: PhantomData,
        }
    }

    /// Valid window as `(left, top, width, height)` in image pixels.
    #[inline]
    pub fn rect(&self) -> (i32, i32, i32, i32) {
        self.region.rect()
    }

    /// Fetches a single pixel and decodes it into the working format `W`.
    /// `x` and `y` are relative to the region's valid rect (0..width, 0..height).
    #[inline]
    pub fn get<W: Pixel>(&self, x: i32, y: i32) -> W {
        let (_, top, _, _) = self.region.rect();
        let p = self.region.pixels::<P>(top + y)[x as usize];
        W::pack_one(p.unpack(), self.policy)
    }

    /// Fetches a pixel with edge clamping.
    /// `x` and `y` are relative to the region's valid rect (0..width, 0..height).
    #[inline]
    pub fn get_clamped<W: Pixel>(&self, x: i32, y: i32) -> W {
        let (_, top, w, h) = self.region.rect();
        let xc = x.clamp(0, w - 1);
        let yc = y.clamp(0, h - 1);
        let p = self.region.pixels::<P>(top + yc)[xc as usize];
        W::pack_one(p.unpack(), self.policy)
    }

    /// Fetches 4 pixels and decodes them into the working format `W` using SIMD.
    /// `x` and `y` are relative to the region's valid rect (0..width, 0..height).
    #[inline]
    pub fn get_x4<W: Pixel>(&self, x: i32, y: i32) -> [W; 4] {
        let (_, top, _, _) = self.region.rect();
        let slice = &self.region.pixels::<P>(top + y)[x as usize..x as usize + 4];
        let (r, g, b, a) = P::unpack_x4(slice);
        let mut out = [W::pack_one([0.0; 4], self.policy); 4];
        W::pack_x4(r, g, b, a, self.policy, &mut out);
        out
    }
}

/// A zero-copy mutable wrapper over a `CustomRegion` for a specific physical
/// storage format `P`. It can encode pixels from any working pixel `W` on the fly.
pub struct RegionViewMut<'a, P: Pixel> {
    pub region: &'a mut CustomRegion,
    pub source: Option<&'a CustomRegion>,
    pub policy: AlphaPolicy,
    _marker: PhantomData<P>,
}

impl<'a, P: Pixel> RegionViewMut<'a, P> {
    pub fn new(region: &'a mut CustomRegion, policy: AlphaPolicy) -> Self {
        Self {
            region,
            source: None,
            policy,
            _marker: PhantomData,
        }
    }

    /// Creates a mutable view that will automatically preserve alpha from the given `source` region
    /// when writing opaque working pixels (e.g. `Rgb`, `Hsv`).
    pub fn with_source(
        region: &'a mut CustomRegion,
        source: &'a CustomRegion,
        policy: AlphaPolicy,
    ) -> Self {
        Self {
            region,
            source: Some(source),
            policy,
            _marker: PhantomData,
        }
    }

    /// Valid window as `(left, top, width, height)` in image pixels.
    #[inline]
    pub fn rect(&self) -> (i32, i32, i32, i32) {
        self.region.rect()
    }

    /// Constant-folded heuristic to check if the working model `W` has an alpha channel.
    #[inline]
    fn w_has_alpha<W: Pixel>(policy: AlphaPolicy) -> bool {
        let test = W::pack_one([0.0, 0.0, 0.0, 0.5], policy);
        test.unpack()[3] < 0.99
    }

    /// Encodes a working pixel `W` and writes it to the region.
    /// `x` and `y` are relative to the region's valid rect (0..width, 0..height).
    /// Preserves original alpha if `source` is available and `W` drops alpha.
    #[inline]
    pub fn set<W: Pixel>(&mut self, x: i32, y: i32, v: W) {
        let (_, top, _, _) = self.region.rect();
        let mut unpacked = v.unpack();

        if !Self::w_has_alpha::<W>(self.policy) {
            if let Some(src) = self.source {
                let existing = src.pixels::<P>(top + y)[x as usize].unpack();
                unpacked[3] = existing[3];
            }
        }

        let p = P::pack_one(unpacked, self.policy);
        self.region.pixels_mut::<P>(top + y)[x as usize] = p;
    }

    /// Encodes 4 working pixels `W` using SIMD and writes them to the region.
    /// `x` and `y` are relative to the region's valid rect (0..width, 0..height).
    /// Preserves original alpha if `source` is available and `W` drops alpha.
    #[inline]
    pub fn set_x4<W: Pixel>(&mut self, x: i32, y: i32, v: [W; 4]) {
        let (_, top, _, _) = self.region.rect();
        let (r, g, b, mut a) = W::unpack_x4(&v);

        if !Self::w_has_alpha::<W>(self.policy) {
            if let Some(src) = self.source {
                let slice = &src.pixels::<P>(top + y)[x as usize..x as usize + 4];
                let (_, _, _, src_a) = P::unpack_x4(slice);
                a = src_a;
            }
        }

        let slice = &mut self.region.pixels_mut::<P>(top + y)[x as usize..x as usize + 4];
        P::pack_x4(r, g, b, a, self.policy, slice);
    }
}

/// A trait for processing regions in the working color space.
/// Implementing this allows you to write a generic pixel loop that is automatically
/// dispatched to the physical storage format.
pub trait RegionProcessor {
    fn process<P: Pixel>(&self, src: &RegionView<P>, dst: &mut RegionViewMut<P>);
}

/// Dispatch `$body!(ConcretePixel)` for the region's `(Storage, bands)`, or
/// return an "unsupported format" error.
#[macro_export]
macro_rules! dispatch_format {
    ($storage:expr, $bands:expr, $body:ident) => {
        match ($storage, $bands) {
            ($crate::Storage::U8, 3) => $body!($crate::Rgb<u8>),
            ($crate::Storage::U8, 4) => $body!($crate::Rgba<u8>),
            ($crate::Storage::U16, 3) => $body!($crate::Rgb<u16>),
            ($crate::Storage::U16, 4) => $body!($crate::Rgba<u16>),
            ($crate::Storage::F16, 3) => $body!($crate::Rgb<$crate::f16>),
            ($crate::Storage::F16, 4) => $body!($crate::Rgba<$crate::f16>),
            ($crate::Storage::F32, 3) => $body!($crate::Rgb<f32>),
            ($crate::Storage::F32, 4) => $body!($crate::Rgba<f32>),
            (storage, bands) => {
                return Err($crate::Error::Vips(format!(
                    "working sandwich: unsupported storage {storage:?} x{bands} bands; convert to an RGB(A) format first"
                )));
            }
        }
    };
}

/// Executes a `RegionProcessor` on a `CustomRegion`, automatically dispatching
/// the physical pixel format to the generic `process` method.
pub fn execute_processor<R: RegionProcessor>(
    input: &CustomRegion,
    out: &mut CustomRegion,
    policy: AlphaPolicy,
    processor: &R,
) -> Result<(), Error> {
    macro_rules! go {
        ($p:ty) => {{
            let src = RegionView::<$p>::new(input, policy);
            let mut dst = RegionViewMut::<$p>::with_source(out, input, policy);
            processor.process(&src, &mut dst);
        }};
    }
    dispatch_format!(input.storage(), input.bands(), go);
    Ok(())
}
