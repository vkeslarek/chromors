use crate::backend::Backend;
use crate::data::image::{Image2D, ImageKind};
use crate::error::Error;
use crate::node::Data;
use crate::pixel::{PixelLayout, Storage};
use crate::work_unit::Region;
use std::hash::Hasher;
use std::sync::Arc;

/// A zero-input procedural image source.
pub trait Generator: Send + Sync + 'static {
    /// Output metadata: pixel layout + full-res extent.
    fn spec(&self) -> Arc<ImageKind>;

    /// Identity for the pipeline cache key.
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

/// Wrapper that carries a `Generator` as a graph leaf.
pub struct GenSource<G: Generator>(pub G);

/// A generator that fills the region with a constant color.
pub struct Constant {
    pub w: i32,
    pub h: i32,
    pub layout: PixelLayout,
    pub color: [f32; 4],
}

impl Generator for Constant {
    fn spec(&self) -> Arc<ImageKind> {
        Arc::new(ImageKind::new(self.layout, self.w, self.h))
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.w);
        state.write_i32(self.h);
        state.write_u8(self.layout.storage as u8);
        for c in self.color {
            state.write_u32(c.to_bits());
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearGradient {
    pub w: i32,
    pub h: i32,
    pub layout: PixelLayout,
    pub c0: [f32; 4],
    pub c1: [f32; 4],
    pub angle: f32,
}

impl Generator for LinearGradient {
    fn spec(&self) -> Arc<ImageKind> {
        Arc::new(ImageKind::new(self.layout, self.w, self.h))
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.w);
        state.write_i32(self.h);
        state.write_u8(self.layout.storage as u8);
        for c in self.c0 {
            state.write_u32(c.to_bits());
        }
        for c in self.c1 {
            state.write_u32(c.to_bits());
        }
        state.write_u32(self.angle.to_bits());
    }
}

#[derive(Debug, Clone)]
pub struct Xyz {
    pub w: i32,
    pub h: i32,
    pub layout: PixelLayout,
}

impl Generator for Xyz {
    fn spec(&self) -> Arc<ImageKind> {
        Arc::new(ImageKind::new(self.layout, self.w, self.h))
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.w);
        state.write_i32(self.h);
        state.write_u8(self.layout.storage as u8);
    }
}

#[derive(Debug, Clone)]
pub struct GaussNoise {
    pub w: i32,
    pub h: i32,
    pub layout: PixelLayout,
    pub mean: f32,
    pub sigma: f32,
    pub seed: u32,
}

impl Generator for GaussNoise {
    fn spec(&self) -> Arc<ImageKind> {
        Arc::new(ImageKind::new(self.layout, self.w, self.h))
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.w);
        state.write_i32(self.h);
        state.write_u8(self.layout.storage as u8);
        state.write_u32(self.mean.to_bits());
        state.write_u32(self.sigma.to_bits());
        state.write_u32(self.seed);
    }
}

/// Build a generator pipeline tip on backend `B`.
pub fn from_gen<B: Backend, G: Generator>(g: G, ctx: Arc<B::Ctx>) -> Image2D<B>
where
    GenSource<G>: crate::io::Source<B, Kind = ImageKind>,
{
    Data::from_source(Arc::new(GenSource(g)), ctx)
}

impl<B: Backend> Image2D<B> {
    pub fn constant(ctx: Arc<B::Ctx>, w: i32, h: i32, layout: PixelLayout, color: [f32; 4]) -> Self
    where
        GenSource<Constant>: crate::io::Source<B, Kind = ImageKind>,
    {
        from_gen(
            Constant {
                w,
                h,
                layout,
                color,
            },
            ctx,
        )
    }

    pub fn gradient(
        ctx: Arc<B::Ctx>,
        w: i32,
        h: i32,
        layout: PixelLayout,
        c0: [f32; 4],
        c1: [f32; 4],
        angle: f32,
    ) -> Self
    where
        GenSource<LinearGradient>: crate::io::Source<B, Kind = ImageKind>,
    {
        from_gen(
            LinearGradient {
                w,
                h,
                layout,
                c0,
                c1,
                angle,
            },
            ctx,
        )
    }

    pub fn xyz(ctx: Arc<B::Ctx>, w: i32, h: i32, layout: PixelLayout) -> Self
    where
        GenSource<Xyz>: crate::io::Source<B, Kind = ImageKind>,
    {
        from_gen(Xyz { w, h, layout }, ctx)
    }

    pub fn gauss_noise(
        ctx: Arc<B::Ctx>,
        w: i32,
        h: i32,
        layout: PixelLayout,
        mean: f32,
        sigma: f32,
        seed: u32,
    ) -> Self
    where
        GenSource<GaussNoise>: crate::io::Source<B, Kind = ImageKind>,
    {
        from_gen(
            GaussNoise {
                w,
                h,
                layout,
                mean,
                sigma,
                seed,
            },
            ctx,
        )
    }
}
