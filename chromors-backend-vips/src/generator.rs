use crate::{VipsBackend, VipsBuilder, VipsHandle};
use chromors_core::buffer::Buffer;
use chromors_core::data::image::ImageKind;
use chromors_core::error::Error;
use chromors_core::generator::{GenSource, Generator};
use chromors_core::io::Source;
use chromors_core::work_unit::{Region, WorkUnit, WorkUnitFor};
use std::sync::Arc;

pub trait VipsGenerator: Generator {
    fn render_cpu(&self, region: &Region) -> Result<Vec<u8>, Error>;
}

impl<G: VipsGenerator> Source<VipsBackend> for GenSource<G> {
    type Kind = ImageKind;

    fn spec(&self) -> Arc<ImageKind> {
        self.0.spec()
    }

    fn fetch(&self, _ctx: &(), wu: &Region) -> Result<Buffer<VipsBackend>, Error> {
        let bytes = self.0.render_cpu(wu)?;
        let spec = self.0.spec();
        let handle = crate::backend::image_from_memory(&bytes, wu.w, wu.h, spec.layout)?;
        Ok(Buffer {
            payload: handle,
            spec,
        })
    }

    fn lower(&self, cx: &mut VipsBuilder) {
        let wu = Region::typed(cx.wu()).expect("generator expects a Region");
        match self.fetch(&(), &wu) {
            Ok(buf) => cx.emit((*buf.payload).clone()),
            Err(e) => cx.fail(e),
        }
    }

    fn dyn_hash(&self, state: &mut dyn std::hash::Hasher) {
        self.0.dyn_hash(state);
    }
}

use chromors_core::generator::Constant;
use chromors_core::pixel::{PixelLayout, Storage, f16};

impl VipsGenerator for Constant {
    fn render_cpu(&self, region: &Region) -> Result<Vec<u8>, Error> {
        let bpp = self.layout.channel_count() * self.layout.storage.bytes_per_sample();
        let size = (region.w.max(0) as usize) * (region.h.max(0) as usize) * bpp;
        let mut out = vec![0u8; size];

        let mut pixel_bytes = vec![0u8; bpp];
        let max = self.layout.storage.component_max();
        for i in 0..self.layout.channel_count() {
            let c = self.color[i.min(3)]; // fallback for extra channels
            let val = (c.clamp(0.0, 1.0) * max).round();
            match self.layout.storage {
                Storage::U8 => pixel_bytes[i] = val as u8,
                Storage::U16 => {
                    pixel_bytes[i * 2..i * 2 + 2].copy_from_slice(&(val as u16).to_ne_bytes())
                }
                Storage::F16 => {
                    pixel_bytes[i * 2..i * 2 + 2].copy_from_slice(&f16::from_f32(c).to_ne_bytes())
                }
                Storage::F32 => pixel_bytes[i * 4..i * 4 + 4].copy_from_slice(&c.to_ne_bytes()),
            }
        }

        for chunk in out.chunks_exact_mut(bpp) {
            chunk.copy_from_slice(&pixel_bytes);
        }
        Ok(out)
    }
}

use crate::generator_rng::{gauss, key};
use chromors_core::generator::{GaussNoise, LinearGradient, Xyz};
use rayon::prelude::*;

#[inline(always)]
fn encode_pixel(c: [f32; 4], layout: PixelLayout, out: &mut [u8]) {
    let max = layout.storage.component_max();
    for i in 0..layout.channel_count() {
        let v = c[i.min(3)].clamp(0.0, 1.0);
        let val = (v * max).round();
        match layout.storage {
            Storage::U8 => out[i] = val as u8,
            Storage::U16 => out[i * 2..i * 2 + 2].copy_from_slice(&(val as u16).to_ne_bytes()),
            Storage::F16 => {
                out[i * 2..i * 2 + 2].copy_from_slice(&f16::from_f32(c[i.min(3)]).to_ne_bytes())
            }
            Storage::F32 => out[i * 4..i * 4 + 4].copy_from_slice(&c[i.min(3)].to_ne_bytes()),
        }
    }
}

impl VipsGenerator for LinearGradient {
    fn render_cpu(&self, region: &Region) -> Result<Vec<u8>, Error> {
        let bpp = self.layout.channel_count() * self.layout.storage.bytes_per_sample();
        let w = region.w.max(0) as usize;
        let h = region.h.max(0) as usize;
        let mut out = vec![0u8; w * h * bpp];

        let scale = 1 << region.lod.0;
        let fw = self.w.max(1) as f32;
        let fh = self.h.max(1) as f32;
        let dir_x = self.angle.cos();
        let dir_y = self.angle.sin();

        out.as_mut_slice()
            .par_chunks_exact_mut(w * bpp)
            .enumerate()
            .for_each(|(ly, row)| {
                let gy = region.y + ly as i32;
                let cy = ((gy as f32) + 0.5) * scale as f32;
                let v = cy / fh;

                for lx in 0..w {
                    let gx = region.x + lx as i32;
                    let cx = ((gx as f32) + 0.5) * scale as f32;
                    let u = cx / fw;

                    let t = (u * dir_x + v * dir_y).clamp(0.0, 1.0);
                    let mut pixel = [0.0; 4];
                    for i in 0..4 {
                        pixel[i] = self.c0[i] + (self.c1[i] - self.c0[i]) * t;
                    }
                    encode_pixel(pixel, self.layout, &mut row[lx * bpp..(lx + 1) * bpp]);
                }
            });

        Ok(out)
    }
}

impl VipsGenerator for Xyz {
    fn render_cpu(&self, region: &Region) -> Result<Vec<u8>, Error> {
        let bpp = self.layout.channel_count() * self.layout.storage.bytes_per_sample();
        let w = region.w.max(0) as usize;
        let h = region.h.max(0) as usize;
        let mut out = vec![0u8; w * h * bpp];

        let scale = 1 << region.lod.0;

        out.as_mut_slice()
            .par_chunks_exact_mut(w * bpp)
            .enumerate()
            .for_each(|(ly, row)| {
                let gy = region.y + ly as i32;
                let cy = ((gy as f32) + 0.5) * scale as f32;

                for lx in 0..w {
                    let gx = region.x + lx as i32;
                    let cx = ((gx as f32) + 0.5) * scale as f32;

                    let pixel = [cx, cy, 0.0, 1.0];
                    encode_pixel(pixel, self.layout, &mut row[lx * bpp..(lx + 1) * bpp]);
                }
            });

        Ok(out)
    }
}

impl VipsGenerator for GaussNoise {
    fn render_cpu(&self, region: &Region) -> Result<Vec<u8>, Error> {
        let bpp = self.layout.channel_count() * self.layout.storage.bytes_per_sample();
        let w = region.w.max(0) as usize;
        let h = region.h.max(0) as usize;
        let mut out = vec![0u8; w * h * bpp];

        let lod = region.lod.0 as u32;

        out.as_mut_slice()
            .par_chunks_exact_mut(w * bpp)
            .enumerate()
            .for_each(|(ly, row)| {
                let gy = region.y + ly as i32;

                for lx in 0..w {
                    let gx = region.x + lx as i32;

                    let k0 = key(self.seed, gx as u32, gy as u32);
                    // We mix lod in the key as per spec
                    // Wait, docs say: key(lod_coord, lod, seed)
                    // Let's implement that logic locally to match shaders exactly
                    let mix_s0 = self.seed;
                    let mix_s1 = self
                        .seed
                        .wrapping_mul(747796405)
                        .wrapping_add(lod.wrapping_mul(2891336453));

                    let gx_u = gx as u32;
                    let gy_u = gy as u32;

                    let k0 = crate::generator_rng::pcg2d([gx_u ^ mix_s0, gy_u ^ mix_s1]);
                    let (g0, _) = gauss(k0);

                    let k1 =
                        crate::generator_rng::pcg2d([gx_u ^ mix_s0.wrapping_add(1), gy_u ^ mix_s1]); // Not quite, see seed+1
                    // Docs: gauss(c.lod_coord, c.lod, seed + 0u), etc.

                    let get_gauss = |offset: u32| {
                        let s = self.seed.wrapping_add(offset);
                        let s_mix = s
                            .wrapping_mul(747796405)
                            .wrapping_add(lod.wrapping_mul(2891336453));
                        let k = crate::generator_rng::pcg2d([gx_u ^ s, gy_u ^ s_mix]);
                        gauss(k).0
                    };

                    let pixel = [
                        self.mean + self.sigma * get_gauss(0),
                        self.mean + self.sigma * get_gauss(1),
                        self.mean + self.sigma * get_gauss(2),
                        1.0,
                    ];
                    encode_pixel(pixel, self.layout, &mut row[lx * bpp..(lx + 1) * bpp]);
                }
            });

        Ok(out)
    }
}
