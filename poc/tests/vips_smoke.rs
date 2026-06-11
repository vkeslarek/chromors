//! Compile-only proof that the **vips** backend is a first-class citizen of the
//! SAME generic model as the GPU backend: one `ImageKind` (agnostic) + a
//! `VipsBand` capability, an `ImageSource: Source<VipsBackend>`, and a
//! `Blur<VipsBackend>` via `Operation<VipsBackend>` + `Lower<VipsBackend>`.
//! No second operation system, no old `Image2D`, no eager `execute`.
//!
//! Bodies are `unimplemented!()` — constructing real `VipsHandle`s needs FFI;
//! the point is that the trait wiring type-checks. Never runs.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use poc::backend::Backend;
use poc::backend::vips::{VipsBackend, VipsBand, VipsBuilder};
use poc::buffer::Buffer;
use poc::error::Error;
use poc::io::Source;
use poc::kind::{AnyKind, Kind};
use poc::node::Data;
use poc::operation::{AnyInput, Input, Lower, Operation};
use poc::work_unit::{Region, Shape, WorkUnit};

// ── The SAME agnostic datatype, now with a vips capability instead of GpuView ──

#[derive(Clone, Debug)]
struct ImageKind {
    w: i32,
    h: i32,
}

impl AnyKind for ImageKind {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn shape(&self) -> Shape {
        Shape::Region
    }
    fn byte_size(&self, wu: &WorkUnit) -> u64 {
        match wu {
            WorkUnit::Region(r) => (r.w as u64) * (r.h as u64) * 16,
            _ => 0,
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.w);
        state.write_i32(self.h);
    }
}

impl Kind for ImageKind {
    type WorkUnit = Region;
}

// Vips lowering capability — the symmetric counterpart to `GpuView`.
impl VipsBand for ImageKind {
    fn band_format(&self) -> i32 {
        10 // VIPS_FORMAT_FLOAT, say
    }
}

type Image2D<B> = Data<ImageKind, B>;

// ── A vips source leaf, via the generic `Source<VipsBackend>` ─────────────────

struct ImageSource {
    spec: Arc<ImageKind>,
}

impl Source<VipsBackend> for ImageSource {
    type Kind = ImageKind;
    fn spec(&self) -> Arc<ImageKind> {
        self.spec.clone()
    }
    fn fetch(&self, _ctx: &<VipsBackend as Backend>::Ctx, _wu: &Region) -> Result<Buffer<VipsBackend>, Error> {
        unimplemented!("vips load (FFI)")
    }
    fn lower(&self, _cx: &mut VipsBuilder) {
        // Real impl: build a `vips_source` op and `cx.emit(handle)`. The
        // band format would come from `self.spec.band_format()`.
        unimplemented!("vips source lower (FFI)")
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── A blur, same generic structure as GPU; only `Lower` differs per backend ───

struct Blur<B: Backend> {
    input: Input<ImageKind, B>,
    radius: f32,
}

impl<B: Backend> Operation<B> for Blur<B>
where
    Blur<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.expanded(self.radius.ceil() as i32)))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.radius.to_bits());
    }
}

// Execution half — vips builds a libvips op (no Slang, no params, no views).
impl Lower<VipsBackend> for Blur<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let _input = cx.input(&self.input.src); // upstream VipsHandle, by edge identity
        // Real impl: `vips_gaussblur(_input, sigma=self.radius)` then `cx.emit(out)`.
        unimplemented!("vips gaussblur (FFI)")
    }
}

// Ergonomic sugar via extension trait (downstream-style; orphan rule).
trait BlurExt<B: Backend> {
    fn blur(&self, radius: f32) -> Image2D<B>;
}
impl<B: Backend> BlurExt<B> for Image2D<B>
where
    Blur<B>: Lower<B>,
{
    fn blur(&self, radius: f32) -> Image2D<B> {
        self.push(Blur {
            input: self.as_input(),
            radius,
        })
    }
}

// The check: a vips pipeline type-checks through the same surface as GPU.
fn _chain(img: Image2D<VipsBackend>) -> Image2D<VipsBackend> {
    img.blur(2.0).blur(4.0)
}

fn _is_source(s: ImageSource) -> Arc<dyn poc::io::AnySource<VipsBackend>> {
    Arc::new(s)
}

#[test]
fn vips_uses_the_generic_model() {
    // Compiling is the assertion.
}
