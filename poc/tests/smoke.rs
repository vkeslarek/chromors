//! Compile-only smoke test: define a real Kind + a real multi-backend op and
//! chain them. Never runs (no GPU ctx needed) — its job is to prove the whole
//! generic machinery (erased `AnyOperation` bridge, `Lower` supertrait,
//! `GpuView` capability, `push` bounds) type-checks for a concrete op.

use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use poc::backend::Backend;
use poc::backend::gpu::{GpuBackend, GpuBuilder, GpuView};
use poc::backend::gpu::view::{OutBuffer, OutputWrap, RegionParams, View};
use poc::buffer::Buffer;
use poc::error::Error;
use poc::io::Source;
use poc::kind::{AnyKind, Kind};
use poc::node::Data;
use poc::operation::{AnyInput, Input, Lower, Operation};
use poc::work_unit::{Region, Shape, WorkUnit};

// ── A datatype ────────────────────────────────────────────────────────────────

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

impl GpuView for ImageKind {
    // The Kind owns its codec: a decode wrapper at the input, an encode sandwich
    // at the output. Ops never touch this.
    fn input(&self) -> View {
        View::new("uint", "CodecRegion<U8Codec, 0>", "{ {buf}, {region} }")
    }
    fn output(&self) -> OutputWrap {
        OutputWrap {
            arg_type: "RWRegion".into(),
            arg_ctor: "{ {buf}, {region} }".into(),
            arg_buffer: OutBuffer::Scratch,
            encode: Some(View::new("Atomic<uint>", "RWCodecRegion<U8Codec, 0>", "{ {buf}, {region} }")),
        }
    }
}

type Image2D<B> = Data<ImageKind, B>;

// ── A multi-backend operation ─────────────────────────────────────────────────

struct Blur<B: Backend> {
    input: Input<ImageKind, B>,
    radius: f32,
}

// Structural half — written once, generic over B.
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

// Execution half — GPU only here (drop this and Blur is unusable on GPU).
// Note the views are pulled from the CONCRETE Kinds here (`ImageKind: GpuView`),
// then injected into the builder — the materializer never sees them.
impl Lower<GpuBackend> for Blur<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        // The op contributes ONLY its kernel; the output Kind contributes how
        // its result lands in the target (`output()` — the codec sandwich).
        cx.kernel("blur_main").param("radius", self.radius);
        cx.output(self.output_spec().output());
    }
}

// ── A source leaf, so the DAG has a bottom ────────────────────────────────────

struct ImageSource {
    spec: Arc<ImageKind>,
}

impl Source<GpuBackend> for ImageSource {
    type Kind = ImageKind;
    fn spec(&self) -> Arc<ImageKind> {
        self.spec.clone()
    }
    fn fetch(&self, _ctx: &<GpuBackend as Backend>::Ctx, _wu: &Region) -> Result<Buffer<GpuBackend>, Error> {
        unimplemented!("upload (mock)")
    }
    fn lower(&self, cx: &mut GpuBuilder) {
        // A source provides the decode view + geometry + the fetched buffer.
        let WorkUnit::Region(r) = cx.wu().clone() else { return };
        if let Ok(buf) = self.fetch(cx.ctx().as_ref(), &r) {
            cx.input(self.spec.input(), RegionParams::tight(r.w, r.h), buf.payload);
        }
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// Ergonomic method. NOTE: a downstream crate (like this test) cannot add an
// *inherent* method to `Data` (E0116) — only the crate that defines `Data`
// can. So an op authored outside the engine exposes its sugar via an
// extension trait. First-party engine ops can use inherent impls instead.
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

// ── The actual check: a chain type-checks end to end ──────────────────────────

fn _chain(img: Image2D<GpuBackend>) -> Image2D<GpuBackend> {
    img.blur(2.0).blur(4.0)
}

// The erased bridge is reachable: a Node<B> built from a Blur exposes the
// object-safe surface the materializer walks.
fn _erased(op: Blur<GpuBackend>) -> Arc<dyn poc::operation::AnyOperation<GpuBackend>> {
    Arc::new(op)
}

#[test]
fn model_type_checks() {
    // Nothing to run — compiling this file is the assertion.
}
