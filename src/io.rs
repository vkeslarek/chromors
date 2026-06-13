use crate::backend::Backend;
use crate::buffer::Buffer;
use crate::error::Error;
use crate::kind::{AnyKind, Kind};
use crate::work_unit::{WorkUnit, WorkUnitFor};
use std::hash::Hasher;
use std::sync::Arc;

/// Object-safe surface the materializer drives at a Source leaf without knowing
/// the concrete Kind. A blanket impl bridges every typed `Source<B>`.
pub trait AnySource<B: Backend>: Send + Sync + 'static {
    fn output_kind(&self) -> Arc<dyn AnyKind>;
    /// Erased `fetch`: the bridge downcasts `wu` to the source's typed shape.
    fn fetch_erased(&self, ctx: &B::Ctx, wu: &WorkUnit) -> Result<Buffer<B>, Error>;
    /// Inject this leaf's buffer config (GPU: decode view + geometry; Vips:
    /// wire the input) into the builder. Concrete-type site — the materializer
    /// never reads a backend-specific view from the erased Kind. (Resolved
    /// WorkUnit is carried by the builder for backends that need it.)
    fn lower(&self, cx: &mut B::Builder);
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

/// The only door into the model — brings data in (from any backend) and lands
/// it in a `Buffer<B>`. A source produces exactly one Kind, so it's an
/// associated type (not a generic param — that would leave the blanket
/// `AnySource` impl's type parameter unconstrained).
pub trait Source<B: Backend>: Send + Sync + 'static {
    type Kind: Kind;
    fn spec(&self) -> Arc<Self::Kind>;
    fn fetch(&self, ctx: &B::Ctx, wu: &<Self::Kind as Kind>::WorkUnit) -> Result<Buffer<B>, Error>;
    /// Lower this leaf into the backend builder (concrete Kind known here).
    fn lower(&self, cx: &mut B::Builder);
    /// Identity of this leaf for the cache key (e.g. file path + mtime).
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

impl<B: Backend, T: Source<B>> AnySource<B> for T {
    fn output_kind(&self) -> Arc<dyn AnyKind> {
        self.spec()
    }

    fn fetch_erased(&self, ctx: &B::Ctx, wu: &WorkUnit) -> Result<Buffer<B>, Error> {
        let typed = <<T as Source<B>>::Kind as Kind>::WorkUnit::typed(wu)
            .expect("work unit shape mismatch for source");
        self.fetch(ctx, &typed)
    }

    fn lower(&self, cx: &mut B::Builder) {
        Source::lower(self, cx)
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        Source::dyn_hash(self, state);
    }
}

/// The only door out of the model. Extracts a backend-resident buffer into a
/// host value (download + decode), a side effect (write to disk, `Out = ()`),
/// or a still-resident `Buffer<B>` (the viewport exit — clones the Arc, no
/// download). Every exit is a typed Target; there is no raw download path.
pub trait Target<K: Kind, B: Backend>: Send + Sync {
    type Out;
    fn extract(&self, buf: &Buffer<B>, wu: &K::WorkUnit, ctx: &B::Ctx) -> Result<Self::Out, Error>;
}
