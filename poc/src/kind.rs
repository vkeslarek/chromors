use std::any::Any;
use std::fmt::Debug;
use std::hash::Hasher;
use crate::work_unit::{WorkUnit, WorkUnitFor};

/// Object-safe, **backend-agnostic** datatype metadata. The runtime talks to
/// this without knowing the concrete Kind *or* the backend. No `view`/`params`
/// here — those are Slang-specific and live in the GPU lowering capability
/// (`GpuView`); a Vips-only Kind impls `VipsBand` instead.
pub trait AnyKind: Send + Sync + Debug + 'static {
    fn as_any(&self) -> &dyn Any;
    /// Size in bytes of the given WorkUnit for this Kind.
    fn byte_size(&self, wu: &WorkUnit) -> u64;
    /// Feed this Kind's identity into a hasher (cache key, `Cached` adapter).
    /// Hashes raw bits, not a `Debug` string.
    fn dyn_hash(&self, state: &mut dyn Hasher);
}

/// Typed surface for generic code. Just the WorkUnit it divides into.
/// No `residency` or `Value` decode — turning a result into a host value is a
/// `Target`'s job, and data stays backend-resident until then.
pub trait Kind: AnyKind + Clone + Sized {
    type WorkUnit: WorkUnitFor;
}
