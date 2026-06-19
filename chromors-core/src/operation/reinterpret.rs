use std::hash::Hasher;

use crate::backend::Backend;
use crate::kind::Kind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{WorkUnit, WorkUnitFor};

/// A zero-cost typed cast in the graph: output Kind differs, payload is the
/// input's payload, untouched. Lowering forwards the input — no kernel on the
/// GPU, handle passthrough on vips.
pub struct Reinterpret<K: Kind, T: Kind, B: Backend> {
    pub input: Input<K, B>,
    pub spec: T,
}

impl<K, T, B> Operation<B> for Reinterpret<K, T, B>
where
    K: Kind,
    T: Kind<WorkUnit = K::WorkUnit>,
    B: Backend,
    Self: Lower<B>,
{
    type Output = T;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }

    fn demand(&self, out: &T::WorkUnit) -> Vec<Option<WorkUnit>> {
        let wu = out.erase();
        debug_assert_eq!(
            self.input.spec.byte_size(&wu),
            self.spec.byte_size(&wu),
            "Reinterpret requires byte-identical payloads"
        );
        vec![Some(wu)]
    }

    fn output_spec(&self) -> T {
        self.spec.clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        self.spec.dyn_hash(state);
    }
}
