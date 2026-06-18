use crate::prelude::*;

impl<K, T> Lower<GpuBackend> for crate::Reinterpret<K, T, GpuBackend>
where
    K: Kind,
    T: GpuView<WorkUnit = K::WorkUnit>,
{
    fn lower(&self, cx: &mut GpuBuilder) {
        eprintln!("DEBUG crate::Reinterpret::lower called");
        cx.forward();
        cx.output(self.spec.output(cx.wu()));
    }
}

