use crate::prelude::*;
use std::hash::Hasher;

use crate::GpuView;
use chromors_core::backend::Backend;
use chromors_core::data::image::ImageKind;
use chromors_core::operation::{AnyInput, Input, Lower, Operation};
use chromors_core::work_unit::{Region, WorkUnit};

impl chromors_core::operation::Lower<crate::GpuBackend>
    for chromors_core::operation::opacity::Opacity<crate::GpuBackend>
{
    fn lower(&self, cx: &mut crate::GpuBuilder) {
        cx.param_block(crate::view::ParamBlock::new().param("amount", self.amount));
        cx.kernel("ops.opacity", "opacity_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}
