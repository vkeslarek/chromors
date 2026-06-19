use crate::color_params::{ConvertParams, color_read_wrap};
use crate::prelude::*;

impl Lower<GpuBackend> for crate::Convert<GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let src = self.input.spec.layout;
        let params = ConvertParams::build(src, self.target).unwrap_or_else(|e| {
            cx.fail(e);
            ConvertParams::identity()
        });

        if src.channel_count() == 5 {
            // CmykA source: the K-alpha 5th channel can't ride through the
            // generic float4 read wrap — bind the storage codec directly.
            cx.param_block(ParamBlock::from_pod("cc", &params));
            cx.kernel("lib.color.convert", "convert_cmyka_kernel");
        } else {
            cx.kernel("lib.io", "copy_kernel");
            cx.read_wrap(color_read_wrap(params));
        }
        cx.output(self.output_spec().output(cx.wu()));
    }
}
