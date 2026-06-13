use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::gpu::GpuView;
use crate::backend::vips::{VipsBackend, VipsBuilder};
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

pub struct Opacity<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub amount: f32,
}

impl<B: Backend> Operation<B> for Opacity<B>
where
    Opacity<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        // If it didn't have alpha, it now does (same codec, +1 band).
        let channels = spec.format.channel_count();
        if channels == 1 || channels == 3 {
            spec.format = spec.with_band_count(channels as i32 + 1);
        }
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.amount.to_bits());
    }
}

impl Lower<VipsBackend> for Opacity<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());

        let mut ptr = input_handle.ptr;
        let bands = unsafe { crate::ffi::vips_image_get_bands(ptr) };
        let format = unsafe { crate::ffi::vips_image_get_format(ptr) };

        // 1. If no alpha, bandjoin max val
        if bands == 1 || bands == 3 {
            let max_val: f64 = if format == crate::ffi::VipsBandFormat_VIPS_FORMAT_USHORT
                || format == crate::ffi::VipsBandFormat_VIPS_FORMAT_SHORT
            {
                65535.0
            } else {
                255.0
            };

            let mut out: *mut crate::ffi::VipsImage = std::ptr::null_mut();
            let arr = [max_val];
            let ret = unsafe {
                crate::ffi::vips_bandjoin_const(ptr, &mut out, arr.as_ptr() as *mut f64, 1)
            };
            if ret != 0 {
                panic!("vips_bandjoin_const failed");
            }
            ptr = out;
        }

        let bands = unsafe { crate::ffi::vips_image_get_bands(ptr) };

        if bands < 2 {
            let mut op = crate::backend::vips::gobject::VipsGObject::new(b"linear\0").unwrap();
            op.set_image("in", ptr);
            op.set_array_double("a", &[self.amount as f64]);
            op.set_array_double("b", &[0.0]);
            let out_handle = op.run().unwrap();
            cx.emit(out_handle);
            return;
        }

        // 2. Extract RGB
        let mut op_rgb =
            crate::backend::vips::gobject::VipsGObject::new(b"extract_band\0").unwrap();
        op_rgb.set_image("in", ptr);
        op_rgb.set_int("band", 0);
        op_rgb.set_int("n", bands - 1);
        let rgb_handle = op_rgb.run().unwrap();

        // 3. Extract Alpha
        let mut op_alpha =
            crate::backend::vips::gobject::VipsGObject::new(b"extract_band\0").unwrap();
        op_alpha.set_image("in", ptr);
        op_alpha.set_int("band", bands - 1);
        op_alpha.set_int("n", 1);
        let alpha_handle = op_alpha.run().unwrap();

        // 4. Scale Alpha
        let mut op_lin = crate::backend::vips::gobject::VipsGObject::new(b"linear\0").unwrap();
        op_lin.set_image("in", alpha_handle.ptr);
        op_lin.set_array_double("a", &[self.amount as f64]);
        op_lin.set_array_double("b", &[0.0]);
        let uchar = format == crate::ffi::VipsBandFormat_VIPS_FORMAT_UCHAR;
        if uchar {
            op_lin.set_bool("uchar", true);
        }
        let scaled_alpha_handle = op_lin.run().unwrap();

        // 5. Cast back
        let mut op_cast = crate::backend::vips::gobject::VipsGObject::new(b"cast\0").unwrap();
        op_cast.set_image("in", scaled_alpha_handle.ptr);
        op_cast.set_int("format", format);
        let cast_alpha_handle = op_cast.run().unwrap();

        // 6. Bandjoin
        let mut out: *mut crate::ffi::VipsImage = std::ptr::null_mut();
        let ptrs = [rgb_handle.ptr, cast_alpha_handle.ptr];
        let ret = unsafe {
            crate::ffi::vips_bandjoin(
                ptrs.as_ptr() as *mut *mut crate::ffi::VipsImage,
                &mut out,
                2,
                crate::backend::vips::null(),
            )
        };
        if ret != 0 {
            panic!("vips_bandjoin failed");
        }

        cx.emit(crate::backend::vips::VipsHandle { ptr: out });
    }
}

impl crate::operation::Lower<crate::backend::gpu::GpuBackend>
    for Opacity<crate::backend::gpu::GpuBackend>
{
    fn lower(&self, cx: &mut crate::backend::gpu::GpuBuilder) {
        cx.param_block(crate::backend::gpu::view::ParamBlock::new().param("amount", self.amount));
        cx.kernel("ops.opacity", "opacity_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Opacity<B>: crate::operation::Lower<B>,
{
    pub fn opacity(&self, amount: f32) -> Self {
        self.push(Opacity {
            input: self.as_input(),
            amount,
        })
    }
}
