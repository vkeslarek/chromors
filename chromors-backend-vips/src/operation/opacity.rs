use crate::prelude::*;

impl Lower<VipsBackend> for crate::Opacity<VipsBackend> {
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
            let mut op = VipsGObject::new(b"linear\0").unwrap();
            op.set_image("in", ptr);
            op.set_array_double("a", &[self.amount as f64]);
            op.set_array_double("b", &[0.0]);
            let out_handle = op.run().unwrap();
            cx.emit(out_handle);
            return;
        }

        // 2. Extract RGB
        let mut op_rgb =
            VipsGObject::new(b"extract_band\0").unwrap();
        op_rgb.set_image("in", ptr);
        op_rgb.set_int("band", 0);
        op_rgb.set_int("n", bands - 1);
        let rgb_handle = op_rgb.run().unwrap();

        // 3. Extract Alpha
        let mut op_alpha =
            VipsGObject::new(b"extract_band\0").unwrap();
        op_alpha.set_image("in", ptr);
        op_alpha.set_int("band", bands - 1);
        op_alpha.set_int("n", 1);
        let alpha_handle = op_alpha.run().unwrap();

        // 4. Scale Alpha
        let mut op_lin = VipsGObject::new(b"linear\0").unwrap();
        op_lin.set_image("in", alpha_handle.ptr);
        op_lin.set_array_double("a", &[self.amount as f64]);
        op_lin.set_array_double("b", &[0.0]);
        let uchar = format == crate::ffi::VipsBandFormat_VIPS_FORMAT_UCHAR;
        if uchar {
            op_lin.set_bool("uchar", true);
        }
        let scaled_alpha_handle = op_lin.run().unwrap();

        // 5. Cast back
        let mut op_cast = VipsGObject::new(b"cast\0").unwrap();
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
                crate::null(),
            )
        };
        if ret != 0 {
            panic!("vips_bandjoin failed");
        }

        cx.emit(VipsHandle { ptr: out });
    }
}

