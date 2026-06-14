with open("src/data/mask2d.rs", "r") as f:
    code = f.read()

old_fetch = """        // vips_conv/conva/compass/morph read the mask's "scale"/"offset"
        // double properties (defaulting to 0 if unset, which divides the
        // convolution result by zero).
        unsafe {
            let scale = std::ffi::CString::new("scale").unwrap();
            let offset = std::ffi::CString::new("offset").unwrap();
            crate::ffi::vips_image_set_double(ptr, scale.as_ptr(), self.scale);
            crate::ffi::vips_image_set_double(ptr, offset.as_ptr(), self.offset);
        }"""

new_fetch = """        // vips_conv/conva/compass/morph read the mask's "scale"/"offset"
        // double properties (defaulting to 0 if unset, which divides the
        // convolution result by zero).
        unsafe {
            let scale = std::ffi::CString::new("scale").unwrap();
            let offset = std::ffi::CString::new("offset").unwrap();
            let xoffset = std::ffi::CString::new("xoffset").unwrap();
            let yoffset = std::ffi::CString::new("yoffset").unwrap();
            crate::ffi::vips_image_set_double(ptr, scale.as_ptr(), self.scale);
            crate::ffi::vips_image_set_double(ptr, offset.as_ptr(), self.offset);
            crate::ffi::vips_image_set_int(ptr, xoffset.as_ptr(), self.spec.width / 2);
            crate::ffi::vips_image_set_int(ptr, yoffset.as_ptr(), self.spec.height / 2);
        }"""

code = code.replace(old_fetch, new_fetch)

with open("src/data/mask2d.rs", "w") as f:
    f.write(code)
