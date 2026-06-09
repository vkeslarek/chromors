use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

pub struct DrawCircle {
    pub ink: Vec<f64>,
    pub center_x: i32,
    pub center_y: i32,
    pub radius: i32,
}
impl VipsOperation for DrawCircle {
    type Output = ();
    fn name() -> &'static [u8] {
        b"draw_circle\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("image", image);
        op.set_array_double("ink", &self.ink);
        op.set_int("cx", self.center_x);
        op.set_int("cy", self.center_y);
        op.set_int("radius", self.radius);
    }
}

pub struct DrawRect {
    pub ink: Vec<f64>,
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}
impl VipsOperation for DrawRect {
    type Output = ();
    fn name() -> &'static [u8] {
        b"draw_rect\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("image", image);
        op.set_array_double("ink", &self.ink);
        op.set_int("left", self.left);
        op.set_int("top", self.top);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}

pub struct DrawLine {
    pub ink: Vec<f64>,
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
}
impl VipsOperation for DrawLine {
    type Output = ();
    fn name() -> &'static [u8] {
        b"draw_line\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("image", image);
        op.set_array_double("ink", &self.ink);
        op.set_int("x1", self.x1);
        op.set_int("y1", self.y1);
        op.set_int("x2", self.x2);
        op.set_int("y2", self.y2);
    }
}

pub struct DrawImage<'a> {
    pub sub: &'a crate::data::image::Image<crate::backend::vips::VipsBackend>,
    pub x: i32,
    pub y: i32,
}
impl VipsOperation for DrawImage<'_> {
    type Output = ();
    fn name() -> &'static [u8] {
        b"draw_image\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("image", image);
        op.set_image("sub", self.sub.vips_ptr());
        op.set_int("x", self.x);
        op.set_int("y", self.y);
    }
}

pub struct DrawFlood {
    pub ink: Vec<f64>,
    pub x: i32,
    pub y: i32,
}
impl VipsOperation for DrawFlood {
    type Output = ();
    fn name() -> &'static [u8] {
        b"draw_flood\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("image", image);
        op.set_array_double("ink", &self.ink);
        op.set_int("x", self.x);
        op.set_int("y", self.y);
    }
}

pub struct DrawMask<'a> {
    pub ink: Vec<f64>,
    pub mask: &'a crate::data::image::Image<crate::backend::vips::VipsBackend>,
    pub x: i32,
    pub y: i32,
}
impl VipsOperation for DrawMask<'_> {
    type Output = ();
    fn name() -> &'static [u8] {
        b"draw_mask\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("image", image);
        op.set_array_double("ink", &self.ink);
        op.set_image("mask", self.mask.vips_ptr());
        op.set_int("x", self.x);
        op.set_int("y", self.y);
    }
}

pub struct DrawSmudge {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}
impl VipsOperation for DrawSmudge {
    type Output = ();
    fn name() -> &'static [u8] {
        b"draw_smudge\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("image", image);
        op.set_int("left", self.left);
        op.set_int("top", self.top);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
    }
}
