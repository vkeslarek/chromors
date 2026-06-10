//! Image2D mosaicing / stitching operations (`VipsMosaicing`).
//!
//! The reference image is `self`; the secondary image is passed as `sec`.
//! Operations that auto-detect alignment expose only the joined `out` image;
//! the detected-offset outputs are not surfaced.

use super::Direction;
use crate::backend::vips::IntoVipsEnum;
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::libvips_ffi as ffi;

/// Join two images at a single tie-point (`mosaic`), searching for the best fit.
pub struct MosaicOperation<'a> {
    pub secondary: &'a crate::data::image::Image2D<crate::backend::vips::VipsBackend>,
    pub direction: Direction,
    pub x_reference: i32,
    pub y_reference: i32,
    pub x_secondary: i32,
    pub y_secondary: i32,
    pub half_window: Option<i32>,
    pub half_area: Option<i32>,
    pub max_blend: Option<i32>,
    pub search_band: Option<i32>,
}
impl VipsOperation for MosaicOperation<'_> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"mosaic\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("ref", image);
        op.set_image("sec", self.secondary.vips_ptr());
        op.set_int("direction", self.direction.into_vips());
        op.set_int("xref", self.x_reference);
        op.set_int("yref", self.y_reference);
        op.set_int("xsec", self.x_secondary);
        op.set_int("ysec", self.y_secondary);
        if let Some(v) = self.half_window {
            op.set_int("hwindow", v);
        }
        if let Some(v) = self.half_area {
            op.set_int("harea", v);
        }
        if let Some(v) = self.max_blend {
            op.set_int("mblend", v);
        }
        if let Some(v) = self.search_band {
            op.set_int("bandno", v);
        }
    }
}

/// Join two images at two tie-points (`mosaic1`), correcting scale and rotation.
pub struct Mosaic1Operation<'a> {
    pub secondary: &'a crate::data::image::Image2D<crate::backend::vips::VipsBackend>,
    pub direction: Direction,
    pub x_reference_1: i32,
    pub y_reference_1: i32,
    pub x_secondary_1: i32,
    pub y_secondary_1: i32,
    pub x_reference_2: i32,
    pub y_reference_2: i32,
    pub x_secondary_2: i32,
    pub y_secondary_2: i32,
    pub half_window: Option<i32>,
    pub half_area: Option<i32>,
    pub search: Option<bool>,
    pub interpolate: Option<&'a crate::backend::vips::Interpolate>,
    pub max_blend: Option<i32>,
}
impl VipsOperation for Mosaic1Operation<'_> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"mosaic1\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("ref", image);
        op.set_image("sec", self.secondary.vips_ptr());
        op.set_int("direction", self.direction.into_vips());
        op.set_int("xr1", self.x_reference_1);
        op.set_int("yr1", self.y_reference_1);
        op.set_int("xs1", self.x_secondary_1);
        op.set_int("ys1", self.y_secondary_1);
        op.set_int("xr2", self.x_reference_2);
        op.set_int("yr2", self.y_reference_2);
        op.set_int("xs2", self.x_secondary_2);
        op.set_int("ys2", self.y_secondary_2);
        if let Some(v) = self.half_window {
            op.set_int("hwindow", v);
        }
        if let Some(v) = self.half_area {
            op.set_int("harea", v);
        }
        if let Some(v) = self.search {
            op.set_bool("search", v);
        }
        if let Some(v) = self.interpolate {
            op.set_interpolate("interpolate", v);
        }
        if let Some(v) = self.max_blend {
            op.set_int("mblend", v);
        }
    }
}

/// First-order geometric match of two images at two tie-points (`match`).
pub struct MatchOperation<'a> {
    pub secondary: &'a crate::data::image::Image2D<crate::backend::vips::VipsBackend>,
    pub x_reference_1: i32,
    pub y_reference_1: i32,
    pub x_secondary_1: i32,
    pub y_secondary_1: i32,
    pub x_reference_2: i32,
    pub y_reference_2: i32,
    pub x_secondary_2: i32,
    pub y_secondary_2: i32,
    pub half_window: Option<i32>,
    pub half_area: Option<i32>,
    pub search: Option<bool>,
    pub interpolate: Option<&'a crate::backend::vips::Interpolate>,
}
impl VipsOperation for MatchOperation<'_> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"match\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("ref", image);
        op.set_image("sec", self.secondary.vips_ptr());
        op.set_int("xr1", self.x_reference_1);
        op.set_int("yr1", self.y_reference_1);
        op.set_int("xs1", self.x_secondary_1);
        op.set_int("ys1", self.y_secondary_1);
        op.set_int("xr2", self.x_reference_2);
        op.set_int("yr2", self.y_reference_2);
        op.set_int("xs2", self.x_secondary_2);
        op.set_int("ys2", self.y_secondary_2);
        if let Some(v) = self.half_window {
            op.set_int("hwindow", v);
        }
        if let Some(v) = self.half_area {
            op.set_int("harea", v);
        }
        if let Some(v) = self.search {
            op.set_bool("search", v);
        }
        if let Some(v) = self.interpolate {
            op.set_interpolate("interpolate", v);
        }
    }
}

/// Directly merge two pre-aligned images at a fixed displacement (`merge`).
pub struct MergeOperation<'a> {
    pub secondary: &'a crate::data::image::Image2D<crate::backend::vips::VipsBackend>,
    pub direction: Direction,
    pub dx: i32,
    pub dy: i32,
    pub max_blend: Option<i32>,
}
impl VipsOperation for MergeOperation<'_> {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"merge\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("ref", image);
        op.set_image("sec", self.secondary.vips_ptr());
        op.set_int("direction", self.direction.into_vips());
        op.set_int("dx", self.dx);
        op.set_int("dy", self.dy);
        if let Some(v) = self.max_blend {
            op.set_int("mblend", v);
        }
    }
}

/// Re-balance the brightness of a mosaic built from many sub-images (`globalbalance`).
pub struct GlobalBalanceOperation {
    pub gamma: Option<f64>,
    pub integer_output: Option<bool>,
}
impl VipsOperation for GlobalBalanceOperation {
    type Output = crate::data::image::Image2D<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"globalbalance\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.gamma {
            op.set_double("gamma", v);
        }
        if let Some(v) = self.integer_output {
            op.set_bool("int_output", v);
        }
    }
}
