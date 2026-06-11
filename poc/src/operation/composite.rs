use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{IntoVipsEnum, VipsBackend, VipsBuilder};
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};
use crate::operation::geometry::Direction;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Align {
    Low,
    Centre,
    High,
}
impl IntoVipsEnum for Align {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlendMode {
    Clear,
    Source,
    Over,
    In,
    Out,
    Atop,
    Dest,
    DestOver,
    DestIn,
    DestOut,
    DestAtop,
    Xor,
    Add,
    Saturate,
}
impl IntoVipsEnum for BlendMode {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

// ── Composite2 ────────────────────────────────────────────────────────────────

pub struct Composite2<B: Backend> {
    pub base: Input<ImageKind, B>,
    pub overlay: Input<ImageKind, B>,
    pub mode: BlendMode,
    pub x: Option<i32>,
    pub y: Option<i32>,
    pub premultiplied: Option<bool>,
}

impl<B: Backend> Operation<B> for Composite2<B>
where
    Composite2<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.base, &self.overlay]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.base.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.mode.into_vips());
        state.write_i32(self.x.unwrap_or(0));
        state.write_i32(self.y.unwrap_or(0));
    }
}

impl Lower<VipsBackend> for Composite2<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let base_handle = cx.input(self.base.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"composite2\0")
            .expect("failed to create vips composite2 op");
        op.set_image("base", base_handle.ptr);
        let overlay_handle = cx.input(self.overlay.src());
        op.set_image("overlay", overlay_handle.ptr);
        op.set_int("mode", self.mode.into_vips());
        if let Some(v) = self.x {
            op.set_int("x", v);
        }
        if let Some(v) = self.y {
            op.set_int("y", v);
        }
        if let Some(v) = self.premultiplied {
            op.set_bool("premultiplied", v);
        }
        let out_handle = op.run().expect("vips composite2 failed");
        cx.emit(out_handle);
    }
}

// ── Join ──────────────────────────────────────────────────────────────────────

pub struct Join<B: Backend> {
    pub in1: Input<ImageKind, B>,
    pub in2: Input<ImageKind, B>,
    pub direction: Direction,
    pub expand: Option<bool>,
    pub shim: Option<i32>,
    pub align: Option<Align>,
}

impl<B: Backend> Operation<B> for Join<B>
where
    Join<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.in1, &self.in2]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.in1.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.direction.into_vips());
        state.write_i32(self.shim.unwrap_or(0));
        if let Some(a) = self.align {
            state.write_i32(a.into_vips());
        }
    }
}

impl Lower<VipsBackend> for Join<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let in1_handle = cx.input(self.in1.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"join\0")
            .expect("failed to create vips join op");
        op.set_image("in1", in1_handle.ptr);
        let in2_handle = cx.input(self.in2.src());
        op.set_image("in2", in2_handle.ptr);
        op.set_int("direction", self.direction.into_vips());
        if let Some(v) = self.expand {
            op.set_bool("expand", v);
        }
        if let Some(v) = self.shim {
            op.set_int("shim", v);
        }
        if let Some(v) = self.align {
            op.set_int("align", v.into_vips());
        }
        let out_handle = op.run().expect("vips join failed");
        cx.emit(out_handle);
    }
}

// ── Insert ────────────────────────────────────────────────────────────────────

pub struct Insert<B: Backend> {
    pub main: Input<ImageKind, B>,
    pub sub: Input<ImageKind, B>,
    pub x: i32,
    pub y: i32,
    pub expand: Option<bool>,
}

impl<B: Backend> Operation<B> for Insert<B>
where
    Insert<B>: Lower<B>,
{
    type Output = ImageKind;

    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.main, &self.sub]
    }

    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }

    fn output_spec(&self) -> ImageKind {
        (*self.main.spec).clone()
    }

    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.x);
        state.write_i32(self.y);
    }
}

impl Lower<VipsBackend> for Insert<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let main_handle = cx.input(self.main.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"insert\0")
            .expect("failed to create vips insert op");
        op.set_image("main", main_handle.ptr);
        let sub_handle = cx.input(self.sub.src());
        op.set_image("sub", sub_handle.ptr);
        op.set_int("x", self.x);
        op.set_int("y", self.y);
        if let Some(v) = self.expand {
            op.set_bool("expand", v);
        }
        let out_handle = op.run().expect("vips insert failed");
        cx.emit(out_handle);
    }
}
