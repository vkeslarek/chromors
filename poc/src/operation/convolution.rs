use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{IntoVipsEnum, VipsBackend, VipsBuilder};
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation, OperationMorphology};
use crate::work_unit::{Region, WorkUnit};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    Integer,
    Float,
    Approximate,
}
impl IntoVipsEnum for Precision {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

// ── Convolution ───────────────────────────────────────────────────────────────

pub struct Convolution<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<ImageKind, B>,
    pub precision: Option<Precision>,
    pub layers: Option<i32>,
    pub cluster: Option<i32>,
}

impl<B: Backend> Operation<B> for Convolution<B> where Convolution<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.mask] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.precision { state.write_i32(v.into_vips()); }
        if let Some(v) = self.layers { state.write_i32(v); }
        if let Some(v) = self.cluster { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for Convolution<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"conv\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        if let Some(v) = self.precision {
            op.set_int("precision", v.into_vips());
        }
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        if let Some(v) = self.cluster {
            op.set_int("cluster", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Compass ───────────────────────────────────────────────────────────────────

pub struct Compass<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Compass<B> where Compass<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.mask] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Compass<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"compass\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Morph ─────────────────────────────────────────────────────────────────────

pub struct Morph<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<ImageKind, B>,
    pub morph: OperationMorphology,
}

impl<B: Backend> Operation<B> for Morph<B> where Morph<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.mask] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.morph.into_vips());
    }
}

impl Lower<VipsBackend> for Morph<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"morph\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        op.set_int("morph", self.morph.into_vips());
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Conva ─────────────────────────────────────────────────────────────────────

pub struct Conva<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<ImageKind, B>,
    pub layers: Option<i32>,
    pub cluster: Option<i32>,
}

impl<B: Backend> Operation<B> for Conva<B> where Conva<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.mask] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.layers { state.write_i32(v); }
        if let Some(v) = self.cluster { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for Conva<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"conva\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        if let Some(v) = self.cluster {
            op.set_int("cluster", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Convf ─────────────────────────────────────────────────────────────────────

pub struct Convf<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Convf<B> where Convf<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.mask] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Convf<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"convf\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Convi ─────────────────────────────────────────────────────────────────────

pub struct Convi<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Convi<B> where Convi<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.mask] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Convi<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"convi\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Convsep ───────────────────────────────────────────────────────────────────

pub struct Convsep<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<ImageKind, B>,
    pub precision: Option<Precision>,
    pub layers: Option<i32>,
    pub cluster: Option<i32>,
}

impl<B: Backend> Operation<B> for Convsep<B> where Convsep<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.mask] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.precision { state.write_i32(v.into_vips()); }
        if let Some(v) = self.layers { state.write_i32(v); }
        if let Some(v) = self.cluster { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for Convsep<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"convsep\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        if let Some(v) = self.precision {
            op.set_int("precision", v.into_vips());
        }
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        if let Some(v) = self.cluster {
            op.set_int("cluster", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Convasep ──────────────────────────────────────────────────────────────────

pub struct Convasep<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub mask: Input<ImageKind, B>,
    pub layers: Option<i32>,
}

impl<B: Backend> Operation<B> for Convasep<B> where Convasep<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input, &self.mask] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mw = self.mask.spec.width;
        let mh = self.mask.spec.height;
        let halo = (mw / 2).max(mh / 2);
        vec![
            Some(WorkUnit::Region(Region {
                x: out.x - halo,
                y: out.y - halo,
                w: out.w + 2 * halo,
                h: out.h + 2 * halo,
                lod: out.lod,
            })),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: mw,
                h: mh,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.layers { state.write_i32(v); }
    }
}

impl Lower<VipsBackend> for Convasep<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mask_handle = cx.input(self.mask.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"convasep\0").unwrap();
        op.set_image("in", input_handle.ptr);
        op.set_image("mask", mask_handle.ptr);
        if let Some(v) = self.layers {
            op.set_int("layers", v);
        }
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}
