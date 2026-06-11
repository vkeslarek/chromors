use std::hash::Hasher;

use crate::backend::Backend;
use crate::backend::vips::{VipsBackend, VipsBuilder};
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Region, WorkUnit};

// ── ForwardFft ────────────────────────────────────────────────────────────────

pub struct ForwardFft<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for ForwardFft<B> where ForwardFft<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        // FFT needs the whole image to work properly, but typically we'll demand the whole thing.
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for ForwardFft<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"fwfft\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── InverseFft ────────────────────────────────────────────────────────────────

pub struct InverseFft<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for InverseFft<B> where InverseFft<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for InverseFft<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"invfft\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}

// ── Spectrum ──────────────────────────────────────────────────────────────────

pub struct Spectrum<B: Backend> {
    pub input: Input<ImageKind, B>,
}

impl<B: Backend> Operation<B> for Spectrum<B> where Spectrum<B>: Lower<B> {
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> { vec![&self.input] }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind { (*self.input.spec).clone() }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

impl Lower<VipsBackend> for Spectrum<VipsBackend> {
    fn lower(&self, cx: &mut VipsBuilder) {
        let input_handle = cx.input(self.input.src());
        let mut op = crate::backend::vips::gobject::VipsGObject::new(b"spectrum\0").unwrap();
        op.set_image("in", input_handle.ptr);
        let out_handle = op.run().unwrap();
        cx.emit(out_handle);
    }
}


impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    ForwardFft<B>: crate::operation::Lower<B>,
{
    pub fn forward_fft(&self) -> Self {
        self.push(ForwardFft { input: self.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    InverseFft<B>: crate::operation::Lower<B>,
{
    pub fn inverse_fft(&self) -> Self {
        self.push(InverseFft { input: self.as_input() })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Spectrum<B>: crate::operation::Lower<B>,
{
    pub fn spectrum(&self) -> Self {
        self.push(Spectrum { input: self.as_input() })
    }
}
