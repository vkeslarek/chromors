use crate::operation::IntoVipsEnum;
use std::hash::Hasher;

use crate::backend::Backend;
use crate::data::image::ImageKind;
use crate::operation::{AnyInput, Input, Lower, Operation, OperationBoolean, OperationRelational};
use crate::pixel::PixelLayout;
use crate::work_unit::{Range, Region, WorkUnit};

/// Pixel access pattern hint for cache operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    Random,
    Sequential,
    SequentialUnbuffered,
}
impl IntoVipsEnum for Access {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

// ── Cast ──────────────────────────────────────────────────────────────────────

pub struct Cast<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub target: PixelLayout,
    pub shift: Option<bool>,
}

impl<B: Backend> Operation<B> for Cast<B>
where
    Cast<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        self.input.spec.with_layout(self.target)
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write(format!("{:?}", self.target).as_bytes());
        if let Some(v) = self.shift {
            state.write_u8(v as u8);
        }
    }
}

// ── Copy ──────────────────────────────────────────────────────────────────────

pub struct Copy<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub bands: Option<i32>,
    pub format: Option<i32>,
    pub interpretation: Option<i32>,
    pub horizontal_resolution: Option<f64>,
    pub vertical_resolution: Option<f64>,
    pub offset_x: Option<i32>,
    pub offset_y: Option<i32>,
}

impl<B: Backend> Operation<B> for Copy<B>
where
    Copy<B>: Lower<B>,
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
        if let Some(w) = self.width {
            spec.width = w;
        }
        if let Some(h) = self.height {
            spec.height = h;
        }
        // For bands/format/interpretation, one should probably map properly, but we keep it simple.
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.width {
            state.write_i32(v);
        }
        if let Some(v) = self.height {
            state.write_i32(v);
        }
        if let Some(v) = self.bands {
            state.write_i32(v);
        }
        if let Some(v) = self.format {
            state.write_i32(v);
        }
        if let Some(v) = self.interpretation {
            state.write_i32(v);
        }
        if let Some(v) = self.horizontal_resolution {
            state.write_u64(v.to_bits());
        }
        if let Some(v) = self.vertical_resolution {
            state.write_u64(v.to_bits());
        }
        if let Some(v) = self.offset_x {
            state.write_i32(v);
        }
        if let Some(v) = self.offset_y {
            state.write_i32(v);
        }
    }
}

// ── TileCache ─────────────────────────────────────────────────────────────────

pub struct TileCache<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub tile_width: Option<i32>,
    pub tile_height: Option<i32>,
    pub maximum_tiles: Option<i32>,
    pub access: Option<Access>,
    pub threaded: Option<bool>,
    pub persistent: Option<bool>,
}

impl<B: Backend> Operation<B> for TileCache<B>
where
    TileCache<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.tile_width {
            state.write_i32(v);
        }
        if let Some(v) = self.tile_height {
            state.write_i32(v);
        }
        if let Some(v) = self.maximum_tiles {
            state.write_i32(v);
        }
        if let Some(v) = self.access {
            state.write_i32(v.into_vips());
        }
        if let Some(v) = self.threaded {
            state.write_u8(v as u8);
        }
        if let Some(v) = self.persistent {
            state.write_u8(v as u8);
        }
    }
}

// ── Basic unary ops ───────────────────────────────────────────────────────────

macro_rules! define_unary_op {
    ($name:ident, $vips_name:expr) => {
        pub struct $name<B: Backend> {
            pub input: Input<ImageKind, B>,
        }
        impl<B: Backend> Operation<B> for $name<B>
        where
            $name<B>: Lower<B>,
        {
            type Output = ImageKind;
            fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
                vec![&self.input]
            }
            fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
                vec![Some(WorkUnit::Region(out.clone()))]
            }
            fn output_spec(&self) -> ImageKind {
                (*self.input.spec).clone()
            }
            fn dyn_hash(&self, _state: &mut dyn Hasher) {}
        }
    };
}

define_unary_op!(Clamp, b"clamp\0");
define_unary_op!(ScaleImage, b"scale\0");
define_unary_op!(Wrap, b"wrap\0");
define_unary_op!(Sequential, b"sequential\0");
define_unary_op!(Autorotate, b"autorot\0");
define_unary_op!(Byteswap, b"byteswap\0");
define_unary_op!(Transpose3d, b"transpose3d\0");
define_unary_op!(Falsecolour, b"falsecolour\0");
define_unary_op!(MatrixInvert, b"matrixinvert\0");
define_unary_op!(Rad2float, b"rad2float\0");
define_unary_op!(Float2rad, b"float2rad\0");

// ── Msb ───────────────────────────────────────────────────────────────────────

pub struct Msb<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub band: Option<i32>,
}

impl<B: Backend> Operation<B> for Msb<B>
where
    Msb<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.band {
            state.write_i32(v);
        }
    }
}

// ── Maplut ────────────────────────────────────────────────────────────────────

pub struct Maplut<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub lut: Input<crate::data::lut::LutKind, B>,
    pub band: Option<i32>,
}

impl<B: Backend> Operation<B> for Maplut<B>
where
    Maplut<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.lut]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Range(Range {
                start: 0,
                end: self.lut.spec.entries as i32,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.band {
            state.write_i32(v);
        }
    }
}

// ── Recomb ────────────────────────────────────────────────────────────────────

pub struct Recomb<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub matrix: Input<crate::data::mask2d::Mask2DKind, B>,
}

impl<B: Backend> Operation<B> for Recomb<B>
where
    Recomb<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input, &self.matrix]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Region(Region {
                x: 0,
                y: 0,
                w: self.matrix.spec.width,
                h: self.matrix.spec.height,
                lod: out.lod,
            })),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── Ifthenelse ────────────────────────────────────────────────────────────────

pub struct Ifthenelse<B: Backend> {
    pub cond: Input<ImageKind, B>,
    pub if_true: Input<ImageKind, B>,
    pub if_false: Input<ImageKind, B>,
    pub blend: Option<bool>,
}

impl<B: Backend> Operation<B> for Ifthenelse<B>
where
    Ifthenelse<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.cond, &self.if_true, &self.if_false]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Region(out.clone())),
            Some(WorkUnit::Region(out.clone())),
        ]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.if_true.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.blend {
            state.write_u8(v as u8);
        }
    }
}

// ── Invertlut ─────────────────────────────────────────────────────────────────

pub struct Invertlut<B: Backend> {
    pub input: Input<crate::data::lut::LutKind, B>,
    pub size: Option<i32>,
}

impl<B: Backend> Operation<B> for Invertlut<B>
where
    Invertlut<B>: Lower<B>,
{
    type Output = crate::data::lut::LutKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &crate::work_unit::Range) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Range(out.clone()))]
    }
    fn output_spec(&self) -> crate::data::lut::LutKind {
        crate::data::lut::LutKind {
            entries: self.size.unwrap_or(256) as u32,
            bands: self.input.spec.bands,
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.size {
            state.write_i32(v);
        }
    }
}

// ── Linecache ─────────────────────────────────────────────────────────────────

pub struct Linecache<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub tile_height: Option<i32>,
    pub access: Option<Access>,
    pub threaded: Option<bool>,
    pub persistent: Option<bool>,
}

impl<B: Backend> Operation<B> for Linecache<B>
where
    Linecache<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        if let Some(v) = self.tile_height {
            state.write_i32(v);
        }
        if let Some(v) = self.access {
            state.write_i32(v.into_vips());
        }
        if let Some(v) = self.threaded {
            state.write_u8(v as u8);
        }
        if let Some(v) = self.persistent {
            state.write_u8(v as u8);
        }
    }
}

// ── Case ──────────────────────────────────────────────────────────────────────

pub struct Case<B: Backend> {
    pub input: Input<ImageKind, B>, // used as `index`
    pub cases: Vec<Input<ImageKind, B>>,
}

impl<B: Backend> Operation<B> for Case<B>
where
    Case<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        let mut res: Vec<&dyn AnyInput<B>> = vec![&self.input];
        for c in &self.cases {
            res.push(c as &dyn AnyInput<B>);
        }
        res
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let mut res = vec![Some(WorkUnit::Region(out.clone()))];
        for _ in 0..self.cases.len() {
            res.push(Some(WorkUnit::Region(out.clone())));
        }
        res
    }
    fn output_spec(&self) -> ImageKind {
        (*self.cases[0].spec).clone()
    }
    fn dyn_hash(&self, _state: &mut dyn Hasher) {}
}

// ── Exposure / Brightness / NoiseReduction ────────────────────────────────────

pub struct Exposure<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub stops: f32,
    pub preserve: f32,
}

impl<B: Backend> Operation<B> for Exposure<B>
where
    Exposure<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.stops.to_bits());
        state.write_u32(self.preserve.to_bits());
    }
}

pub struct Brightness<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub value: f32,
}

impl<B: Backend> Operation<B> for Brightness<B>
where
    Brightness<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.value.to_bits());
    }
}

pub struct NoiseReduction<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub strength: f32,
}

impl<B: Backend> NoiseReduction<B> {
    /// Median window side length: grows with `strength`, always odd, >= 1.
    pub fn size(&self) -> i32 {
        (1 + (self.strength * 4.0) as i32 * 2).max(1)
    }
}

impl<B: Backend> Operation<B> for NoiseReduction<B>
where
    NoiseReduction<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let halo = self.size() / 2;
        vec![Some(WorkUnit::Region(out.expanded(halo)))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.strength.to_bits());
    }
}

// ── Saturation ────────────────────────────────────────────────────────────────

pub struct Saturation<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub amount: f32,
}

impl<B: Backend> Operation<B> for Saturation<B>
where
    Saturation<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u32(self.amount.to_bits());
    }
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Cast<B>: crate::operation::Lower<B>,
{
    pub fn cast(&self, target: PixelLayout, shift: Option<bool>) -> Self {
        self.push(Cast {
            input: self.as_input(),
            target,
            shift,
        })
    }

    /// Casts only the sample storage, keeping model/alpha/color-space.
    pub fn cast_storage(&self, storage: crate::pixel::Storage, shift: Option<bool>) -> Self {
        self.cast(self.spec.layout.with_storage(storage), shift)
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Copy<B>: crate::operation::Lower<B>,
{
    pub fn copy(
        &self,
        width: Option<i32>,
        height: Option<i32>,
        bands: Option<i32>,
        format: Option<i32>,
        interpretation: Option<i32>,
        horizontal_resolution: Option<f64>,
        vertical_resolution: Option<f64>,
        offset_x: Option<i32>,
        offset_y: Option<i32>,
    ) -> Self {
        self.push(Copy {
            input: self.as_input(),
            width,
            height,
            bands,
            format,
            interpretation,
            horizontal_resolution,
            vertical_resolution,
            offset_x,
            offset_y,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    TileCache<B>: crate::operation::Lower<B>,
{
    pub fn tile_cache(
        &self,
        tile_width: Option<i32>,
        tile_height: Option<i32>,
        maximum_tiles: Option<i32>,
        access: Option<Access>,
        threaded: Option<bool>,
        persistent: Option<bool>,
    ) -> Self {
        self.push(TileCache {
            input: self.as_input(),
            tile_width,
            tile_height,
            maximum_tiles,
            access,
            threaded,
            persistent,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Msb<B>: crate::operation::Lower<B>,
{
    pub fn msb(&self, band: Option<i32>) -> Self {
        self.push(Msb {
            input: self.as_input(),
            band,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Maplut<B>: crate::operation::Lower<B>,
{
    pub fn maplut(&self, lut: Input<crate::data::lut::LutKind, B>, band: Option<i32>) -> Self {
        self.push(Maplut {
            input: self.as_input(),
            lut,
            band,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Recomb<B>: crate::operation::Lower<B>,
{
    pub fn recomb(&self, matrix: Input<crate::data::mask2d::Mask2DKind, B>) -> Self {
        self.push(Recomb {
            input: self.as_input(),
            matrix,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::lut::Lut<B>
where
    Invertlut<B>: crate::operation::Lower<B>,
{
    pub fn invertlut(&self, size: Option<i32>) -> Self {
        self.push(Invertlut {
            input: self.as_input(),
            size,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Linecache<B>: crate::operation::Lower<B>,
{
    pub fn linecache(
        &self,
        tile_height: Option<i32>,
        access: Option<Access>,
        threaded: Option<bool>,
        persistent: Option<bool>,
    ) -> Self {
        self.push(Linecache {
            input: self.as_input(),
            tile_height,
            access,
            threaded,
            persistent,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Case<B>: crate::operation::Lower<B>,
{
    pub fn case(&self, cases: Vec<Input<ImageKind, B>>) -> Self {
        self.push(Case {
            input: self.as_input(),
            cases,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Ifthenelse<B>: crate::operation::Lower<B>,
{
    pub fn ifthenelse(
        &self,
        if_true: Input<ImageKind, B>,
        if_false: Input<ImageKind, B>,
        blend: Option<bool>,
    ) -> Self {
        self.push(Ifthenelse {
            cond: self.as_input(),
            if_true,
            if_false,
            blend,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Exposure<B>: crate::operation::Lower<B>,
{
    pub fn exposure(&self, stops: f32, preserve: f32) -> Self {
        self.push(Exposure {
            input: self.as_input(),
            stops,
            preserve,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Brightness<B>: crate::operation::Lower<B>,
{
    pub fn brightness(&self, value: f32) -> Self {
        self.push(Brightness {
            input: self.as_input(),
            value,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    NoiseReduction<B>: crate::operation::Lower<B>,
{
    pub fn noise_reduction(&self, strength: f32) -> Self {
        self.push(NoiseReduction {
            input: self.as_input(),
            strength,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Saturation<B>: crate::operation::Lower<B>,
{
    pub fn saturation(&self, amount: f32) -> Self {
        self.push(Saturation {
            input: self.as_input(),
            amount,
        })
    }
}

// ── Boolean and Relational operations ────────────────────────────────────────

pub struct Boolean<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
    pub boolean_op: OperationBoolean,
}
impl<B: Backend> Operation<B> for Boolean<B>
where
    Boolean<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.boolean_op.into_vips());
    }
}

pub struct Relational<B: Backend> {
    pub left: Input<ImageKind, B>,
    pub right: Input<ImageKind, B>,
    pub relational: OperationRelational,
}
impl<B: Backend> Operation<B> for Relational<B>
where
    Relational<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.left, &self.right]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone())); 2]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.left.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.relational.into_vips());
    }
}

pub struct BooleanConst<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub boolean_op: OperationBoolean,
    pub c: Vec<f64>,
}
impl<B: Backend> Operation<B> for BooleanConst<B>
where
    BooleanConst<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.boolean_op.into_vips());
        for &v in &self.c {
            state.write(&v.to_le_bytes());
        }
    }
}

pub struct RelationalConst<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub relational: OperationRelational,
    pub c: Vec<f64>,
}
impl<B: Backend> Operation<B> for RelationalConst<B>
where
    RelationalConst<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(out.clone()))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.relational.into_vips());
        for &v in &self.c {
            state.write(&v.to_le_bytes());
        }
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Boolean<B>: crate::operation::Lower<B>,
{
    pub fn boolean(
        &self,
        right: &crate::data::image::Image2D<B>,
        boolean_op: OperationBoolean,
    ) -> Self {
        self.push(Boolean {
            left: self.as_input(),
            right: right.as_input(),
            boolean_op,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Relational<B>: crate::operation::Lower<B>,
{
    pub fn relational(
        &self,
        right: &crate::data::image::Image2D<B>,
        relational: OperationRelational,
    ) -> Self {
        self.push(Relational {
            left: self.as_input(),
            right: right.as_input(),
            relational,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    BooleanConst<B>: crate::operation::Lower<B>,
{
    pub fn boolean_const(&self, boolean_op: OperationBoolean, c: Vec<f64>) -> Self {
        self.push(BooleanConst {
            input: self.as_input(),
            boolean_op,
            c,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    RelationalConst<B>: crate::operation::Lower<B>,
{
    pub fn relational_const(&self, relational: OperationRelational, c: Vec<f64>) -> Self {
        self.push(RelationalConst {
            input: self.as_input(),
            relational,
            c,
        })
    }
}
