use std::hash::Hasher;
use crate::operation::IntoVipsEnum;

use crate::backend::Backend;
use crate::data::histogram::HistogramKind;
use crate::data::image::{Image2D, ImageKind};
use crate::operation::{AnyInput, Input, Lower, Operation};
use crate::work_unit::{Atomic, Lod, Region, WorkUnit};

// ── Remap (zero-cost index-remapping view adapter) ──────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RemapKind {
    Identity = 0,
    FlipH = 1,
    FlipV = 2,
    Rot180 = 3,
    Scale = 4,
    Tile = 5,
    Translate = 6,
    Rot90 = 7,
    Rot270 = 8,
}

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RemapParams {
    pub out_w: u32,
    pub out_h: u32,
    pub sx: f32,
    pub sy: f32,
    pub in_w: u32,
    pub in_h: u32,
    pub tx: i32,
    pub ty: i32,
}

impl Default for RemapParams {
    fn default() -> Self {
        Self {
            out_w: 0,
            out_h: 0,
            sx: 1.0,
            sy: 1.0,
            in_w: 0,
            in_h: 0,
            tx: 0,
            ty: 0,
        }
    }
}

// ── Enums ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kernel {
    Nearest,
    Linear,
    Cubic,
    Mitchell,
    Lanczos2,
    Lanczos3,
}
impl IntoVipsEnum for Kernel {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Direction {
    Horizontal,
    Vertical,
}
impl IntoVipsEnum for Direction {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Angle {
    D0,
    D90,
    D180,
    D270,
}
impl IntoVipsEnum for Angle {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Angle45 {
    D0,
    D45,
    D90,
    D135,
    D180,
    D225,
    D270,
    D315,
}
impl IntoVipsEnum for Angle45 {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Extend {
    Black,
    Copy,
    Repeat,
    Mirror,
    White,
    Background,
}
impl IntoVipsEnum for Extend {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Interesting {
    None,
    Centre,
    Entropy,
    Attention,
    Low,
    High,
    All,
}
impl IntoVipsEnum for Interesting {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompassDirection {
    Centre,
    North,
    East,
    South,
    West,
    NorthEast,
    SouthEast,
    SouthWest,
    NorthWest,
}
impl IntoVipsEnum for CompassDirection {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Size {
    Both,
    Up,
    Down,
    Force,
}
impl IntoVipsEnum for Size {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

// ── Operations ────────────────────────────────────────────────────────────────

pub struct Crop<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}
impl<B: Backend> Operation<B> for Crop<B>
where
    Crop<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(Region {
            x: out.x + self.left,
            y: out.y + self.top,
            w: out.w,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width = self.width;
        spec.height = self.height;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.left);
        state.write_i32(self.top);
        state.write_i32(self.width);
        state.write_i32(self.height);
    }
}

// ── GPU Lowering ──────────────────────────────────────────────────────────────

pub struct Embed<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
    pub extend: Option<Extend>,
    pub background: Option<[f64; 3]>,
}
impl<B: Backend> Operation<B> for Embed<B>
where
    Embed<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(Region {
            x: out.x - self.x,
            y: out.y - self.y,
            w: out.w,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width = self.width;
        spec.height = self.height;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.x);
        state.write_i32(self.y);
        state.write_i32(self.width);
        state.write_i32(self.height);
        if let Some(e) = self.extend {
            state.write_i32(e.into_vips());
        }
    }
}

pub struct Flip<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub direction: Direction,
}
impl<B: Backend> Operation<B> for Flip<B>
where
    Flip<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let spec = &*self.input.spec;
        let (x, y) = match self.direction {
            Direction::Horizontal => (spec.width - out.x - out.w, out.y),
            Direction::Vertical => (out.x, spec.height - out.y - out.h),
        };
        vec![Some(WorkUnit::Region(Region {
            x,
            y,
            w: out.w,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.direction.into_vips());
    }
}

pub struct Rot90<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub angle: Angle,
}
impl<B: Backend> Operation<B> for Rot90<B>
where
    Rot90<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let spec = &*self.input.spec;
        let (w, h) = (spec.width, spec.height);
        let region = match self.angle {
            Angle::D0 => out.clone(),
            Angle::D90 => Region {
                x: out.y,
                y: h - out.x - out.w,
                w: out.h,
                h: out.w,
                lod: out.lod,
            },
            Angle::D180 => Region {
                x: w - out.x - out.w,
                y: h - out.y - out.h,
                w: out.w,
                h: out.h,
                lod: out.lod,
            },
            Angle::D270 => Region {
                x: w - out.y - out.h,
                y: out.x,
                w: out.h,
                h: out.w,
                lod: out.lod,
            },
        };
        vec![Some(WorkUnit::Region(region))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        if matches!(self.angle, Angle::D90 | Angle::D270) {
            std::mem::swap(&mut spec.width, &mut spec.height);
        }
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.angle.into_vips());
    }
}

pub struct Rot45<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub angle: Angle45,
}
impl<B: Backend> Operation<B> for Rot45<B>
where
    Rot45<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let spec = &*self.input.spec;
        let (w, h) = (spec.width, spec.height);
        match self.angle {
            Angle45::D0 => vec![Some(WorkUnit::Region(out.clone()))],
            Angle45::D180 => vec![Some(WorkUnit::Region(Region {
                x: w - out.x - out.w,
                y: h - out.y - out.h,
                w: out.w,
                h: out.h,
                lod: out.lod,
            }))],
            _ => vec![Some(WorkUnit::Region(Region::full((w, h), out.lod)))],
        }
    }
    fn output_spec(&self) -> ImageKind {
        let spec = (*self.input.spec).clone();
        match self.angle {
            Angle45::D0 | Angle45::D90 | Angle45::D180 | Angle45::D270 => spec,
            _ => {
                let mut spec = spec;
                let diag = ((spec.width as f64 + spec.height as f64) / std::f64::consts::SQRT_2)
                    .ceil() as i32;
                spec.width = diag;
                spec.height = diag;
                spec
            }
        }
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.angle.into_vips());
    }
}

pub struct Rotate<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub angle: f64,
    pub background: Option<[f64; 3]>,
    pub offset_input_x: Option<f64>,
    pub offset_input_y: Option<f64>,
    pub offset_output_x: Option<f64>,
    pub offset_output_y: Option<f64>,
}
impl<B: Backend> Operation<B> for Rotate<B>
where
    Rotate<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let spec = &*self.input.spec;
        vec![Some(WorkUnit::Region(Region::full(
            (spec.width, spec.height),
            out.lod,
        )))]
    }
    fn output_spec(&self) -> ImageKind {
        (*self.input.spec).clone()
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.angle.to_bits());
        state.write_u64(self.offset_input_x.unwrap_or(0.0).to_bits());
        state.write_u64(self.offset_input_y.unwrap_or(0.0).to_bits());
        state.write_u64(self.offset_output_x.unwrap_or(0.0).to_bits());
        state.write_u64(self.offset_output_y.unwrap_or(0.0).to_bits());
    }
}

pub struct Smartcrop<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub width: i32,
    pub height: i32,
    pub interesting: Option<Interesting>,
}
impl<B: Backend> Operation<B> for Smartcrop<B>
where
    Smartcrop<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let spec = &*self.input.spec;
        vec![Some(WorkUnit::Region(Region::full(
            (spec.width, spec.height),
            out.lod,
        )))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width = self.width;
        spec.height = self.height;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height);
        if let Some(i) = self.interesting {
            state.write_i32(i.into_vips());
        }
    }
}

// ── SmartcropColScore / SmartcropRowScore: per-column/row "interestingness" ───

/// Sum of `smartcrop_score` (a cheap local-gradient detail proxy) over each
/// column, as a `width`-bin [`HistogramKind`]. Feeds [`Smartcrop::lower`]'s
/// crop-window search — not vips' own ENTROPY/ATTENTION heuristic, but a
/// reasonable approximation for picking a high-detail crop window.

/// Same as [`SmartcropColScore`], summed per row into a `height`-bin
/// [`HistogramKind`].

/// Slides a `window`-wide window over `scores` and returns the start offset
/// that maximizes (or, if `minimize`, minimizes) the windowed sum.
fn smartcrop_best_offset(scores: &[u32], window: usize, minimize: bool) -> usize {
    if window == 0 || window >= scores.len() {
        return 0;
    }
    let mut sum: u64 = scores[..window].iter().map(|&v| v as u64).sum();
    let mut best = sum;
    let mut best_off = 0;
    for i in 1..=(scores.len() - window) {
        sum += scores[i + window - 1] as u64;
        sum -= scores[i - 1] as u64;
        let better = if minimize { sum < best } else { sum > best };
        if better {
            best = sum;
            best_off = i;
        }
    }
    best_off
}

pub struct Gravity<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub direction: CompassDirection,
    pub width: i32,
    pub height: i32,
    pub extend: Option<Extend>,
    pub background: Option<[f64; 3]>,
}
impl<B: Backend> Operation<B> for Gravity<B>
where
    Gravity<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let spec = &*self.input.spec;
        let (old_w, old_h) = (spec.width, spec.height);
        let (new_w, new_h) = (self.width, self.height);
        let (ox, oy) = match self.direction {
            CompassDirection::Centre => ((new_w - old_w) / 2, (new_h - old_h) / 2),
            CompassDirection::North => ((new_w - old_w) / 2, 0),
            CompassDirection::South => ((new_w - old_w) / 2, new_h - old_h),
            CompassDirection::East => (new_w - old_w, (new_h - old_h) / 2),
            CompassDirection::West => (0, (new_h - old_h) / 2),
            CompassDirection::NorthEast => (new_w - old_w, 0),
            CompassDirection::NorthWest => (0, 0),
            CompassDirection::SouthEast => (new_w - old_w, new_h - old_h),
            CompassDirection::SouthWest => (0, new_h - old_h),
        };
        vec![Some(WorkUnit::Region(Region {
            x: out.x - ox,
            y: out.y - oy,
            w: out.w,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width = self.width;
        spec.height = self.height;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.direction.into_vips());
        state.write_i32(self.width);
        state.write_i32(self.height);
        if let Some(e) = self.extend {
            state.write_i32(e.into_vips());
        }
    }
}

pub struct Resize<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub scale: f64,
    pub kernel: Option<Kernel>,
    pub vertical_scale: Option<f64>,
    pub gap: Option<f64>,
}
impl<B: Backend> Operation<B> for Resize<B>
where
    Resize<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let hscale = self.scale;
        let vscale = self.vertical_scale.unwrap_or(self.scale);
        vec![Some(WorkUnit::Region(Region {
            x: (out.x as f64 / hscale).floor() as i32,
            y: (out.y as f64 / vscale).floor() as i32,
            w: ((out.x + out.w) as f64 / hscale).ceil() as i32
                - (out.x as f64 / hscale).floor() as i32,
            h: ((out.y + out.h) as f64 / vscale).ceil() as i32
                - (out.y as f64 / vscale).floor() as i32,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        let hscale = self.scale;
        let vscale = self.vertical_scale.unwrap_or(self.scale);
        spec.width = (spec.width as f64 * hscale).round() as i32;
        spec.height = (spec.height as f64 * vscale).round() as i32;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.scale.to_bits());
        state.write_u64(self.vertical_scale.unwrap_or(0.0).to_bits());
        state.write_u64(self.gap.unwrap_or(0.0).to_bits());
        if let Some(k) = self.kernel {
            state.write_i32(k.into_vips());
        }
    }
}

pub struct Thumbnail<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub width: i32,
    pub height: Option<i32>,
    pub size: Option<i32>,
    pub crop: Option<Interesting>,
    pub linear: Option<bool>,
    pub auto_rotate: Option<bool>,
    pub no_rotate: Option<bool>,
    pub import_profile: Option<String>,
    pub export_profile: Option<String>,
    pub intent: Option<i32>,
    pub fail_on: Option<i32>,
}
impl<B: Backend> Operation<B> for Thumbnail<B>
where
    Thumbnail<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let spec = &*self.input.spec;
        vec![Some(WorkUnit::Region(Region::full(
            (spec.width, spec.height),
            out.lod,
        )))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        let (in_w, in_h) = (spec.width as f64, spec.height as f64);
        let target_w = self.width as f64;
        let (out_w, out_h) = match self.height {
            Some(h) => {
                let target_h = h as f64;
                let scale = (target_w / in_w).min(target_h / in_h);
                (in_w * scale, in_h * scale)
            }
            None => {
                let scale = target_w / in_w;
                (in_w * scale, in_h * scale)
            }
        };
        spec.width = out_w.round().max(1.0) as i32;
        spec.height = out_h.round().max(1.0) as i32;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.width);
        state.write_i32(self.height.unwrap_or(0));
        state.write_i32(self.size.unwrap_or(0));
        if let Some(c) = self.crop {
            state.write_i32(c.into_vips());
        }
        state.write_i32(self.intent.unwrap_or(0));
    }
}

pub struct Shrink<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub horizontal: f64,
    pub vertical: f64,
    pub ceil: Option<bool>,
}
impl<B: Backend> Operation<B> for Shrink<B>
where
    Shrink<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let hf = self.horizontal.ceil() as i32;
        let vf = self.vertical.ceil() as i32;
        vec![Some(WorkUnit::Region(Region {
            x: out.x * hf,
            y: out.y * vf,
            w: out.w * hf,
            h: out.h * vf,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        let round = |dim: i32, factor: f64| -> i32 {
            let scaled = dim as f64 / factor;
            match self.ceil {
                Some(true) => scaled.ceil() as i32,
                _ => scaled.floor() as i32,
            }
            .max(1)
        };
        spec.width = round(spec.width, self.horizontal);
        spec.height = round(spec.height, self.vertical);
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.horizontal.to_bits());
        state.write_u64(self.vertical.to_bits());
    }
}

pub struct Reduce<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub horizontal: f64,
    pub vertical: f64,
    pub kernel: Option<Kernel>,
    pub gap: Option<f64>,
}
impl<B: Backend> Operation<B> for Reduce<B>
where
    Reduce<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let hf = self.horizontal;
        let vf = self.vertical;
        vec![Some(WorkUnit::Region(Region {
            x: (out.x as f64 * hf).floor() as i32,
            y: (out.y as f64 * vf).floor() as i32,
            w: ((out.x + out.w) as f64 * hf).ceil() as i32 - (out.x as f64 * hf).floor() as i32,
            h: ((out.y + out.h) as f64 * vf).ceil() as i32 - (out.y as f64 * vf).floor() as i32,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width = (spec.width as f64 / self.horizontal).floor() as i32;
        spec.height = (spec.height as f64 / self.vertical).floor() as i32;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.horizontal.to_bits());
        state.write_u64(self.vertical.to_bits());
        if let Some(k) = self.kernel {
            state.write_i32(k.into_vips());
        }
        state.write_u64(self.gap.unwrap_or(0.0).to_bits());
    }
}

pub struct ReduceHorizontal<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub shrink: f64,
    pub kernel: Option<Kernel>,
    pub gap: Option<f64>,
}
impl<B: Backend> Operation<B> for ReduceHorizontal<B>
where
    ReduceHorizontal<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let hf = self.shrink;
        vec![Some(WorkUnit::Region(Region {
            x: (out.x as f64 * hf).floor() as i32,
            y: out.y,
            w: ((out.x + out.w) as f64 * hf).ceil() as i32 - (out.x as f64 * hf).floor() as i32,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width = (spec.width as f64 / self.shrink).floor() as i32;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.shrink.to_bits());
        if let Some(k) = self.kernel {
            state.write_i32(k.into_vips());
        }
        state.write_u64(self.gap.unwrap_or(0.0).to_bits());
    }
}

pub struct ReduceVertical<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub shrink: f64,
    pub kernel: Option<Kernel>,
    pub gap: Option<f64>,
}
impl<B: Backend> Operation<B> for ReduceVertical<B>
where
    ReduceVertical<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let vf = self.shrink;
        vec![Some(WorkUnit::Region(Region {
            x: out.x,
            y: (out.y as f64 * vf).floor() as i32,
            w: out.w,
            h: ((out.y + out.h) as f64 * vf).ceil() as i32 - (out.y as f64 * vf).floor() as i32,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.height = (spec.height as f64 / self.shrink).floor() as i32;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_u64(self.shrink.to_bits());
        if let Some(k) = self.kernel {
            state.write_i32(k.into_vips());
        }
        state.write_u64(self.gap.unwrap_or(0.0).to_bits());
    }
}

pub struct ShrinkHorizontal<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub shrink: i32,
    pub ceil: Option<bool>,
}
impl<B: Backend> Operation<B> for ShrinkHorizontal<B>
where
    ShrinkHorizontal<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let hf = self.shrink;
        vec![Some(WorkUnit::Region(Region {
            x: out.x * hf,
            y: out.y,
            w: out.w * hf,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width = (spec.width + self.shrink - 1) / self.shrink;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.shrink);
    }
}

pub struct ShrinkVertical<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub shrink: i32,
    pub ceil: Option<bool>,
}
impl<B: Backend> Operation<B> for ShrinkVertical<B>
where
    ShrinkVertical<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let vf = self.shrink;
        vec![Some(WorkUnit::Region(Region {
            x: out.x,
            y: out.y * vf,
            w: out.w,
            h: out.h * vf,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.height = (spec.height + self.shrink - 1) / self.shrink;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.shrink);
    }
}

pub struct ExtractArea<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}
impl<B: Backend> Operation<B> for ExtractArea<B>
where
    ExtractArea<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(Region {
            x: out.x + self.left,
            y: out.y + self.top,
            w: out.w,
            h: out.h,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width = self.width;
        spec.height = self.height;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.left);
        state.write_i32(self.top);
        state.write_i32(self.width);
        state.write_i32(self.height);
    }
}

pub struct Subsample<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub horizontal: i32,
    pub vertical: i32,
    pub point: Option<bool>,
}
impl<B: Backend> Operation<B> for Subsample<B>
where
    Subsample<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(Region {
            x: out.x * self.horizontal,
            y: out.y * self.vertical,
            w: out.w * self.horizontal,
            h: out.h * self.vertical,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width = (spec.width + self.horizontal - 1) / self.horizontal;
        spec.height = (spec.height + self.vertical - 1) / self.vertical;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.horizontal);
        state.write_i32(self.vertical);
    }
}

pub struct Zoom<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub horizontal: i32,
    pub vertical: i32,
}
impl<B: Backend> Operation<B> for Zoom<B>
where
    Zoom<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        vec![Some(WorkUnit::Region(Region {
            x: out.x / self.horizontal,
            y: out.y / self.vertical,
            w: (out.x + out.w + self.horizontal - 1) / self.horizontal - out.x / self.horizontal,
            h: (out.y + out.h + self.vertical - 1) / self.vertical - out.y / self.vertical,
            lod: out.lod,
        }))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width *= self.horizontal;
        spec.height *= self.vertical;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.horizontal);
        state.write_i32(self.vertical);
    }
}

pub struct Replicate<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub across: i32,
    pub down: i32,
}
impl<B: Backend> Operation<B> for Replicate<B>
where
    Replicate<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let spec = &*self.input.spec;
        let (w, h) = (spec.width, spec.height);
        let x0 = out.x.rem_euclid(w);
        let y0 = out.y.rem_euclid(h);
        if x0 + out.w <= w && y0 + out.h <= h {
            vec![Some(WorkUnit::Region(Region {
                x: x0,
                y: y0,
                w: out.w,
                h: out.h,
                lod: out.lod,
            }))]
        } else {
            vec![Some(WorkUnit::Region(Region::full((w, h), out.lod)))]
        }
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width *= self.across;
        spec.height *= self.down;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.across);
        state.write_i32(self.down);
    }
}

pub struct Grid<B: Backend> {
    pub input: Input<ImageKind, B>,
    pub tile_height: i32,
    pub across: i32,
    pub down: i32,
}
impl<B: Backend> Operation<B> for Grid<B>
where
    Grid<B>: Lower<B>,
{
    type Output = ImageKind;
    fn inputs(&self) -> Vec<&dyn AnyInput<B>> {
        vec![&self.input]
    }
    fn demand(&self, out: &Region) -> Vec<Option<WorkUnit>> {
        let spec = &*self.input.spec;
        vec![Some(WorkUnit::Region(Region::full(
            (spec.width, spec.height),
            out.lod,
        )))]
    }
    fn output_spec(&self) -> ImageKind {
        let mut spec = (*self.input.spec).clone();
        spec.width *= self.across;
        spec.height = self.tile_height * self.down;
        spec
    }
    fn dyn_hash(&self, state: &mut dyn Hasher) {
        state.write_i32(self.tile_height);
        state.write_i32(self.across);
        state.write_i32(self.down);
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Crop<B>: crate::operation::Lower<B>,
{
    pub fn crop(&self, left: i32, top: i32, width: i32, height: i32) -> Self {
        self.push(Crop {
            input: self.as_input(),
            left,
            top,
            width,
            height,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Embed<B>: crate::operation::Lower<B>,
{
    pub fn embed(
        &self,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        extend: Option<Extend>,
        background: Option<[f64; 3]>,
    ) -> Self {
        self.push(Embed {
            input: self.as_input(),
            x,
            y,
            width,
            height,
            extend,
            background,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Flip<B>: crate::operation::Lower<B>,
{
    pub fn flip(&self, direction: Direction) -> Self {
        self.push(Flip {
            input: self.as_input(),
            direction,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Rot90<B>: crate::operation::Lower<B>,
{
    pub fn rot90(&self, angle: Angle) -> Self {
        self.push(Rot90 {
            input: self.as_input(),
            angle,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Rot45<B>: crate::operation::Lower<B>,
{
    pub fn rot45(&self, angle: Angle45) -> Self {
        self.push(Rot45 {
            input: self.as_input(),
            angle,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Rotate<B>: crate::operation::Lower<B>,
{
    pub fn rotate(
        &self,
        angle: f64,
        background: Option<[f64; 3]>,
        offset_input_x: Option<f64>,
        offset_input_y: Option<f64>,
        offset_output_x: Option<f64>,
        offset_output_y: Option<f64>,
    ) -> Self {
        self.push(Rotate {
            input: self.as_input(),
            angle,
            background,
            offset_input_x,
            offset_input_y,
            offset_output_x,
            offset_output_y,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Smartcrop<B>: crate::operation::Lower<B>,
{
    pub fn smartcrop(&self, width: i32, height: i32, interesting: Option<Interesting>) -> Self {
        self.push(Smartcrop {
            input: self.as_input(),
            width,
            height,
            interesting,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Gravity<B>: crate::operation::Lower<B>,
{
    pub fn gravity(
        &self,
        direction: CompassDirection,
        width: i32,
        height: i32,
        extend: Option<Extend>,
        background: Option<[f64; 3]>,
    ) -> Self {
        self.push(Gravity {
            input: self.as_input(),
            direction,
            width,
            height,
            extend,
            background,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Resize<B>: crate::operation::Lower<B>,
{
    pub fn resize(
        &self,
        scale: f64,
        kernel: Option<Kernel>,
        vertical_scale: Option<f64>,
        gap: Option<f64>,
    ) -> Self {
        self.push(Resize {
            input: self.as_input(),
            scale,
            kernel,
            vertical_scale,
            gap,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Thumbnail<B>: crate::operation::Lower<B>,
{
    pub fn thumbnail(
        &self,
        width: i32,
        height: Option<i32>,
        size: Option<i32>,
        crop: Option<Interesting>,
        linear: Option<bool>,
        auto_rotate: Option<bool>,
        no_rotate: Option<bool>,
        import_profile: Option<String>,
        export_profile: Option<String>,
        intent: Option<i32>,
        fail_on: Option<i32>,
    ) -> Self {
        self.push(Thumbnail {
            input: self.as_input(),
            width,
            height,
            size,
            crop,
            linear,
            auto_rotate,
            no_rotate,
            import_profile,
            export_profile,
            intent,
            fail_on,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Shrink<B>: crate::operation::Lower<B>,
{
    pub fn shrink(&self, horizontal: f64, vertical: f64, ceil: Option<bool>) -> Self {
        self.push(Shrink {
            input: self.as_input(),
            horizontal,
            vertical,
            ceil,
        })
    }

    /// Downsamples to the given mip level via box `shrink` by
    /// `lod.scale_factor()`. `Lod(0)` is a no-op (returns `self`).
    pub fn with_lod(&self, lod: crate::work_unit::Lod) -> Self {
        let factor = lod.scale_factor();
        if factor <= 1 {
            return self.clone();
        }
        self.shrink(factor as f64, factor as f64, None)
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Reduce<B>: crate::operation::Lower<B>,
{
    pub fn reduce(
        &self,
        horizontal: f64,
        vertical: f64,
        kernel: Option<Kernel>,
        gap: Option<f64>,
    ) -> Self {
        self.push(Reduce {
            input: self.as_input(),
            horizontal,
            vertical,
            kernel,
            gap,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    ReduceHorizontal<B>: crate::operation::Lower<B>,
{
    pub fn reduce_horizontal(&self, shrink: f64, kernel: Option<Kernel>, gap: Option<f64>) -> Self {
        self.push(ReduceHorizontal {
            input: self.as_input(),
            shrink,
            kernel,
            gap,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    ReduceVertical<B>: crate::operation::Lower<B>,
{
    pub fn reduce_vertical(&self, shrink: f64, kernel: Option<Kernel>, gap: Option<f64>) -> Self {
        self.push(ReduceVertical {
            input: self.as_input(),
            shrink,
            kernel,
            gap,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    ShrinkHorizontal<B>: crate::operation::Lower<B>,
{
    pub fn shrink_horizontal(&self, shrink: i32, ceil: Option<bool>) -> Self {
        self.push(ShrinkHorizontal {
            input: self.as_input(),
            shrink,
            ceil,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    ShrinkVertical<B>: crate::operation::Lower<B>,
{
    pub fn shrink_vertical(&self, shrink: i32, ceil: Option<bool>) -> Self {
        self.push(ShrinkVertical {
            input: self.as_input(),
            shrink,
            ceil,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    ExtractArea<B>: crate::operation::Lower<B>,
{
    pub fn extract_area(&self, left: i32, top: i32, width: i32, height: i32) -> Self {
        self.push(ExtractArea {
            input: self.as_input(),
            left,
            top,
            width,
            height,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Subsample<B>: crate::operation::Lower<B>,
{
    pub fn subsample(&self, horizontal: i32, vertical: i32, point: Option<bool>) -> Self {
        self.push(Subsample {
            input: self.as_input(),
            horizontal,
            vertical,
            point,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Zoom<B>: crate::operation::Lower<B>,
{
    pub fn zoom(&self, horizontal: i32, vertical: i32) -> Self {
        self.push(Zoom {
            input: self.as_input(),
            horizontal,
            vertical,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Replicate<B>: crate::operation::Lower<B>,
{
    pub fn replicate(&self, across: i32, down: i32) -> Self {
        self.push(Replicate {
            input: self.as_input(),
            across,
            down,
        })
    }
}

impl<B: crate::backend::Backend> crate::data::image::Image2D<B>
where
    Grid<B>: crate::operation::Lower<B>,
{
    pub fn grid(&self, tile_height: i32, across: i32, down: i32) -> Self {
        self.push(Grid {
            input: self.as_input(),
            tile_height,
            across,
            down,
        })
    }
}
