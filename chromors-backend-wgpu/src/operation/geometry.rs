use std::hash::Hasher;

use chromors_core::backend::Backend;
use crate::view::{ParamBlock, ViewAdapter};
use crate::{GpuBackend, GpuBuilder, GpuView};
use chromors_core::operation::geometry::*;
use chromors_core::operation::{Lower, Operation};
use chromors_core::IntoVipsEnum;
use chromors_core::work_unit::{Region, WorkUnit};
use bytemuck;

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

/// Wraps a node's value in `RemapView<{inner}>` — a zero-cost index-remapping
/// `IRegion` (flip/rotate/scale/tile/translate). `kind` selects the remap
/// formula; `geo` carries its parameters as a `RemapGeo` std430 struct.
pub fn remap_adapter(kind: RemapKind, geo: RemapParams) -> ViewAdapter {
    ViewAdapter {
        wrapper: "RemapView<{inner}>".into(),
        ctor: "{ {value}, {params}[0].{p}_kind, {params}[0].{p}_geo }".into(),
        params: ParamBlock::scalar("{p}_kind", kind as u32).field("{p}_geo", "RemapGeo", geo),
        module: "lib.region",
    }
}

use chromors_core::operation::geometry::{Kernel, Direction, Angle, Angle45, Extend, Interesting, CompassDirection, Size};
use chromors_core::operation::geometry::*;

















// ── Operations ────────────────────────────────────────────────────────────────





// ── GPU Lowering ──────────────────────────────────────────────────────────────


impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Crop<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.adapt(remap_adapter(
            RemapKind::Translate,
            RemapParams {
                tx: self.left,
                ty: self.top,
                ..Default::default()
            },
        ));
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::ExtractArea<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.adapt(remap_adapter(
            RemapKind::Translate,
            RemapParams {
                tx: self.left,
                ty: self.top,
                ..Default::default()
            },
        ));
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Flip<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let kind = match self.direction {
            Direction::Horizontal => RemapKind::FlipH,
            Direction::Vertical => RemapKind::FlipV,
        };
        let out_spec = self.output_spec();
        cx.adapt(remap_adapter(
            kind,
            RemapParams {
                out_w: out_spec.width as u32,
                out_h: out_spec.height as u32,
                ..Default::default()
            },
        ));
        cx.output(out_spec.output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Rot90<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let out_spec = self.output_spec();
        match self.angle {
            Angle::D0 => {
                cx.adapt(remap_adapter(RemapKind::Identity, Default::default()));
            }
            Angle::D90 => {
                cx.adapt(remap_adapter(
                    RemapKind::Rot90,
                    RemapParams {
                        out_w: out_spec.width as u32,
                        ..Default::default()
                    },
                ));
            }
            Angle::D180 => {
                cx.adapt(remap_adapter(
                    RemapKind::Rot180,
                    RemapParams {
                        out_w: out_spec.width as u32,
                        out_h: out_spec.height as u32,
                        ..Default::default()
                    },
                ));
            }
            Angle::D270 => {
                cx.adapt(remap_adapter(
                    RemapKind::Rot270,
                    RemapParams {
                        out_h: out_spec.height as u32,
                        ..Default::default()
                    },
                ));
            }
        }
        cx.output(out_spec.output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Subsample<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.adapt(remap_adapter(
            RemapKind::Scale,
            RemapParams {
                sx: self.horizontal as f32,
                sy: self.vertical as f32,
                ..Default::default()
            },
        ));
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Zoom<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.adapt(remap_adapter(
            RemapKind::Scale,
            RemapParams {
                sx: 1.0 / (self.horizontal as f32),
                sy: 1.0 / (self.vertical as f32),
                ..Default::default()
            },
        ));
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Replicate<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let in_spec = &*self.input.spec;
        cx.adapt(remap_adapter(
            RemapKind::Tile,
            RemapParams {
                in_w: in_spec.width as u32,
                in_h: in_spec.height as u32,
                ..Default::default()
            },
        ));
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Resize<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let hscale = self.scale;
        let vscale = self.vertical_scale.unwrap_or(self.scale);
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("inv_hscale", (1.0 / hscale) as f32)
                .param("inv_vscale", (1.0 / vscale) as f32),
        );
        cx.kernel("ops.geometry", "resize_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Reduce<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("inv_hscale", self.horizontal as f32)
                .param("inv_vscale", self.vertical as f32),
        );
        cx.kernel("ops.geometry", "resize_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::ReduceHorizontal<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("inv_hscale", self.shrink as f32)
                .param("inv_vscale", 1.0f32),
        );
        cx.kernel("ops.geometry", "resize_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::ReduceVertical<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("inv_hscale", 1.0f32)
                .param("inv_vscale", self.shrink as f32),
        );
        cx.kernel("ops.geometry", "resize_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Embed<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let in_spec = &*self.input.spec;
        let bg = self.background.unwrap_or([0.0, 0.0, 0.0]);
        let extend_mode = self.extend.map(|e| e.into_vips()).unwrap_or(0);
        let (out_x, out_y) = match cx.wu() {
            chromors_core::work_unit::WorkUnit::Region(r) => (r.x, r.y),
            _ => (0, 0),
        };
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("out_x", out_x)
                .param("out_y", out_y)
                .param("ox", self.x)
                .param("oy", self.y)
                .param("in_w", in_spec.width as u32)
                .param("in_h", in_spec.height as u32)
                .param("extend_mode", extend_mode as u32)
                .param("bg_r", bg[0] as f32)
                .param("bg_g", bg[1] as f32)
                .param("bg_b", bg[2] as f32),
        );
        cx.kernel("ops.geometry", "embed_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Rot45<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let in_spec = &*self.input.spec;
        let out_spec = self.output_spec();
        let angle_deg = match self.angle {
            Angle45::D45 => 45.0,
            Angle45::D135 => 135.0,
            Angle45::D225 => 225.0,
            Angle45::D315 => 315.0,
            Angle45::D0 => 0.0,
            Angle45::D90 => 90.0,
            Angle45::D180 => 180.0,
            Angle45::D270 => 270.0,
        };
        let th = angle_deg * std::f64::consts::PI / 180.0;
        let cos_th = th.cos() as f32;
        let sin_th = th.sin() as f32;
        let c_in_x = in_spec.width as f32 / 2.0;
        let c_in_y = in_spec.height as f32 / 2.0;
        let c_out_x = out_spec.width as f32 / 2.0;
        let c_out_y = out_spec.height as f32 / 2.0;

        cx.param_block(
            crate::view::ParamBlock::new()
                .param("inv_r00", cos_th)
                .param("inv_r01", sin_th)
                .param("inv_r10", -sin_th)
                .param("inv_r11", cos_th)
                .param("cx", c_in_x)
                .param("cy", c_in_y)
                .param("ox", c_out_x)
                .param("oy", c_out_y)
                .param("bg_r", 0.0f32)
                .param("bg_g", 0.0f32)
                .param("bg_b", 0.0f32)
                .param("in_w", in_spec.width as u32)
                .param("in_h", in_spec.height as u32),
        );
        cx.kernel("ops.geometry", "rotate_kernel");
        cx.output(out_spec.output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Rotate<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let in_spec = &*self.input.spec;
        let out_spec = self.output_spec();
        let th = self.angle * std::f64::consts::PI / 180.0;
        let cos_th = th.cos() as f32;
        let sin_th = th.sin() as f32;
        let c_in_x = self.offset_input_x.unwrap_or(in_spec.width as f64 / 2.0) as f32;
        let c_in_y = self.offset_input_y.unwrap_or(in_spec.height as f64 / 2.0) as f32;
        let c_out_x = self.offset_output_x.unwrap_or(out_spec.width as f64 / 2.0) as f32;
        let c_out_y = self.offset_output_y.unwrap_or(out_spec.height as f64 / 2.0) as f32;
        let bg = self.background.unwrap_or([0.0, 0.0, 0.0]);

        cx.param_block(
            crate::view::ParamBlock::new()
                .param("inv_r00", cos_th)
                .param("inv_r01", sin_th)
                .param("inv_r10", -sin_th)
                .param("inv_r11", cos_th)
                .param("cx", c_in_x)
                .param("cy", c_in_y)
                .param("ox", c_out_x)
                .param("oy", c_out_y)
                .param("bg_r", bg[0] as f32)
                .param("bg_g", bg[1] as f32)
                .param("bg_b", bg[2] as f32)
                .param("in_w", in_spec.width as u32)
                .param("in_h", in_spec.height as u32),
        );
        cx.kernel("ops.geometry", "rotate_kernel");
        cx.output(out_spec.output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Gravity<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let in_spec = &*self.input.spec;
        let old_w = in_spec.width;
        let old_h = in_spec.height;
        let new_w = self.width;
        let new_h = self.height;
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
        let bg = self.background.unwrap_or([0.0, 0.0, 0.0]);
        let extend_mode = self.extend.map(|e| e.into_vips()).unwrap_or(0);
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("ox", ox)
                .param("oy", oy)
                .param("in_w", in_spec.width as u32)
                .param("in_h", in_spec.height as u32)
                .param("extend_mode", extend_mode as u32)
                .param("bg_r", bg[0] as f32)
                .param("bg_g", bg[1] as f32)
                .param("bg_b", bg[2] as f32),
        );
        cx.kernel("ops.geometry", "embed_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Thumbnail<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let in_spec = &*self.input.spec;
        let out_spec = self.output_spec();
        let inv_hscale = in_spec.width as f32 / out_spec.width as f32;
        let inv_vscale = in_spec.height as f32 / out_spec.height as f32;
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("inv_hscale", inv_hscale)
                .param("inv_vscale", inv_vscale),
        );
        cx.kernel("ops.geometry", "resize_kernel");
        cx.output(out_spec.output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Shrink<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("h_factor", self.horizontal.ceil() as u32)
                .param("v_factor", self.vertical.ceil() as u32),
        );
        cx.kernel("ops.geometry", "shrink_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::ShrinkHorizontal<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("h_factor", self.shrink as u32)
                .param("v_factor", 1u32),
        );
        cx.kernel("ops.geometry", "shrink_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::ShrinkVertical<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        cx.param_block(
            crate::view::ParamBlock::new()
                .param("h_factor", 1u32)
                .param("v_factor", self.shrink as u32),
        );
        cx.kernel("ops.geometry", "shrink_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Smartcrop<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        use chromors_core::operation::geometry::Interesting;
        use chromors_core::data::image::{Image2D, RamImageTarget};
        use chromors_core::io::Target;
        use chromors_core::work_unit::Lod;

        let spec = &*self.input.spec;
        let (in_w, in_h) = (spec.width as i32, spec.height as i32);
        let (out_w, out_h) = (self.width, self.height);

        // Clamp crop size to input dimensions.
        let out_w = out_w.min(in_w);
        let out_h = out_h.min(in_h);

        let (left, top) = match self.interesting {
            None | Some(Interesting::None) | Some(Interesting::Centre) | Some(Interesting::All) => {
                ((in_w - out_w) / 2, (in_h - out_h) / 2)
            }
            Some(Interesting::Low) => (0, 0),
            Some(Interesting::High) => (in_w - out_w, in_h - out_h),
            Some(_) => {
                // Attention / Entropy: gradient-proxy scoring on CPU.
                let img = Image2D::<crate::GpuBackend> {
                    root: self.input.src.clone(),
                    ctx: cx.ctx().clone(),
                    spec: self.input.spec.clone(),
                };
                let full = Region::full((in_w, in_h), Lod(0));
                match img.pull(&RamImageTarget, full) {
                    Ok(bytes) => {
                        let bands = spec.layout.channel_count().max(1) as usize;
                        let stride = in_w as usize * bands;
                        let w = in_w as usize;
                        let h = in_h as usize;
                        // Per-column gradient sum.
                        let mut col_scores = vec![0u64; w];
                        // Per-row gradient sum.
                        let mut row_scores = vec![0u64; h];
                        for y in 0..h {
                            for x in 0..w {
                                let idx = y * stride + x * bands;
                                // Horizontal gradient (right neighbor, clamped).
                                if x + 1 < w {
                                    for b in 0..bands.min(3) {
                                        let d = (bytes[idx + b] as i32 - bytes[idx + bands + b] as i32).unsigned_abs() as u64;
                                        col_scores[x] += d;
                                        col_scores[x + 1] += d;
                                    }
                                }
                                // Vertical gradient (bottom neighbor, clamped).
                                if y + 1 < h {
                                    for b in 0..bands.min(3) {
                                        let d = (bytes[idx + b] as i32 - bytes[idx + stride + b] as i32).unsigned_abs() as u64;
                                        row_scores[y] += d;
                                        row_scores[y + 1] += d;
                                    }
                                }
                            }
                        }
                        let best_col = smartcrop_best_window(&col_scores, out_w as usize);
                        let best_row = smartcrop_best_window(&row_scores, out_h as usize);
                        (best_col as i32, best_row as i32)
                    }
                    Err(e) => {
                        cx.fail(e);
                        return;
                    }
                }
            }
        };

        let left = left.clamp(0, in_w - out_w);
        let top = top.clamp(0, in_h - out_h);

        cx.adapt(remap_adapter(
            RemapKind::Translate,
            RemapParams { tx: left, ty: top, ..Default::default() },
        ));
        cx.output(self.output_spec().output(cx.wu()));
    }
}

fn smartcrop_best_window(scores: &[u64], window: usize) -> usize {
    if window == 0 || window >= scores.len() { return 0; }
    let mut sum: u64 = scores[..window].iter().sum();
    let mut best = sum;
    let mut best_off = 0;
    for i in 1..=(scores.len() - window) {
        sum += scores[i + window - 1];
        sum -= scores[i - 1];
        if sum > best { best = sum; best_off = i; }
    }
    best_off
}

impl chromors_core::operation::Lower<crate::GpuBackend> for chromors_core::operation::geometry::Grid<crate::GpuBackend> {
    fn lower(&self, cx: &mut GpuBuilder) {
        let in_spec = &*self.input.spec;
        let (out_x, out_y) = match cx.wu() {
            chromors_core::work_unit::WorkUnit::Region(r) => (r.x, r.y),
            _ => (0, 0),
        };
        cx.param_block(
            ParamBlock::new()
                .param("out_x", out_x)
                .param("out_y", out_y)
                .param("in_w", in_spec.width as u32)
                .param("tile_height", self.tile_height as u32)
                .param("across", self.across as u32),
        );
        cx.kernel("ops.geometry", "grid_kernel");
        cx.output(self.output_spec().output(cx.wu()));
    }
}

