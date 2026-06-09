use crate::backend::gpu::gpu_data::{HistogramData, ImageData};
use crate::backend::gpu::graph::{Graph, NodeId};
use crate::backend::gpu::op::GpuOperation;
use crate::backend::gpu::op::MaterializePlan;
use crate::backend::gpu::op::emit_unary;
use crate::backend::gpu::param::Param;
use crate::backend::gpu::value::ValueKind;
use crate::backend::gpu::work_unit::{Atomic, Region, WorkUnit};
use crate::geometry::Rect;
use std::sync::Arc;

use super::Direction;
use crate::backend::gpu::op::OutputSpec;
use crate::backend::vips::IntoVipsEnum;
use crate::backend::vips::gobject::Runner;
use crate::backend::vips::gobject::VipsGObject;
use crate::backend::vips::operation::VipsOperation;
use crate::data::histogram::HistogramResult;
use crate::error::Error;
use crate::libvips_ffi as ffi;

/// How `hist_find_indexed` combines pixels falling in the same bin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CombineMode {
    Max,
    Sum,
    Min,
}
impl IntoVipsEnum for CombineMode {
    fn into_vips(self) -> i32 {
        self as i32
    }
}

/// Integer bounding box, output of operations like `find_trim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bounds {
    pub left: i32,
    pub top: i32,
    pub width: i32,
    pub height: i32,
}

impl Runner for Bounds {
    fn run(mut op: VipsGObject) -> Result<Self, Error> {
        op.build()?;
        let bounds = unsafe {
            let bounds = Bounds {
                left: op.output_int("left"),
                top: op.output_int("top"),
                width: op.output_int("width"),
                height: op.output_int("height"),
            };
            ffi::vips_object_unref_outputs(op.ptr as *mut ffi::VipsObject);
            bounds
        };
        Ok(bounds)
    }
}

/// A pair of single-band result images (e.g. `profile`/`project` → columns, rows).
pub struct ImagePair {
    pub columns: crate::data::image::Image<crate::backend::vips::VipsBackend>,
    pub rows: crate::data::image::Image<crate::backend::vips::VipsBackend>,
}
impl Runner for ImagePair {
    fn run(mut op: VipsGObject) -> Result<Self, Error> {
        op.build()?;
        unsafe {
            let columns = op.output_image("columns")?;
            let rows = op.output_image("rows")?;
            ffi::vips_object_unref_outputs(op.ptr as *mut ffi::VipsObject);
            Ok(ImagePair { columns, rows })
        }
    }
}

/// Output of `labelregions`: the label mask plus the region count.
pub struct Labels {
    pub mask: crate::data::image::Image<crate::backend::vips::VipsBackend>,
    pub segments: i32,
}
impl Runner for Labels {
    fn run(mut op: VipsGObject) -> Result<Self, Error> {
        op.build()?;
        unsafe {
            let mask = op.output_image("mask")?;
            let segments = op.output_int("segments");
            ffi::vips_object_unref_outputs(op.ptr as *mut ffi::VipsObject);
            Ok(Labels { mask, segments })
        }
    }
}

/// Output of `fill_nearest`: filled image plus the distance map.
pub struct Filled {
    pub value: crate::data::image::Image<crate::backend::vips::VipsBackend>,
    pub distance: crate::data::image::Image<crate::backend::vips::VipsBackend>,
}
impl Runner for Filled {
    fn run(mut op: VipsGObject) -> Result<Self, Error> {
        op.build()?;
        unsafe {
            let value = op.output_image("out")?;
            let distance = op.output_image("distance")?;
            ffi::vips_object_unref_outputs(op.ptr as *mut ffi::VipsObject);
            Ok(Filled { value, distance })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Threshold(pub i32);
impl Runner for Threshold {
    fn run(mut op: VipsGObject) -> Result<Self, Error> {
        op.build()?;
        let v = unsafe {
            let v = op.output_int("threshold");
            ffi::vips_object_unref_outputs(op.ptr as *mut ffi::VipsObject);
            v
        };
        Ok(Threshold(v))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineCount(pub f64);
impl Runner for LineCount {
    fn run(mut op: VipsGObject) -> Result<Self, Error> {
        op.build()?;
        let v = unsafe {
            let v = op.output_double("nolines");
            ffi::vips_object_unref_outputs(op.ptr as *mut ffi::VipsObject);
            v
        };
        Ok(LineCount(v))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Monotonic(pub bool);
impl Runner for Monotonic {
    fn run(mut op: VipsGObject) -> Result<Self, Error> {
        op.build()?;
        let v = unsafe {
            let v = op.output_bool("monotonic");
            ffi::vips_object_unref_outputs(op.ptr as *mut ffi::VipsObject);
            v
        };
        Ok(Monotonic(v))
    }
}

/// Output of `getpoint`: the pixel's band values.
#[derive(Debug, Clone, PartialEq)]
pub struct PixelValues(pub Vec<f64>);
impl Runner for PixelValues {
    fn run(mut op: VipsGObject) -> Result<Self, Error> {
        op.build()?;
        let v = unsafe {
            let v = op.output_array_double("out_array");
            ffi::vips_object_unref_outputs(op.ptr as *mut ffi::VipsObject);
            v
        };
        Ok(PixelValues(v))
    }
}

pub struct AverageOperation;
impl VipsOperation for AverageOperation {
    type Output = f64;
    fn name() -> &'static [u8] {
        b"avg\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct DeviateOperation;
impl VipsOperation for DeviateOperation {
    type Output = f64;
    fn name() -> &'static [u8] {
        b"deviate\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct HistogramEntropyOperation;
impl VipsOperation for HistogramEntropyOperation {
    type Output = f64;
    fn name() -> &'static [u8] {
        b"hist_entropy\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct MinimumOperation {
    pub size: Option<i32>,
    pub x: Option<i32>,
    pub y: Option<i32>,
}
impl VipsOperation for MinimumOperation {
    type Output = f64;
    fn name() -> &'static [u8] {
        b"min\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.size {
            op.set_int("size", v);
        }
        if let Some(v) = self.x {
            op.set_int("x", v);
        }
        if let Some(v) = self.y {
            op.set_int("y", v);
        }
    }
}

pub struct MaximumOperation {
    pub size: Option<i32>,
    pub x: Option<i32>,
    pub y: Option<i32>,
}
impl VipsOperation for MaximumOperation {
    type Output = f64;
    fn name() -> &'static [u8] {
        b"max\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.size {
            op.set_int("size", v);
        }
        if let Some(v) = self.x {
            op.set_int("x", v);
        }
        if let Some(v) = self.y {
            op.set_int("y", v);
        }
    }
}

pub struct FindTrimOperation {
    pub background: Option<[f64; 3]>,
    pub threshold: Option<f64>,
    pub line_art: Option<bool>,
}
impl VipsOperation for FindTrimOperation {
    type Output = Bounds;
    fn name() -> &'static [u8] {
        b"find_trim\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = &self.background {
            op.set_array_double("background", v);
        }
        if let Some(v) = self.threshold {
            op.set_double("threshold", v);
        }
        if let Some(v) = self.line_art {
            op.set_bool("line_art", v);
        }
    }
}

pub struct HistogramFindOperation {
    pub band: Option<i32>,
}
impl VipsOperation for HistogramFindOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hist_find\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.band {
            op.set_int("band", v);
        }
    }
}

pub struct HistogramEqualizeOperation {
    pub band: Option<i32>,
}
impl VipsOperation for HistogramEqualizeOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hist_equal\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.band {
            op.set_int("band", v);
        }
    }
}

pub struct HistogramCumulativeOperation;
impl VipsOperation for HistogramCumulativeOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hist_cum\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct HistogramNormalizeOperation;
impl VipsOperation for HistogramNormalizeOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hist_norm\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct HistogramPlotOperation;
impl VipsOperation for HistogramPlotOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hist_plot\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct HistFindIndexedOperation<'a> {
    pub index: &'a crate::data::image::Image<crate::backend::vips::VipsBackend>,
    pub combine: Option<CombineMode>,
}
impl VipsOperation for HistFindIndexedOperation<'_> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hist_find_indexed\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("index", self.index.vips_ptr());
        if let Some(v) = self.combine {
            op.set_int("combine", v.into_vips());
        }
    }
}

pub struct HistFindNdimOperation {
    pub bins: Option<i32>,
}
impl VipsOperation for HistFindNdimOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hist_find_ndim\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        if let Some(v) = self.bins {
            op.set_int("bins", v);
        }
    }
}

pub struct HistLocalOperation {
    pub width: i32,
    pub height: i32,
    pub max_slope: Option<i32>,
}
impl VipsOperation for HistLocalOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hist_local\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(v) = self.max_slope {
            op.set_int("max_slope", v);
        }
    }
}

pub struct HistMatchOperation<'a> {
    pub reference: &'a crate::data::image::Image<crate::backend::vips::VipsBackend>,
}
impl VipsOperation for HistMatchOperation<'_> {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"hist_match\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_image("ref", self.reference.vips_ptr());
    }
}

pub struct StdifOperation {
    pub width: i32,
    pub height: i32,
    pub new_deviation: Option<f64>,
    pub deviation_weight: Option<f64>,
    pub new_mean: Option<f64>,
    pub mean_weight: Option<f64>,
}
impl VipsOperation for StdifOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"stdif\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("width", self.width);
        op.set_int("height", self.height);
        if let Some(v) = self.new_deviation {
            op.set_double("s0", v);
        }
        if let Some(v) = self.deviation_weight {
            op.set_double("b", v);
        }
        if let Some(v) = self.new_mean {
            op.set_double("m0", v);
        }
        if let Some(v) = self.mean_weight {
            op.set_double("a", v);
        }
    }
}

pub struct MeasureOperation {
    pub horizontal: i32,
    pub vertical: i32,
    pub left: Option<i32>,
    pub top: Option<i32>,
    pub width: Option<i32>,
    pub height: Option<i32>,
}
impl VipsOperation for MeasureOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"measure\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("h", self.horizontal);
        op.set_int("v", self.vertical);
        if let Some(v) = self.left {
            op.set_int("left", v);
        }
        if let Some(v) = self.top {
            op.set_int("top", v);
        }
        if let Some(v) = self.width {
            op.set_int("width", v);
        }
        if let Some(v) = self.height {
            op.set_int("height", v);
        }
    }
}

pub struct StatsOperation;
impl VipsOperation for StatsOperation {
    type Output = crate::data::image::Image<crate::backend::vips::VipsBackend>;
    fn name() -> &'static [u8] {
        b"stats\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct ProfileOperation;
impl VipsOperation for ProfileOperation {
    type Output = ImagePair;
    fn name() -> &'static [u8] {
        b"profile\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct ProjectOperation;
impl VipsOperation for ProjectOperation {
    type Output = ImagePair;
    fn name() -> &'static [u8] {
        b"project\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct LabelregionsOperation;
impl VipsOperation for LabelregionsOperation {
    type Output = Labels;
    fn name() -> &'static [u8] {
        b"labelregions\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct FillNearestOperation;
impl VipsOperation for FillNearestOperation {
    type Output = Filled;
    fn name() -> &'static [u8] {
        b"fill_nearest\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct PercentOperation {
    pub percent: f64,
}
impl VipsOperation for PercentOperation {
    type Output = Threshold;
    fn name() -> &'static [u8] {
        b"percent\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_double("percent", self.percent);
    }
}

pub struct CountlinesOperation {
    pub direction: Direction,
}
impl VipsOperation for CountlinesOperation {
    type Output = LineCount;
    fn name() -> &'static [u8] {
        b"countlines\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("direction", self.direction.into_vips());
    }
}

pub struct HistIsmonotonicOperation;
impl VipsOperation for HistIsmonotonicOperation {
    type Output = Monotonic;
    fn name() -> &'static [u8] {
        b"hist_ismonotonic\0"
    }
    fn build(&self, o: &mut VipsGObject, i: *mut ffi::VipsImage) {
        o.set_image("in", i);
    }
}

pub struct GetpointOperation {
    pub x: i32,
    pub y: i32,
    pub unpack_complex: Option<bool>,
}
impl VipsOperation for GetpointOperation {
    type Output = PixelValues;
    fn name() -> &'static [u8] {
        b"getpoint\0"
    }
    fn build(&self, op: &mut VipsGObject, image: *mut ffi::VipsImage) {
        op.set_image("in", image);
        op.set_int("x", self.x);
        op.set_int("y", self.y);
        if let Some(v) = self.unpack_complex {
            op.set_bool("unpack_complex", v);
        }
    }
}

// ── HistogramOp ───────────────────────────────────────────────────────────────

/// Per-pixel histogram accumulation into `bins` atomic uint counters.
/// `channel`: 0=R, 1=G, 2=B, 3=A, 4=luminance (BT.709 linear).
#[derive(Clone, Debug)]
pub struct HistogramOp {
    pub bins: u32,
    pub channel: u32,
}

pub struct HistogramHandle {
    pub node: crate::backend::gpu::GraphNodeHandle,
    pub width: u32,
    pub height: u32,
}

impl HistogramHandle {
    pub fn materialize(&self) -> Result<HistogramResult, crate::error::Error> {
        self.materialize_at_lod(crate::backend::gpu::Lod::FULL)
    }

    pub fn materialize_at_lod(
        &self,
        lod: crate::backend::gpu::Lod,
    ) -> Result<HistogramResult, crate::error::Error> {
        let region = crate::backend::gpu::region::GpuRegion::new(
            self.node.graph.clone(),
            self.node.ctx.cache.clone(),
            self.node.root_id,
            self.node.ctx.clone(),
            lod,
        );
        let scale = 1.0 / lod.scale_factor();
        let w = (self.width as f64 * scale).ceil() as i32;
        let h = (self.height as f64 * scale).ceil() as i32;
        region.prepare(Rect::new(0, 0, w, h));
        let mat = region
            .materialize()
            .map_err(|e| crate::Error::Gpu(format!("{:?}", e)))?;
        match &*mat {
            crate::backend::gpu::GraphValue::Raw { bytes, .. } => {
                Ok(HistogramResult::from_bytes(bytes))
            }
            _ => Err(crate::Error::Gpu("expected raw bytes for histogram".into())),
        }
    }
}

impl HistogramOp {
    /// Compute a histogram from `image`. Returns a `HistogramHandle` for lazy
    /// materialization.
    pub fn apply(
        &self,
        image: &crate::data::image::Image<crate::backend::gpu::GpuBackend>,
    ) -> Result<HistogramHandle, crate::error::Error> {
        let handle = &image.handle;
        let self_arc: Arc<dyn GpuOperation> = Arc::new(self.clone());
        let node_id = {
            let mut graph = handle.node.graph.lock().unwrap();
            self.emit(handle.node.root_id, &mut graph, self_arc)
        };
        Ok(HistogramHandle {
            node: crate::backend::gpu::GraphNodeHandle {
                graph: handle.node.graph.clone(),
                root_id: node_id,
                ctx: handle.node.ctx.clone(),
            },
            width: handle.width,
            height: handle.height,
        })
    }
}

impl GpuOperation for HistogramOp {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_unary(
            graph,
            input,
            self_arc,
            "ops.histogram",
            "histogram_kernel",
            vec![Param::U32(self.channel)],
            ValueKind::Histogram { bins: self.bins },
        )
    }

    fn output_spec(&self, _w: u32, _h: u32) -> OutputSpec {
        OutputSpec::Histogram { bins: self.bins }
    }

    fn input_demands(
        &self,
        out: &WorkUnit,
        w: u32,
        h: u32,
        lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, WorkUnit)> {
        use crate::backend::gpu::work_unit::AnyWorkUnit;
        MaterializePlan::<HistogramData>::plan(
            self,
            Atomic::from_work_unit(out).unwrap_or(Atomic),
            w,
            h,
            lod,
        )
    }

    fn output_encoder(&self) -> crate::backend::gpu::op::Encoder {
        // Raw u32 atomic bin counters — no color encode, write straight through.
        crate::backend::gpu::op::Encoder::Passthrough
    }

    fn dispatch_grid(&self) -> crate::backend::gpu::op::DispatchGrid {
        // Scans the input image to fold pixels into bins — the thread grid
        // must cover the source pixels, not the `bins`-shaped output.
        crate::backend::gpu::op::DispatchGrid::Input(0)
    }
}

impl MaterializePlan<HistogramData> for HistogramOp {
    fn plan(&self, _request: Atomic, w: u32, h: u32, lod: crate::backend::gpu::Lod) -> Vec<(usize, WorkUnit)> {
        let s = lod.scale_factor();
        let full = Rect::new(0, 0, (w as f64 / s).ceil() as i32, (h as f64 / s).ceil() as i32);
        // Histogram scans its full image input regardless of output demand.
        // The input's natural WU is Region — we request the full bounds.
        vec![(0, WorkUnit::Region(full))]
    }
}

// ── VectorscopeOp ─────────────────────────────────────────────────────────────

/// 2-D Cb/Cr density grid computed entirely on GPU.
///
/// Stores results in a flat `grid_size × grid_size` atomic-uint buffer
/// (`ValueKind::Histogram { bins: grid_size² }`).  Apply to a *display*
/// image (sRGB Rgba8) for standard BT.601 Cb/Cr positions.
#[derive(Clone, Debug)]
pub struct VectorscopeOp {
    pub grid_size: u32,
}

pub struct VectorscopeHandle {
    pub node: crate::backend::gpu::GraphNodeHandle,
    pub width: u32,
    pub height: u32,
    pub grid_size: u32,
}

impl VectorscopeHandle {
    pub fn materialize_at_lod(
        &self,
        lod: crate::backend::gpu::Lod,
    ) -> Result<Vec<u32>, crate::error::Error> {
        let region = crate::backend::gpu::region::GpuRegion::new(
            self.node.graph.clone(),
            self.node.ctx.cache.clone(),
            self.node.root_id,
            self.node.ctx.clone(),
            lod,
        );
        let scale = 1.0 / lod.scale_factor();
        let w = (self.width as f64 * scale).ceil() as i32;
        let h = (self.height as f64 * scale).ceil() as i32;
        region.prepare(Rect::new(0, 0, w, h));
        let mat = region
            .materialize()
            .map_err(|e| crate::Error::Gpu(format!("{:?}", e)))?;
        match &*mat {
            crate::backend::gpu::GraphValue::Raw { bytes, .. } => Ok(bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            _ => Err(crate::Error::Gpu(
                "expected raw bytes for vectorscope".into(),
            )),
        }
    }
}

impl VectorscopeOp {
    pub fn apply(
        &self,
        image: &crate::data::image::Image<crate::backend::gpu::GpuBackend>,
    ) -> Result<VectorscopeHandle, crate::error::Error> {
        let handle = &image.handle;
        let self_arc: Arc<dyn GpuOperation> = Arc::new(self.clone());
        let node_id = {
            let mut graph = handle.node.graph.lock().unwrap();
            self.emit(handle.node.root_id, &mut graph, self_arc)
        };
        Ok(VectorscopeHandle {
            node: crate::backend::gpu::GraphNodeHandle {
                graph: handle.node.graph.clone(),
                root_id: node_id,
                ctx: handle.node.ctx.clone(),
            },
            width: handle.width,
            height: handle.height,
            grid_size: self.grid_size,
        })
    }
}

impl GpuOperation for VectorscopeOp {
    fn emit(&self, input: NodeId, graph: &mut Graph, self_arc: Arc<dyn GpuOperation>) -> NodeId {
        emit_unary(
            graph,
            input,
            self_arc,
            "ops.vectorscope",
            "vectorscope_kernel",
            vec![Param::U32(self.grid_size)],
            ValueKind::Histogram {
                bins: self.grid_size * self.grid_size,
            },
        )
    }

    fn output_spec(&self, _w: u32, _h: u32) -> OutputSpec {
        OutputSpec::Histogram {
            bins: self.grid_size * self.grid_size,
        }
    }

    fn input_demands(
        &self,
        out: &WorkUnit,
        w: u32,
        h: u32,
        lod: crate::backend::gpu::Lod,
    ) -> Vec<(usize, WorkUnit)> {
        use crate::backend::gpu::work_unit::AnyWorkUnit;
        MaterializePlan::<HistogramData>::plan(
            self,
            Atomic::from_work_unit(out).unwrap_or(Atomic),
            w,
            h,
            lod,
        )
    }

    fn output_encoder(&self) -> crate::backend::gpu::op::Encoder {
        // Raw u32 grid bins — no color encode, write straight through.
        crate::backend::gpu::op::Encoder::Passthrough
    }

    fn dispatch_grid(&self) -> crate::backend::gpu::op::DispatchGrid {
        // Scans the input image to plot Cb/Cr density — same as histogram.
        crate::backend::gpu::op::DispatchGrid::Input(0)
    }
}

impl MaterializePlan<HistogramData> for VectorscopeOp {
    fn plan(&self, _request: Atomic, w: u32, h: u32, lod: crate::backend::gpu::Lod) -> Vec<(usize, WorkUnit)> {
        let s = lod.scale_factor();
        let full = Rect::new(0, 0, (w as f64 / s).ceil() as i32, (h as f64 / s).ceil() as i32);
        // Vectorscope scans the entire input image to plot Cb/Cr density.
        vec![(0, WorkUnit::Region(full))]
    }
}
