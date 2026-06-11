use askama::Template;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use crate::color::space::ColorSpace;
use crate::pixel::PixelFormat;

use super::datatype::DataType;
use super::graph::NodeEval;
use super::graph::{Graph, NodeId};
use super::materialize::MaterializePlan;
use super::op::{DispatchGrid, InputEncoder, OutputDecoder, working_image_type};
use super::param::{GpuPixelEncoding, Param};
use super::source::AnyGpuSource;
use super::value::WriteMode;

// ── Public output ─────────────────────────────────────────────────────────────

pub struct EmittedIr {
    pub text: String,
    pub source_count: usize,
    pub target_count: usize,
    pub temp_buffer_sizes: Vec<u64>,
    pub params_bytes: Vec<u8>,
    pub entry_points: Vec<(String, u32, u32)>,
    pub target_output_kinds: Vec<Arc<dyn DataType>>,
}

impl EmittedIr {
    /// A stable 64-bit hash of the shader IR text.
    ///
    /// Used as the pipeline cache key.  Structurally identical graphs produce
    /// identical IR text (all names are positional) and therefore the same key.
    pub fn cache_key(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        0xdead_beef_u64.hash(&mut hasher);
        self.text.hash(&mut hasher);
        hasher.finish()
    }
}

impl MaterializePlan {
    /// Emit IR with LOD-scaled params baked in.
    pub fn emit_ir_with_layout(
        &self,
        graph: &Graph,
        wg_dim: u32,
        lod: super::Lod,
    ) -> (EmittedIr, LayoutPlan) {
        let (layout, color) = LayoutPlan::build(graph, self, lod);
        let emitter = SlangEmitter::new(&layout, &color, graph);
        let ir = emitter.emit(wg_dim);
        (ir, layout)
    }
}

// ── Layout plan — pure data, inspectable ─────────────────────────────────────

/// All buffer-allocation decisions for one materialization pass.
///
/// Computed before any code is written — `{:#?}` it to see what the shader will contain.
/// All Slang names are POSITIONAL (binding indices, not NodeIds) so structurally identical
/// graphs produce identical shader text and hit the pipeline cache.
///
/// Color/format concerns live in [`ColorPipeline`], not here.
#[derive(Debug)]
pub struct LayoutPlan {
    pub order: Vec<NodeId>,
    pub needed: BTreeSet<NodeId>,
    pub src_set: BTreeSet<NodeId>,
    pub imports: BTreeSet<&'static str>,
    pub sources: Vec<SourceSlot>,
    pub temps: BTreeMap<NodeId, TempSlot>,
    pub targets: Vec<TargetSlot>,
    pub target_map: HashMap<NodeId, usize>,
    pub params: ParamLayout,
    /// node_id → index in `sources` (= source.binding). Used for stable Slang naming.
    pub source_pos: HashMap<NodeId, usize>,
    /// node_id → index in `targets` (= target.binding). Used for stable Slang naming.
    pub target_pos: HashMap<NodeId, usize>,
}

/// Color-space and pixel-format decisions for one materialization pass.
///
/// Separated from [`LayoutPlan`] because color/format concerns are image-domain
/// knowledge that the generic buffer-allocation layer should not carry.
///
/// Both `dst_encoding` and `dst_format` are `None` for non-image passes
/// (histograms, masks, FFT) that never execute the `from_working` encode step.
#[derive(Debug)]
pub struct ColorPipeline {
    /// Per-source color encoding (one per [`SourceSlot`]).
    pub src_encodings: Vec<GpuPixelEncoding>,
    /// Output color encoding — `None` for non-image output kinds.
    pub dst_encoding: Option<GpuPixelEncoding>,
    /// Output pixel format — `None` for non-image output kinds.
    pub dst_format: Option<PixelFormat>,
}

#[derive(Debug)]
pub struct SourceSlot {
    pub node_id: NodeId,
    pub binding: u32, // group 0, positional index
    pub format: PixelFormat,
    pub buf_stride: u32,
    pub view_x: u32,
    pub view_y: u32,
    pub view_width: u32,
    pub view_height: u32,
}

#[derive(Debug)]
pub struct TempSlot {
    pub node_id: NodeId,
    pub binding: u32, // group 1, slot index (0-based, reused)
    pub size_bytes: u64,
    pub dims: (u32, u32),
}

#[derive(Debug)]
pub struct TargetSlot {
    pub node_id: NodeId,
    pub binding: u32, // group 2, positional index
    pub output: Arc<dyn DataType>,
    pub dims: (u32, u32),
}

#[derive(Debug)]
pub struct ParamLayout {
    /// (field_name, slang_type) in the order they appear in ChainParams.
    pub fields: Vec<(String, &'static str)>,
    /// Raw bytes in field order — written to the params GPU buffer.
    pub bytes: Vec<u8>,
    /// First param field index (into `fields`) for each op node.
    pub node_base: HashMap<NodeId, u32>,
}

// ── GpuPixelEncoding does not derive Debug — provide a minimal impl ──────────────

impl std::fmt::Debug for GpuPixelEncoding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GpuPixelEncoding {{ transfer: {}, alpha: {}, model: {}, channels: {} }}",
            self.transfer, self.alpha, self.model, self.channels
        )
    }
}

impl GpuPixelEncoding {
    pub fn from_color_space(cs: ColorSpace) -> Self {
        let meta = crate::pixel::PixelMeta::new(
            PixelFormat::RgbaF32,
            cs,
            crate::pixel::AlphaPolicy::Straight,
        );
        Self::from_meta(&meta, false)
    }
}

// ── LayoutPlan construction ───────────────────────────────────────────────────

impl LayoutPlan {
    pub fn build(graph: &Graph, plan: &MaterializePlan, lod: super::Lod) -> (Self, ColorPipeline) {
        LayoutBuilder::new(graph, plan, lod).build()
    }
}

struct LayoutBuilder<'a> {
    graph: &'a Graph,
    plan: &'a MaterializePlan,
    lod: super::Lod,
}

impl<'a> LayoutBuilder<'a> {
    fn new(graph: &'a Graph, plan: &'a MaterializePlan, lod: super::Lod) -> Self {
        Self { graph, plan, lod }
    }

    fn build(&self) -> (LayoutPlan, ColorPipeline) {
        let order = self.graph.topo_order();
        let src_set: BTreeSet<NodeId> = self.plan.sources.iter().copied().collect();
        let needed = self.compute_needed_nodes();
        let imports = self.collect_imports(&order, &needed);
        let sources = self.alloc_sources();
        let liveness = self.compute_liveness(&order, &needed);
        let temps = self.alloc_temps(&order, &needed, &src_set, &liveness);
        let targets = self.alloc_targets();

        let target_map = targets
            .iter()
            .enumerate()
            .map(|(i, t)| (t.node_id, i))
            .collect();

        let source_pos: HashMap<NodeId, usize> = sources
            .iter()
            .map(|s| (s.node_id, s.binding as usize))
            .collect();

        let target_pos: HashMap<NodeId, usize> = targets
            .iter()
            .map(|t| (t.node_id, t.binding as usize))
            .collect();

        let params = self.build_params(&order, &needed, &sources, &temps, &targets, self.lod);

        // ── Color pipeline (image-domain, separate from alloc) ─────────────────
        let src_encodings = sources
            .iter()
            .map(|src| {
                let cs = self
                    .graph
                    .get_source(src.node_id)
                    .map(|s| s.source.color_space())
                    .unwrap_or(ColorSpace::SRGB);
                // is_source=true: matrix direction is cs→hub (source→sRGB_linear).
                // The shader's to_working() applies this matrix AFTER decode_tf so
                // it must go source_linear→sRGB_linear, not the other way around.
                let meta = crate::pixel::PixelMeta::new(
                    PixelFormat::RgbaF32,
                    cs,
                    crate::pixel::AlphaPolicy::Straight,
                );
                GpuPixelEncoding::from_meta(&meta, true)
            })
            .collect();

        let has_image_output = self.plan.targets.iter().any(|t| {
            self.graph
                .get_node(t.node_id)
                .map(|n| {
                    matches!(
                        n.op.output_decoder(),
                        OutputDecoder::WorkingEncodeRegion { .. }
                    )
                })
                .unwrap_or(false)
        });

        let (dst_encoding, dst_format) = if has_image_output {
            let override_codec = order.iter().rev().find_map(|&id| {
                self.graph
                    .get_node(id)
                    .and_then(|n| match n.op.output_decoder() {
                        OutputDecoder::WorkingEncodeRegion { codec: Some(c) } => Some(c),
                        _ => None,
                    })
            });
            match override_codec {
                Some(codec) => {
                    let meta = crate::pixel::PixelMeta::new(
                        codec.format,
                        codec.color_space,
                        crate::pixel::AlphaPolicy::Straight,
                    );
                    (
                        Some(GpuPixelEncoding::from_meta(&meta, false)),
                        Some(codec.format),
                    )
                }
                None => {
                    let src_cs = self
                        .graph
                        .sources
                        .first()
                        .map(|s| s.source.color_space())
                        .unwrap_or(ColorSpace::SRGB);
                    (
                        Some(GpuPixelEncoding::from_color_space(src_cs)),
                        Some(PixelFormat::RgbaF32),
                    )
                }
            }
        } else {
            (None, None)
        };

        let color = ColorPipeline {
            src_encodings,
            dst_encoding,
            dst_format,
        };

        let layout = LayoutPlan {
            order,
            needed,
            src_set,
            imports,
            sources,
            temps,
            targets,
            target_map,
            params,
            source_pos,
            target_pos,
        };

        (layout, color)
    }

    fn compute_needed_nodes(&self) -> BTreeSet<NodeId> {
        self.plan
            .node_outputs
            .iter()
            .map(|(id, _)| *id)
            .chain(self.plan.targets.iter().map(|t| t.node_id))
            .collect()
    }

    fn collect_imports(
        &self,
        order: &[NodeId],
        needed: &BTreeSet<NodeId>,
    ) -> BTreeSet<&'static str> {
        order
            .iter()
            .filter(|id| needed.contains(id))
            .filter_map(|&id| self.graph.get_node(id))
            .map(|n| {
                let NodeEval::Kernel(k) = &n.eval;
                k.module
            })
            .collect()
    }

    fn compute_liveness(
        &self,
        order: &[NodeId],
        needed: &BTreeSet<NodeId>,
    ) -> HashMap<NodeId, usize> {
        let targets = &self.plan.targets;
        let mut last_read: HashMap<NodeId, usize> = HashMap::new();

        // Target nodes must stay alive until the end of the pass.
        for t in targets {
            last_read.insert(t.node_id, order.len());
        }

        // last_read[id] = max topo-order index at which id's output is read as input.
        // Iterating forward is sufficient: each node's consumers come after it in topo order.
        for (i, &id) in order.iter().enumerate() {
            if !needed.contains(&id) {
                continue;
            }
            if let Some(n) = self.graph.get_node(id) {
                for &inp in &n.inputs {
                    let e = last_read.entry(inp).or_insert(0);
                    if i > *e {
                        *e = i;
                    }
                }
            }
        }

        // Re-seed targets in case the forward pass overwrote them with a lower value.
        for t in targets {
            last_read.insert(t.node_id, order.len());
        }

        last_read
    }

    fn alloc_temps(
        &self,
        order: &[NodeId],
        needed: &BTreeSet<NodeId>,
        src_set: &BTreeSet<NodeId>,
        liveness: &HashMap<NodeId, usize>,
    ) -> BTreeMap<NodeId, TempSlot> {
        let mut slot_max_size: Vec<u64> = Vec::new();
        // Dims are fixed per slot: slots may only be reused by nodes with the
        // same (w, h) so that the single `temp_region_{b}` entry in ChainParams
        // remains unambiguous across all entry points that reference binding b.
        let mut slot_dims: Vec<(u32, u32)> = Vec::new();
        let mut free_slots: BTreeSet<usize> = BTreeSet::new();
        let mut active: HashMap<NodeId, usize> = HashMap::new();
        let mut result: BTreeMap<NodeId, TempSlot> = BTreeMap::new();

        for (i, &id) in order.iter().enumerate() {
            let expired: Vec<NodeId> = active
                .keys()
                .copied()
                .filter(|&nid| liveness.get(&nid).copied().unwrap_or(0) < i)
                .collect();
            for nid in expired {
                if let Some(slot) = active.remove(&nid) {
                    free_slots.insert(slot);
                }
            }

            if src_set.contains(&id) || !needed.contains(&id) {
                continue;
            }

            let kind = self
                .graph
                .get_node(id)
                .map(|n| n.datatype.clone())
                .unwrap_or_else(working_image_type);
            if !kind.needs_fused_temp() {
                continue;
            }

            let dims = self.plan_node_dims(id);
            let required = (dims.0 as u64 * dims.1 as u64 * 16).max(64);

            // Only reuse a free slot when dimensions match exactly so the
            // shared `temp_region_{b}` field in ChainParams stays unambiguous.
            let slot = if let Some(&free) = free_slots
                .iter()
                .find(|&&s| slot_dims.get(s).copied() == Some(dims))
            {
                free_slots.remove(&free);
                free
            } else {
                let s = slot_max_size.len();
                slot_max_size.push(0);
                slot_dims.push(dims);
                s
            };
            let max = &mut slot_max_size[slot];
            if required > *max {
                *max = required;
            }

            active.insert(id, slot);
            result.insert(
                id,
                TempSlot {
                    node_id: id,
                    binding: slot as u32,
                    size_bytes: required,
                    dims,
                },
            );
        }

        for slot in result.values_mut() {
            slot.size_bytes = slot_max_size[slot.binding as usize];
        }
        result
    }

    fn alloc_sources(&self) -> Vec<SourceSlot> {
        self.plan
            .sources
            .iter()
            .enumerate()
            .map(|(i, &sid)| {
                let src_node = self.graph.get_source(sid).unwrap();
                let (fetch_w, fetch_h) = self.plan_source_dims(sid);
                let (view_x, view_y) = self.plan_source_view_xy(sid);
                SourceSlot {
                    node_id: sid,
                    binding: i as u32,
                    format: src_node.source.format(),
                    buf_stride: fetch_w,
                    view_x,
                    view_y,
                    view_width: fetch_w.saturating_sub(view_x),
                    view_height: fetch_h.saturating_sub(view_y),
                }
            })
            .collect()
    }

    fn alloc_targets(&self) -> Vec<TargetSlot> {
        self.plan
            .targets
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let output = self
                    .graph
                    .get_node(t.node_id)
                    .map(|n| n.datatype.clone())
                    .unwrap_or_else(working_image_type);
                TargetSlot {
                    node_id: t.node_id,
                    binding: i as u32,
                    output,
                    dims: (t.rect.width as u32, t.rect.height as u32),
                }
            })
            .collect()
    }

    fn build_params(
        &self,
        order: &[NodeId],
        needed: &BTreeSet<NodeId>,
        sources: &[SourceSlot],
        temps: &BTreeMap<NodeId, TempSlot>,
        targets: &[TargetSlot],
        lod: super::Lod,
    ) -> ParamLayout {
        let mut fields: Vec<(String, &'static str)> = Vec::new();
        let mut bytes: Vec<u8> = Vec::new();

        macro_rules! push_region {
            ($name:expr, $stride:expr, $x:expr, $y:expr, $w:expr, $h:expr) => {{
                fields.push(($name, "BufferRegion"));
                bytes.extend_from_slice(&($stride as u32).to_le_bytes());
                bytes.extend_from_slice(&($x as u32).to_le_bytes());
                bytes.extend_from_slice(&($y as u32).to_le_bytes());
                bytes.extend_from_slice(&($w as u32).to_le_bytes());
                bytes.extend_from_slice(&($h as u32).to_le_bytes());
            }};
        }

        for src in sources {
            let i = src.binding;
            push_region!(
                format!("inputs_{i}"),
                src.buf_stride,
                src.view_x,
                src.view_y,
                src.view_width,
                src.view_height
            );
        }

        {
            let mut emitted: BTreeSet<u32> = BTreeSet::new();
            for &id in order {
                if let Some(tmp) = temps.get(&id) {
                    let b = tmp.binding;
                    if emitted.insert(b) {
                        push_region!(
                            format!("temp_region_{b}"),
                            tmp.dims.0,
                            0,
                            0,
                            tmp.dims.0,
                            tmp.dims.1
                        );
                    }
                }
            }
        }

        for tgt in targets {
            let i = tgt.binding;
            match tgt.output.write_mode() {
                WriteMode::AtomicAccumulate { count } => {
                    fields.push((format!("bin_count_{i}"), "uint"));
                    bytes.extend_from_slice(&count.to_le_bytes());
                }
                WriteMode::Positional => {
                    push_region!(
                        format!("region_target_{i}"),
                        tgt.dims.0,
                        0,
                        0,
                        tgt.dims.0,
                        tgt.dims.1
                    );
                }
            }
        }

        let mut node_base: HashMap<NodeId, u32> = HashMap::new();
        let mut ui = 0u32;
        for &id in order.iter().filter(|&&id| needed.contains(&id)) {
            if let Some(node) = self.graph.get_node(id) {
                node_base.insert(id, ui);
                let scaled = node.op.scale_params_for_lod(&node.params, lod);
                for p in &scaled {
                    match p {
                        Param::I32(v) => {
                            fields.push((format!("u{ui}"), "int"));
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                        Param::U32(v) => {
                            fields.push((format!("u{ui}"), "uint"));
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                        Param::F32(v) => {
                            fields.push((format!("u{ui}"), "float"));
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }
                        Param::Struct { .. } | Param::Region { .. } => {
                            unimplemented!(
                                "Param::Struct / Param::Region not yet supported in JIT emit"
                            )
                        }
                    }
                    ui += 1;
                }
            }
        }

        ParamLayout {
            fields,
            bytes,
            node_base,
        }
    }

    fn plan_source_dims(&self, sid: NodeId) -> (u32, u32) {
        self.plan
            .source_fetches
            .iter()
            .find(|(id, _)| *id == sid)
            .and_then(|(_, rs)| rs.first())
            .map(|(r, _)| (r.width as u32, r.height as u32))
            .unwrap_or((1, 1))
    }

    fn plan_source_view_xy(&self, sid: NodeId) -> (u32, u32) {
        let src_rect = self
            .plan
            .source_fetches
            .iter()
            .find(|(id, _)| *id == sid)
            .and_then(|(_, rs)| rs.first())
            .map(|(r, _)| *r);
        let out_rect = self.plan.targets.first().map(|t| t.rect);
        match (src_rect, out_rect) {
            (Some(s), Some(o)) => ((o.x - s.x).max(0) as u32, (o.y - s.y).max(0) as u32),
            _ => (0, 0),
        }
    }

    fn plan_node_dims(&self, id: NodeId) -> (u32, u32) {
        self.plan
            .node_outputs
            .iter()
            .find(|(nid, _)| *nid == id)
            .map(|(_, r)| (r.width as u32, r.height as u32))
            .unwrap_or((1, 1))
    }
}

// Removed loose functions, all logic is now encapsulated inside LayoutBuilder.

// ── Slang emitter — thin string builder ──────────────────────────────────────

// ── Pixel codec helpers ───────────────────────────────────────────────────────

pub trait SlangFormatExt {
    fn slang_codec(&self) -> &'static str;
    fn slang_layout(&self) -> &'static str;
}

impl SlangFormatExt for PixelFormat {
    fn slang_codec(&self) -> &'static str {
        match self {
            PixelFormat::RgbaF32 | PixelFormat::RgbF32 => "F32Codec",
            PixelFormat::Rgba16 | PixelFormat::Rgb16 | PixelFormat::Gray16 => "U16Codec",
            _ => "U8Codec",
        }
    }

    fn slang_layout(&self) -> &'static str {
        match self {
            PixelFormat::Rgba8 | PixelFormat::Rgba16 | PixelFormat::RgbaF32 => {
                "ChannelLayout::Rgba"
            }
            PixelFormat::Rgb8 | PixelFormat::Rgb16 | PixelFormat::RgbF32 => "ChannelLayout::Rgb",
            PixelFormat::Gray8 | PixelFormat::Gray16 => "ChannelLayout::Gray",
            _ => "ChannelLayout::Rgba",
        }
    }
}

// ── emit_slang — pure function: LayoutPlan + read-only Graph → Slang text ─────

#[derive(Debug)]
struct ColorSpaceData {
    transfer: u32,
    alpha: u32,
    model: u32,
    channels: u32,
    /// Row 0 of the 3×3 matrix, formatted as "a00f, a01f, a02f"
    mat_row0: String,
    /// Row 1
    mat_row1: String,
    /// Row 2
    mat_row2: String,
}

#[derive(Debug)]
struct ParamFieldData {
    name: String,
    ty: &'static str,
}

#[derive(Debug)]
struct TargetData {
    binding: u32,
    is_histogram: bool,
}

#[derive(Debug)]
struct SourceVarData {
    binding: u32,
    codec: &'static str,
    ch_layout: &'static str,
}

#[derive(Debug)]
struct TempVarData {
    binding: u32,
}

#[derive(Debug)]
struct KernelCallData {
    function: &'static str,
    args: Vec<String>,
}

#[derive(Debug)]
struct OutputEncodeData {
    region_param: String,
    target_name: String,
    dst_codec: &'static str,
    dst_ch_layout: &'static str,
    /// Reads from `temp_buf_{binding}` when set, else from the first source.
    temp_binding: Option<u32>,
    source_binding: Option<u32>,
}

#[derive(Debug)]
struct EntryData {
    name: String,
    bounds_check: String,
    source_vars: Vec<SourceVarData>,
    temp_vars: Vec<TempVarData>,
    histogram_target: Option<u32>,
    kernel_call: Option<KernelCallData>,
    output_encode: Option<OutputEncodeData>,
}

// ── Template struct ───────────────────────────────────────────────────────────

#[derive(Template)]
#[template(path = "shader.slang", escape = "none")]
struct ShaderTemplate {
    imports: Vec<String>,
    has_special_output: bool,
    sources: Vec<SourceBinding>,
    params_binding: u32,
    temp_bindings: Vec<u32>,
    num_temp_bindings: u32,
    targets: Vec<TargetData>,
    src_color_spaces: Vec<ColorSpaceData>,
    dst_cs: Option<ColorSpaceData>,
    param_fields: Vec<ParamFieldData>,
    entries: Vec<EntryData>,
    wg_dim: u32,
}

#[derive(Debug)]
struct SourceBinding {
    binding: u32,
}

// ── Askama filter: integer addition ──────────────────────────────────────────

mod filters {
    pub fn add(value: &u32, addend: &u32) -> ::askama::Result<u32> {
        Ok(value + addend)
    }
}

// ── Conversion helpers ────────────────────────────────────────────────────────

impl From<&GpuPixelEncoding> for ColorSpaceData {
    fn from(enc: &GpuPixelEncoding) -> Self {
        let m = enc.transform;
        ColorSpaceData {
            transfer: enc.transfer,
            alpha: enc.alpha,
            model: enc.model,
            channels: enc.channels,
            mat_row0: format!("{:.8}f, {:.8}f, {:.8}f", m.m[0], m.m[1], m.m[2]),
            mat_row1: format!("{:.8}f, {:.8}f, {:.8}f", m.m[3], m.m[4], m.m[5]),
            mat_row2: format!("{:.8}f, {:.8}f, {:.8}f", m.m[6], m.m[7], m.m[8]),
        }
    }
}

struct SlangEmitter<'a> {
    layout: &'a LayoutPlan,
    color: &'a ColorPipeline,
    graph: &'a Graph,
}

impl<'a> SlangEmitter<'a> {
    fn new(layout: &'a LayoutPlan, color: &'a ColorPipeline, graph: &'a Graph) -> Self {
        Self {
            layout,
            color,
            graph,
        }
    }

    /// Which node id this kernel's thread grid must cover, per the op's
    /// declared [`DispatchGrid`] — `Output` (default) is `id` itself;
    /// `Input(idx)` is whatever feeds that input slot (leaf source or an
    /// upstream computed node — reductions aren't always fed directly by
    /// a leaf, e.g. `blurred.histogram()`).
    fn dispatch_grid_node(&self, id: NodeId) -> NodeId {
        let Some(node) = self.graph.get_node(id) else {
            return id;
        };
        match node.op.dispatch_grid() {
            DispatchGrid::Output => id,
            DispatchGrid::Input(idx) => node.inputs.get(idx).copied().unwrap_or(id),
        }
    }

    /// Compile-time `(width, height)` for a node — leaf source view dims, or
    /// the rect `walk_inverse` assigned it (carried on its temp slot).
    fn node_dims(&self, id: NodeId) -> (u32, u32) {
        self.layout
            .temps
            .get(&id)
            .map(|t| t.dims)
            .or_else(|| {
                self.layout
                    .source_pos
                    .get(&id)
                    .and_then(|&si| self.layout.sources.get(si))
                    .map(|s| (s.view_width, s.view_height))
            })
            .or_else(|| {
                self.layout
                    .targets
                    .iter()
                    .find(|t| t.node_id == id)
                    .map(|t| t.dims)
            })
            .unwrap_or((1, 1))
    }

    /// Runtime bounds-check expression for a node — references whichever
    /// `BufferRegion` param backs it (`inputs_N` for a leaf source,
    /// `temp_region_N` for an upstream computed node, `region_target_N` for
    /// a materialization target).
    fn region_bounds_param(&self, id: NodeId) -> String {
        if let Some(&si) = self.layout.source_pos.get(&id) {
            format!("g_params[0].inputs_{si}")
        } else if let Some(tmp) = self.layout.temps.get(&id) {
            format!("g_params[0].temp_region_{}", tmp.binding)
        } else if let Some(&ti) = self.layout.target_pos.get(&id) {
            format!("g_params[0].region_target_{ti}")
        } else {
            "g_params[0].temp_region_0".into()
        }
    }

    fn source_vars_data(&self) -> Vec<SourceVarData> {
        self.layout
            .sources
            .iter()
            .map(|src| SourceVarData {
                binding: src.binding,
                codec: src.format.slang_codec(),
                ch_layout: src.format.slang_layout(),
            })
            .collect()
    }

    fn temp_vars_data(&self, id: NodeId, node: &super::graph::GraphNode) -> Vec<TempVarData> {
        let mut declared = BTreeSet::new();
        if let Some(tmp) = self.layout.temps.get(&id) {
            declared.insert(tmp.binding);
        }
        for &inp in &node.inputs {
            if let Some(tmp) = self.layout.temps.get(&inp) {
                declared.insert(tmp.binding);
            }
        }
        declared
            .into_iter()
            .map(|binding| TempVarData { binding })
            .collect()
    }

    fn kernel_call_data(
        &self,
        id: NodeId,
        node: &super::graph::GraphNode,
        is_histogram_target: bool,
    ) -> KernelCallData {
        let NodeEval::Kernel(k) = &node.eval;
        let encoders = node.op.input_encoders(node.inputs.len());
        let mut args: Vec<String> = vec!["idx".into()];
        for (slot, &inp) in node.inputs.iter().enumerate() {
            let encoder = encoders
                .get(slot)
                .unwrap_or(&InputEncoder::WorkingDecodeRegion);
            let name = if let Some(&si) = self.layout.source_pos.get(&inp) {
                // For image sources both `_raw_src_{si}` (CodecRegion) and
                // `region_src_{si}` (WorkingDecodeRegion) are declared by
                // the source-var entries; the op's InputEncoder picks which one.
                match encoder {
                    InputEncoder::CodecRegion => format!("_raw_src_{si}"),
                    InputEncoder::WorkingDecodeRegion => format!("region_src_{si}"),
                    InputEncoder::ComplexRegion => format!("complex_src_{si}"),
                    InputEncoder::MaskRegion => format!("mask_src_{si}"),
                }
            } else if let Some(tmp) = self.layout.temps.get(&inp) {
                // Temps hold working-space float4 — always WorkingDecodeRegion-compatible.
                format!("region_tmp_{}", tmp.binding)
            } else {
                "region_src_0".into()
            };
            args.push(name);
        }
        if is_histogram_target {
            let ti = self.layout.target_pos.get(&id).copied().unwrap_or(0);
            args.push(format!("hist_out_{ti}"));
        } else {
            let b = self.layout.temps.get(&id).map(|t| t.binding).unwrap_or(0);
            args.push(format!("region_tmp_{b}"));
        }
        let pi_base = self.layout.params.node_base.get(&id).copied().unwrap_or(0);
        for i in 0..node.params.len() as u32 {
            args.push(format!("g_params[0].u{}", pi_base + i));
        }
        KernelCallData {
            function: k.function,
            args,
        }
    }

    fn output_encode_data(
        &self,
        id: NodeId,
        final_target_id: Option<NodeId>,
    ) -> Option<OutputEncodeData> {
        if Some(id) != final_target_id {
            return None;
        }
        let ti = self.layout.target_pos.get(&id).copied().unwrap_or(0);
        let fmt = self.color.dst_format.unwrap_or(PixelFormat::RgbaF32);
        Some(OutputEncodeData {
            region_param: format!("region_target_{ti}"),
            target_name: format!("target_{ti}"),
            dst_codec: fmt.slang_codec(),
            dst_ch_layout: fmt.slang_layout(),
            temp_binding: self.layout.temps.get(&id).map(|t| t.binding),
            source_binding: self.layout.sources.first().map(|s| s.binding),
        })
    }

    fn emit(&self, wg_dim: u32) -> EmittedIr {
        use std::collections::BTreeMap;

        let imports: Vec<String> = self.layout.imports.iter().map(|s| s.to_string()).collect();
        let has_special = self
            .layout
            .targets
            .iter()
            .any(|t| !t.output.needs_fused_temp());

        let sources: Vec<SourceBinding> = self
            .layout
            .sources
            .iter()
            .map(|s| SourceBinding { binding: s.binding })
            .collect();
        let params_binding = self.layout.sources.len() as u32;

        let mut temp_binding_set: BTreeMap<u32, ()> = BTreeMap::new();
        for tmp in self.layout.temps.values() {
            temp_binding_set.insert(tmp.binding, ());
        }
        let temp_bindings: Vec<u32> = temp_binding_set.keys().copied().collect();
        let num_temp_bindings = temp_bindings.len() as u32;

        let targets: Vec<TargetData> = self
            .layout
            .targets
            .iter()
            .map(|t| TargetData {
                binding: t.binding,
                is_histogram: matches!(t.output.write_mode(), WriteMode::AtomicAccumulate { .. }),
            })
            .collect();

        let src_color_spaces: Vec<ColorSpaceData> = self
            .color
            .src_encodings
            .iter()
            .map(ColorSpaceData::from)
            .collect();
        let dst_cs = self.color.dst_encoding.as_ref().map(ColorSpaceData::from);

        let param_fields: Vec<ParamFieldData> = self
            .layout
            .params
            .fields
            .iter()
            .map(|(name, ty)| ParamFieldData {
                name: name.clone(),
                ty,
            })
            .collect();

        let final_target_id = self.layout.targets.first().map(|t| t.node_id);
        let mut entries: Vec<EntryData> = Vec::new();
        let mut j: u32 = 0;

        for &id in &self.layout.order {
            if self.layout.src_set.contains(&id) || !self.layout.needed.contains(&id) {
                continue;
            }
            let entry_name = format!("entry_{j}");
            j += 1;

            let output_kind = self
                .layout
                .targets
                .iter()
                .find(|t| t.node_id == id)
                .map(|t| t.output.clone())
                .unwrap_or_else(working_image_type);
            let is_atomic_accumulate =
                matches!(output_kind.write_mode(), WriteMode::AtomicAccumulate { .. });
            let is_target = final_target_id == Some(id) || self.layout.target_map.contains_key(&id);
            let is_histogram_target = is_atomic_accumulate && is_target;

            let grid_node = self.dispatch_grid_node(id);
            let bounds_check = if grid_node != id {
                let region = self.region_bounds_param(grid_node);
                format!("if (tid.x >= {region}.width || tid.y >= {region}.height) return;")
            } else if is_target {
                let ti = self.layout.target_pos.get(&id).copied().unwrap_or(0);
                format!(
                    "if (tid.x >= g_params[0].region_target_{ti}.width || tid.y >= g_params[0].region_target_{ti}.height) return;"
                )
            } else {
                let b = self.layout.temps.get(&id).map(|t| t.binding).unwrap_or(0);
                format!(
                    "if (tid.x >= g_params[0].temp_region_{b}.width || tid.y >= g_params[0].temp_region_{b}.height) return;"
                )
            };

            let source_vars = self.source_vars_data();
            let mut temp_vars = Vec::new();
            let mut histogram_target = None;
            let mut kernel_call = None;
            // Default `true` mirrors the pre-existing histogram special case —
            // a node missing from the graph can't declare an encoder, so don't
            // emit a wrap that has nothing to read from.
            let mut skip_encode = true;
            if let Some(node) = self.graph.get_node(id) {
                temp_vars = self.temp_vars_data(id, node);
                if is_histogram_target {
                    histogram_target =
                        Some(self.layout.target_pos.get(&id).copied().unwrap_or(0) as u32);
                }
                kernel_call = Some(self.kernel_call_data(id, node, is_histogram_target));
                // Skip the from_working + codec::encode step for non-image outputs
                // (histogram bins, raw masks, FFT, …).
                skip_encode = !matches!(
                    node.op.output_decoder(),
                    OutputDecoder::WorkingEncodeRegion { .. }
                );
            }
            let output_encode = if skip_encode {
                None
            } else {
                self.output_encode_data(id, final_target_id)
            };

            entries.push(EntryData {
                name: entry_name,
                bounds_check,
                source_vars,
                temp_vars,
                histogram_target,
                kernel_call,
                output_encode,
            });
        }

        let entry_points: Vec<(String, u32, u32)> = entries
            .iter()
            .zip(self.layout.order.iter().filter(|&&id| {
                !self.layout.src_set.contains(&id) && self.layout.needed.contains(&id)
            }))
            .map(|(e, &id)| {
                let grid_node = self.dispatch_grid_node(id);
                let (w, h) = self.node_dims(grid_node);
                (e.name.clone(), w, h)
            })
            .collect();

        let target_output_kinds: Vec<Arc<dyn DataType>> = self
            .layout
            .targets
            .iter()
            .map(|t| t.output.clone())
            .collect();

        let temp_buffer_sizes: Vec<u64> = {
            let mut by_binding: BTreeMap<u32, u64> = BTreeMap::new();
            for tmp in self.layout.temps.values() {
                by_binding.insert(tmp.binding, tmp.size_bytes);
            }
            by_binding.values().copied().collect()
        };

        let text = ShaderTemplate {
            imports,
            has_special_output: has_special,
            sources,
            params_binding,
            temp_bindings,
            num_temp_bindings,
            targets,
            src_color_spaces,
            dst_cs,
            param_fields,
            entries,
            wg_dim,
        }
        .render()
        .expect("Askama shader template render failed");

        EmittedIr {
            text,
            source_count: self.layout.sources.len(),
            target_count: self.layout.targets.len(),
            temp_buffer_sizes,
            params_bytes: self.layout.params.bytes.clone(),
            entry_points,
            target_output_kinds,
        }
    }
}
