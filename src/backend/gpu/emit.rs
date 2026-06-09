use askama::Template;
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::color::space::ColorSpace;
use crate::pixel::PixelFormat;

use super::graph::NodeEval;
use super::graph::{Graph, NodeId};
use super::materialize::MaterializePlan;
use super::op::{Decoder, DispatchGrid, Encoder};
use super::param::{GpuPixelEncoding, Param};
use super::source::AnyGpuSource;
use super::value::{ValueKind, WriteMode};

// ── Public output ─────────────────────────────────────────────────────────────

pub struct EmittedIr {
    pub text: String,
    pub source_count: usize,
    pub target_count: usize,
    pub temp_buffer_sizes: Vec<u64>,
    pub params_bytes: Vec<u8>,
    pub entry_points: Vec<(String, u32, u32)>,
    pub target_output_kinds: Vec<ValueKind>,
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
    pub fn emit_ir(&self, graph: &Graph, wg_dim: u32) -> EmittedIr {
        let (layout, color) = LayoutPlan::build(graph, self);
        SlangEmitter::new(&layout, &color, graph).emit(wg_dim)
    }

    /// Like `emit_ir` but also returns the `LayoutPlan` so callers can
    /// apply LOD-dependent param scaling without rebuilding the layout.
    pub fn emit_ir_with_layout(&self, graph: &Graph, wg_dim: u32) -> (EmittedIr, LayoutPlan) {
        let (layout, color) = LayoutPlan::build(graph, self);
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
#[derive(Debug)]
pub struct ColorPipeline {
    /// Per-source color encoding (one per [`SourceSlot`]).
    pub src_encodings: Vec<GpuPixelEncoding>,
    /// Output color encoding — derived from the last node's `output_codec_override`,
    /// or the source color space if no override is present.
    pub dst_encoding: GpuPixelEncoding,
    /// Output pixel format — used to select the Slang codec and compute buffer sizes.
    pub dst_format: PixelFormat,
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
    pub output: ValueKind,
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
    /// Field indices (into `fields`) that contain full-resolution pixel-space
    /// magnitudes (e.g. blur sigma) and must be divided by `lod.scale_factor()`
    /// before GPU dispatch when materialising at LOD > 0.
    pub lod_scale_fields: Vec<u32>,
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
    /// Build the layout plan and color pipeline for one materialization pass.
    pub fn build(graph: &Graph, plan: &MaterializePlan) -> (Self, ColorPipeline) {
        LayoutBuilder::new(graph, plan).build()
    }
}

struct LayoutBuilder<'a> {
    graph: &'a Graph,
    plan: &'a MaterializePlan,
}

impl<'a> LayoutBuilder<'a> {
    fn new(graph: &'a Graph, plan: &'a MaterializePlan) -> Self {
        Self { graph, plan }
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

        let params = self.build_params(&order, &needed, &sources, &temps, &targets);

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

        let dst_encoding = {
            let override_codec = order.iter().rev().find_map(|&id| {
                self.graph
                    .get_node(id)
                    .and_then(|n| n.op.output_codec_override())
            });
            match override_codec {
                Some(codec) => {
                    let meta = crate::pixel::PixelMeta::new(
                        codec.format,
                        codec.color_space,
                        crate::pixel::AlphaPolicy::Straight,
                    );
                    GpuPixelEncoding::from_meta(&meta, false)
                }
                None => {
                    let img =
                        self.plan.image.as_ref().expect(
                            "MaterializePlan::image must be set for image materializations",
                        );
                    GpuPixelEncoding::from_color_space(img.dst_color_space)
                }
            }
        };

        let img_plan = self
            .plan
            .image
            .as_ref()
            .expect("MaterializePlan::image must be set for image materializations");
        let color = ColorPipeline {
            src_encodings,
            dst_encoding,
            dst_format: img_plan.dst_format,
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
                .map(|n| n.output.clone())
                .unwrap_or(ValueKind::Image);
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
                    .map(|n| n.output.clone())
                    .unwrap_or(ValueKind::Image);
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

        let node_params_global_base = fields.len() as u32;
        let mut node_base: HashMap<NodeId, u32> = HashMap::new();
        let mut lod_scale_fields: Vec<u32> = Vec::new();
        let mut ui = 0u32;
        for &id in order.iter().filter(|&&id| needed.contains(&id)) {
            if let Some(node) = self.graph.get_node(id) {
                node_base.insert(id, ui);
                let scale_indices = node.op.lod_scale_param_indices();
                for (param_local_idx, p) in node.params.iter().enumerate() {
                    if scale_indices.contains(&param_local_idx) {
                        lod_scale_fields.push(node_params_global_base + ui);
                    }
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
            lod_scale_fields,
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
struct EntryData {
    name: String,
    bounds_check: String,
    body: String,
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
    dst_cs: ColorSpaceData,
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

    fn build_source_vars(&self) -> String {
        let mut out = String::new();
        for src in &self.layout.sources {
            let i = src.binding;
            let codec = src.format.slang_codec();
            let ch_layout = src.format.slang_layout();
            out.push_str(&format!(
                "    CodecRegion<{codec}, {ch_layout}> _raw_src_{i} = {{ src_{i}, g_params[0].inputs_{i} }};\n"
            ));
            out.push_str(&format!(
                "    WorkingDecodeRegion<CodecRegion<{codec}, {ch_layout}>> region_src_{i} = {{ _raw_src_{i}, src_cs_{i} }};\n"
            ));
        }
        if !self.layout.sources.is_empty() {
            out.push('\n');
        }
        out
    }

    fn build_temp_vars(&self, id: NodeId, node: &super::graph::GraphNode) -> String {
        let mut out = String::new();
        let mut declared = BTreeSet::new();
        if let Some(tmp) = self.layout.temps.get(&id) {
            let b = tmp.binding;
            declared.insert(b);
            out.push_str(&format!(
                "    RWRegion region_tmp_{b} = {{ temp_buf_{b}, g_params[0].temp_region_{b} }};\n"
            ));
        }
        for &inp in &node.inputs {
            if let Some(tmp) = self.layout.temps.get(&inp) {
                let b = tmp.binding;
                if declared.insert(b) {
                    out.push_str(&format!(
                        "    RWRegion region_tmp_{b} = {{ temp_buf_{b}, g_params[0].temp_region_{b} }};\n"
                    ));
                }
            }
        }
        out
    }

    fn build_kernel_call(
        &self,
        id: NodeId,
        node: &super::graph::GraphNode,
        is_histogram_target: bool,
    ) -> String {
        let NodeEval::Kernel(k) = &node.eval;
        let decoders = node.op.input_decoders(node.inputs.len());
        let mut args: Vec<String> = vec!["idx".into()];
        for (slot, &inp) in node.inputs.iter().enumerate() {
            let decoder = decoders.get(slot).unwrap_or(&Decoder::WorkingSpace);
            let name = if let Some(&si) = self.layout.source_pos.get(&inp) {
                // `_raw_src_{si}` (undecoded `CodecRegion`) vs `region_src_{si}`
                // (`WorkingDecodeRegion`-wrapped) — both are always declared by
                // `build_source_vars`; the op's `Decoder` picks which view it reads.
                match decoder {
                    Decoder::Passthrough => format!("_raw_src_{si}"),
                    Decoder::WorkingSpace => format!("region_src_{si}"),
                }
            } else if let Some(tmp) = self.layout.temps.get(&inp) {
                // Temps only exist for `Image`-kind nodes, which are always
                // `WorkingSpace` — they hold working-space `float4`, no raw view.
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
        format!("    {}({});\n", k.function, args.join(", "))
    }

    fn build_output_encode(&self, id: NodeId, final_target_id: Option<NodeId>) -> String {
        if Some(id) != final_target_id {
            return String::new();
        }
        let ti = self.layout.target_pos.get(&id).copied().unwrap_or(0);
        let tname = format!("target_{ti}");
        let region_param = format!("region_target_{ti}");
        let dst_codec = self.color.dst_format.slang_codec();
        let dst_ch_layout = self.color.dst_format.slang_layout();
        let mut out = String::new();
        out.push('\n');
        out.push_str("    float4 _packed;\n");
        out.push_str("    float _extra;\n");
        if let Some(tmp) = self.layout.temps.get(&id) {
            let b = tmp.binding;
            out.push_str(&format!(
                "    from_working(temp_buf_{b}[region_index(g_params[0].temp_region_{b}, idx.x, idx.y)], dst_cs, _packed, _extra);\n"
            ));
        } else if let Some(src) = self.layout.sources.first() {
            let si = src.binding;
            out.push_str(&format!(
                "    from_working(region_src_{si}.read(idx), dst_cs, _packed, _extra);\n"
            ));
        }
        out.push_str(&format!(
            "    if (idx.x < g_params[0].{region_param}.width && idx.y < g_params[0].{region_param}.height) {{\n"
        ));
        out.push_str(&format!(
            "        uint _linear = region_index(g_params[0].{region_param}, idx.x, idx.y);\n"
        ));
        out.push_str(&format!(
            "        {}::encode({tname}, _linear, _packed, {dst_ch_layout});\n",
            dst_codec
        ));
        out.push_str("    }\n");
        out
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
        let dst_cs = ColorSpaceData::from(&self.color.dst_encoding);

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
                .unwrap_or(ValueKind::Image);
            let is_atomic_accumulate =
                matches!(output_kind.write_mode(), WriteMode::AtomicAccumulate { .. });
            let is_target = final_target_id == Some(id) || self.layout.target_map.contains_key(&id);
            let is_histogram_target = is_atomic_accumulate && is_target;

            let grid_node = self.dispatch_grid_node(id);
            let bounds_check = if grid_node != id {
                let region = self.region_bounds_param(grid_node);
                format!(
                    "if (tid.x >= {region}.width || tid.y >= {region}.height) return;"
                )
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

            let mut body = self.build_source_vars();
            // Default `true` mirrors the pre-existing histogram special case —
            // a node missing from the graph can't declare an encoder, so don't
            // emit a wrap that has nothing to read from.
            let mut skip_encode = true;
            if let Some(node) = self.graph.get_node(id) {
                body.push_str(&self.build_temp_vars(id, node));
                if is_histogram_target {
                    let ti = self.layout.target_pos.get(&id).copied().unwrap_or(0);
                    body.push_str(&format!(
                        "    HistogramOut hist_out_{ti} = {{ target_{ti}, g_params[0].bin_count_{ti} }};\n"
                    ));
                }
                body.push_str(&self.build_kernel_call(id, node, is_histogram_target));
                // The op declares whether its raw result needs the
                // `from_working` + `codec::encode` wrap or writes straight
                // through (histogram bins, scalars, raw masks/FFT, …).
                skip_encode = matches!(node.op.output_encoder(), Encoder::Passthrough);
            }
            if !skip_encode {
                body.push_str(&self.build_output_encode(id, final_target_id));
            }

            entries.push(EntryData {
                name: entry_name,
                bounds_check,
                body,
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

        let target_output_kinds: Vec<ValueKind> = self
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
