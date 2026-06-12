//! GPU/Slang lowering vocabulary. None of this is backend-agnostic — it's the
//! shader-side view of a buffer (`View`/`Binding`), the std430 params blob
//! (`ParamBlock`), fused scratch (`TempSpec`), and the three shader slots
//! (`Role`). Lives under `backend::gpu` so the agnostic core never sees it.
use std::borrow::Cow;

/// The three kinds of buffer a kernel touches: an `Input` it reads, an
/// `Output` it writes, a `Temporary` scratch it both reads and writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    Input,
    Output,
    Temporary,
}

/// Bind group and binding slot for a View.
#[derive(Debug, Clone)]
pub struct Binding {
    pub group: u32,
    pub binding: u32,
}

/// A Slang-side view of a raw buffer. Carries three things the emitter needs:
///
/// 1. `buffer_type` — the raw element type for the `StructuredBuffer<T>` /
///    `RWStructuredBuffer<T>` declaration (e.g. `"uint"`, `"float4"`, `"float2"`).
///
/// 2. `slang` — the Slang wrapper struct name that the kernel function
///    receives (e.g. `"Region"`, `"CodecRegion<U8Codec, 4>"`, `"HistogramOut"`).
///
/// 3. `ctor` — a Slang struct-init expression with two placeholders:
///    `{buf}` for the buffer variable name and `{params}` for the
///    `ChainParams` instance name. The emitter does literal replacement.
///    Examples:
///    - `"{ {buf}, { {params}.stride, {params}.x, {params}.y, {params}.w, {params}.h } }"`
///    - `"{ {buf}, {params}.bin_count }"`
///
/// The `View` is produced by `GpuView::view(role)` on each concrete Kind.
/// The emitter never inspects the Kind — it uses these three strings verbatim.
#[derive(Debug, Clone)]
pub struct View {
    pub buffer_type: Cow<'static, str>,
    pub slang: Cow<'static, str>,
    pub ctor: Cow<'static, str>,
    pub binding: Binding,
}

impl View {
    /// Creates a view with explicit buffer type, Slang wrapper type, and
    /// constructor expression.
    pub fn new(
        buffer_type: impl Into<Cow<'static, str>>,
        slang: impl Into<Cow<'static, str>>,
        ctor: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            buffer_type: buffer_type.into(),
            slang: slang.into(),
            ctor: ctor.into(),
            binding: Binding { group: 0, binding: 0 },
        }
    }

    /// Expand the `ctor` template, replacing `{buf}` (buffer var), `{params}`
    /// (ChainParams var), and `{region}` (this slot's `BufferRegion` accessor,
    /// e.g. `params[0].region_out`).
    pub fn init_expr(&self, buf_var: &str, params_var: &str, region_expr: &str) -> String {
        self.ctor
            .replace("{buf}", buf_var)
            .replace("{params}", params_var)
            .replace("{region}", region_expr)
    }
}

/// One std430 buffer containing geometry params (from Kind) and scalar configs (from Operation).
#[derive(Debug, Clone, Default)]
pub struct ParamBlock {
    pub fields: Vec<(String, &'static str)>,
    pub bytes: Vec<u8>,
}

impl ParamBlock {
    pub fn new() -> Self {
        Self { fields: vec![], bytes: vec![] }
    }
    
    pub fn param<T: bytemuck::Pod>(mut self, name: &str, slang_type: &'static str, value: T) -> Self {
        self.fields.push((name.to_string(), slang_type));
        self.bytes.extend_from_slice(bytemuck::bytes_of(&value));
        self
    }

    pub fn empty() -> Self {
        Self::new()
    }
    /// A helper to emit a basic scalar parameter.
    pub fn scalar<T: bytemuck::Pod>(name: &str, slang_type: &'static str, value: T) -> Self {
        Self::new().param(name, slang_type, value)
    }
}

/// The `BufferRegion { stride, x, y, w, h }` geometry of one buffer slot —
/// the std430 struct every Slang region wrapper indexes with. Emitter-owned
/// (universal to all `Shape::Region` data), named per slot (`region_in_{i}` /
/// `region_out`) so slots never collide.
#[derive(Debug, Clone, Copy)]
pub struct RegionParams {
    pub stride: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl RegionParams {
    /// A tight, origin-aligned region covering a `w×h` buffer.
    pub fn tight(w: i32, h: i32) -> Self {
        let (w, h) = (w.max(0) as u32, h.max(0) as u32);
        Self { stride: w, x: 0, y: 0, w, h }
    }
    /// Push this region as a named `BufferRegion` field (+ std430 bytes) onto a
    /// `ChainParams` block.
    pub fn push_into(&self, block: &mut ParamBlock, name: &str) {
        block.fields.push((name.to_string(), "BufferRegion"));
        for v in [self.stride, self.x, self.y, self.w, self.h] {
            block.bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Which buffer a kernel's output argument binds to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutBuffer {
    /// A `float4` scratch the emitter allocates — the kernel writes working
    /// values here, then the Kind's `encode` step lands them in the target.
    Scratch,
    /// The result buffer directly (e.g. a histogram's atomic accumulate target).
    Target,
}

/// How an output of a given Kind is written into a fused shader — **supplied by
/// the Kind itself** (`GpuView::output`), never classified by the emitter. An
/// image returns its codec sandwich (write working `float4`, then encode); a
/// histogram returns a direct atomic write. The emitter just splices whatever
/// fragments this carries.
#[derive(Debug, Clone)]
pub struct OutputWrap {
    /// Slang type of the kernel's output argument (`"RWRegion"`, `"HistogramOut"`).
    pub arg_type: Cow<'static, str>,
    /// Constructor for that argument, with `{buf}`/`{params}`/`{region}` holes.
    pub arg_ctor: Cow<'static, str>,
    /// Where the argument's buffer comes from.
    pub arg_buffer: OutBuffer,
    /// Raw element type for the `target_buffer` declaration (e.g. `"uint"` for
    /// a `HistogramOut` wrapper around `RWStructuredBuffer<uint>`). Only used
    /// when `encode` is `None` — when `Some`, the encode view's `buffer_type`
    /// (the codec sandwich's scratch type) is used instead.
    pub buffer_type: Cow<'static, str>,
    /// Optional post-kernel step that lands the working result in the target —
    /// the image codec sandwich closes here. `None` ⇒ the kernel wrote the
    /// target directly.
    pub encode: Option<View>,
}
