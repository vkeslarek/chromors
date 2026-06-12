//! GPU/Slang lowering vocabulary. None of this is backend-agnostic — it's the
//! shader-side view of a buffer (`View`), the std430 params blob
//! (`ParamBlock`), and fused scratch (`TempElem`). Lives under `backend::gpu`
//! so the agnostic core never sees it.
use std::borrow::Cow;

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

    /// Expand the `ctor` template for an **input** slot, replacing `{buf}`
    /// (buffer var), `{params}` (ChainParams var), and `{slot}` (this input's
    /// slot index, e.g. `region_in_0`).
    pub fn input_expr(&self, buf_var: &str, slot: usize) -> String {
        self.ctor
            .replace("{buf}", buf_var)
            .replace("{params}", "params")
            .replace("{slot}", &slot.to_string())
    }
}

/// A Rust scalar type with a known Slang/std430 element type. Lets
/// `ParamBlock::param`/`GpuBuilder::param` derive the declared field type
/// from `T` instead of a separately-passed (and easily mismatched) string.
pub trait SlangScalar: bytemuck::Pod {
    const SLANG_TY: &'static str;
}
impl SlangScalar for f32 {
    const SLANG_TY: &'static str = "float";
}
impl SlangScalar for u32 {
    const SLANG_TY: &'static str = "uint";
}
impl SlangScalar for i32 {
    const SLANG_TY: &'static str = "int";
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

    pub fn param<T: SlangScalar>(mut self, name: &str, value: T) -> Self {
        self.fields.push((name.to_string(), T::SLANG_TY));
        self.bytes.extend_from_slice(bytemuck::bytes_of(&value));
        self
    }

    /// A helper to emit a basic scalar parameter.
    pub fn scalar<T: SlangScalar>(name: &str, value: T) -> Self {
        Self::new().param(name, value)
    }

    /// A field of an arbitrary std430 struct type (e.g. `"RemapGeo"`), for
    /// adapter geometry that isn't a single scalar.
    pub fn field<T: bytemuck::Pod>(mut self, name: &str, ty: &'static str, value: T) -> Self {
        self.fields.push((name.to_string(), ty));
        self.bytes.extend_from_slice(bytemuck::bytes_of(&value));
        self
    }
}

/// A zero-cost Slang view interposed between a producer (a source decode or a
/// prior step's working temp) and a consumer — no kernel step, no buffer. The
/// core only stores and splices these strings; the semantics (swizzle, remap,
/// …) live in the constructor function and the Slang struct it names.
///
/// `wrapper`/`ctor` carry three placeholders the builder/emitter expand:
/// - `{inner}` (wrapper only) — the wrapped `IRegion`'s Slang type.
/// - `{value}` (ctor only) — the wrapped value's variable name.
/// - `{params}` — the `ChainParams` instance name.
/// - `{p}` — this adapter's unique param-field prefix, assigned by
///   [`super::GpuBuilder::adapt`] (so two adapters' fields never collide).
#[derive(Debug, Clone)]
pub struct ViewAdapter {
    /// Wrapper type template, e.g. `"SwizzleView<{inner}>"`, `"RemapView<{inner}>"`.
    pub wrapper: Cow<'static, str>,
    /// Ctor expression template, e.g. `"{ {value}, {params}[0].{p}_channel }"`.
    pub ctor: Cow<'static, str>,
    /// Adapter params (field names contain `{p}`, substituted by `adapt`).
    pub params: ParamBlock,
    /// Slang module defining the wrapper struct (currently always `lib.region`,
    /// which the emitter imports unconditionally).
    pub module: &'static str,
}

/// The `BufferRegion { stride, x, y, w, h }` geometry of one buffer slot —
/// the std430 struct every Slang region wrapper indexes with. Emitter-owned
/// (universal to all Region-shaped data), named per slot (`region_in_{i}` /
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

    /// A fresh `ParamBlock` containing just this region under `name`. Used by
    /// `GpuView::output`/`input` to hand the geometry to `GpuBuilder` without
    /// it knowing it's a `BufferRegion`.
    pub fn into_block(self, name: &str) -> ParamBlock {
        let mut block = ParamBlock::new();
        self.push_into(&mut block, name);
        block
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
    /// Type + ctor + raw element type of the kernel's output argument
    /// (`arg.slang`/`arg.ctor`/`arg.buffer_type`, e.g. `"RWRegion"` /
    /// `"HistogramOut"`). `arg.buffer_type` is the `target_buffer` element
    /// type when `encode` is `None`; when `Some`, the encode view's
    /// `buffer_type` (the codec sandwich's scratch type) is used instead.
    pub arg: View,
    /// Where the argument's buffer comes from.
    pub dest: OutBuffer,
    /// Optional post-kernel step that lands the working result in the target —
    /// the image codec sandwich closes here. `None` ⇒ the kernel wrote the
    /// target directly.
    pub encode: Option<View>,
    /// Kind-owned output geometry/config (e.g. a histogram's `bin_count`),
    /// merged into `ChainParams` by `GpuBuilder::output`.
    pub params: ParamBlock,
}
