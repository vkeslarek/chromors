use super::view::OutBuffer;
use super::{BaseInput, GpuBuilder, StepInput, TempElem, View};
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};

/// Core Slang libraries imported unconditionally by every fused pass.
const CORE_MODULES: &[&str] = &["lib.region", "lib.io", "lib.codecs", "lib.pixel"];

/// One binding-group-0 entry, in emission/binding order. The single source of
/// truth for the fused pass's binding layout — used by `emit_slang` (variable
/// declarations), `compile::compile` (BGL `read_only` per entry), and
/// `compile::dispatch` (bind-group entries + work-buffer sizing).
pub enum Slot<'a> {
    /// Binding 0: the output buffer (codec sandwich target or direct wrapper).
    Target,
    /// Binding 1: the `ChainParams` SSBO.
    Params,
    /// One `work_{k}` temp per step that writes one (see `GpuBuilder::work_buffer_count`).
    Work(usize, &'a TempElem),
    /// One `src_{i}` decode buffer per source input.
    Source(usize, &'a View),
}

/// Iterate this pass's binding-group-0 slots in declaration/binding order.
pub fn slots(builder: &GpuBuilder) -> impl Iterator<Item = Slot<'_>> {
    let n_work = builder.work_buffer_count();
    std::iter::once(Slot::Target)
        .chain(std::iter::once(Slot::Params))
        .chain((0..n_work).map(move |k| Slot::Work(k, &builder.steps[k].temp_elem)))
        .chain(
            builder
                .input_views
                .iter()
                .enumerate()
                .map(|(i, v)| Slot::Source(i, v)),
        )
}

/// Collect the distinct `ops.*` modules referenced by this pass's steps and
/// any view adapters they (or the output) read through — deduped against
/// [`CORE_MODULES`], which every pass imports unconditionally.
fn referenced_modules(builder: &GpuBuilder) -> Vec<&'static str> {
    let mut seen = HashSet::new();
    let mut modules = Vec::new();
    let push =
        |m: &'static str, seen: &mut HashSet<&'static str>, modules: &mut Vec<&'static str>| {
            if !CORE_MODULES.contains(&m) && seen.insert(m) {
                modules.push(m);
            }
        };
    for step in &builder.steps {
        push(step.module, &mut seen, &mut modules);
        for inp in &step.inputs {
            if let Some(a) = &inp.adapter {
                push(a.module, &mut seen, &mut modules);
            }
        }
    }
    if let Some(a) = &builder.cur_output_adapter
        && let Some(adapter) = &a.adapter
    {
        push(adapter.module, &mut seen, &mut modules);
    }
    for inp in &builder.cur_inputs {
        if let Some(a) = &inp.adapter {
            push(a.module, &mut seen, &mut modules);
        }
    }
    modules
}

/// Resolve one `StepInput` to a Slang read expression, declaring any
/// intermediate variables it needs. Returns `(decl_lines, expr)`: `decl_lines`
/// are `    Type var = init;\n` statements to emit before use; `expr` is the
/// variable name (or `in_{i}`) to read from.
///
/// - `BaseInput::Source(i)` with no adapter → `in_{i}` directly, no decl.
/// - `BaseInput::Step(j)` with no adapter → declares `{wrapper} {var} = { work_{j}, params[0].domain };`.
/// - With an adapter → declares the base (as above if `Step`), then wraps it
///   in the adapter's `wrapper`/`ctor` templates (`{inner}` ← base's Slang
///   type, `{value}` ← base's variable, `{params}` ← `"params"`).
fn read_expr(builder: &GpuBuilder, input: &StepInput, var: &str) -> (String, String) {
    match &input.adapter {
        None => match input.base {
            BaseInput::Source(i) => (String::new(), format!("in_{i}")),
            BaseInput::Step(j) => {
                let wrapper = builder.steps[j].temp_elem.region_wrapper;
                let decl = format!("    {wrapper} {var} = {{ work_{j}, params[0].domain }};\n");
                (decl, var.to_string())
            }
        },
        Some(adapter) => {
            let (base_slang, base_var, mut decl) = match input.base {
                BaseInput::Source(i) => (
                    builder.input_views[i].slang.to_string(),
                    format!("in_{i}"),
                    String::new(),
                ),
                BaseInput::Step(j) => {
                    let wrapper = builder.steps[j].temp_elem.region_wrapper;
                    let base_var = format!("{var}_base");
                    let decl =
                        format!("    {wrapper} {base_var} = {{ work_{j}, params[0].domain }};\n");
                    (wrapper.to_string(), base_var, decl)
                }
            };
            let wrapper_ty = adapter.wrapper.replace("{inner}", &base_slang);
            let ctor = adapter
                .ctor
                .replace("{value}", &base_var)
                .replace("{params}", "params");
            decl.push_str(&format!("    {wrapper_ty} {var} = {ctor};\n"));
            (decl, var.to_string())
        }
    }
}

/// Emit one fused Slang compute shader from the builder state collected during
/// the lower walk. Fully generic — no datatype/op-specific branches. Each input
/// `View` and the output Kind's `OutputWrap` carry their own Slang wrapper type;
/// the emitter only wires bindings, chains kernel steps, and brackets the codec.
///
/// **Per-step temps (handles fusion + diamonds).** Each kernel step writes its
/// own `work_{s}` temp; inputs are resolved by index to a source (`in_{i}`) or a
/// prior step's temp (`work_{j}`). A node reachable by several consumers is
/// lowered once, so the diamond just reads its temp twice — no special case.
/// The final working temp is encoded into the target by the Kind's `encode`.
///
/// Binding layout (group 0):
/// ```text
///   0:        target     (RW)          ← codec output, or the direct wrapper buffer
///   1:        params     (StructuredBuffer<ChainParams>)
///   2..2+W:   work_k     (RW float4)   ← W = work_buffer_count() (one per temp-writing step)
///   2+W..:    src_i      (StructuredBuffer<…>)   ← one per input view
/// ```
pub fn emit_slang(builder: &GpuBuilder, wg_dim: u32) -> String {
    let output = builder
        .output
        .as_ref()
        .expect("a fused pass needs an output");
    let scratch = output.dest == OutBuffer::Scratch;

    let mut s = String::with_capacity(2048);

    for m in CORE_MODULES {
        writeln!(s, "import {m};").unwrap();
    }
    for m in referenced_modules(builder) {
        writeln!(s, "import {m};").unwrap();
    }
    s.push('\n');

    // ── ChainParams (one SSBO: per-slot BufferRegions + op scalars) ──────────
    s.push_str("struct ChainParams {\n");
    let mut seen = HashSet::new();
    for (name, ty) in &builder.params.fields {
        if seen.insert(name.clone()) {
            writeln!(s, "    {ty} {name};").unwrap();
        }
    }
    s.push_str("};\n\n");

    // ── bindings ─────────────────────────────────────────────────────────────
    // The target's element type: the encode wrapper's (sandwich) or the direct
    // out-arg wrapper's.
    let target_elem = output
        .encode
        .as_ref()
        .map(|e| e.buffer_type.as_ref())
        .unwrap_or(output.arg.buffer_type.as_ref());
    for (binding, slot) in slots(builder).enumerate() {
        match slot {
            Slot::Target => writeln!(
                s,
                "[[vk::binding({binding}, 0)]] RWStructuredBuffer<{target_elem}> target_buffer;"
            )
            .unwrap(),
            Slot::Params => writeln!(
                s,
                "[[vk::binding({binding}, 0)]] StructuredBuffer<ChainParams> params;"
            )
            .unwrap(),
            Slot::Work(k, elem) => writeln!(
                s,
                "[[vk::binding({binding}, 0)]] RWStructuredBuffer<{}> work_{k};",
                elem.buffer_ty
            )
            .unwrap(),
            Slot::Source(i, view) => writeln!(
                s,
                "[[vk::binding({binding}, 0)]] StructuredBuffer<{}> src_{i};",
                view.buffer_type
            )
            .unwrap(),
        }
    }

    // ── main ─────────────────────────────────────────────────────────────────
    writeln!(
        s,
        "\n[shader(\"compute\")]\n[numthreads({wg_dim}, {wg_dim}, 1)]"
    )
    .unwrap();
    s.push_str("void main(uint3 dispatchThreadID : SV_DispatchThreadID) {\n");
    s.push_str("    uint2 idx = dispatchThreadID.xy;\n");
    // Workgroup-aligned dispatch overshoots the domain when w/h aren't
    // multiples of the workgroup size; without this guard those extra
    // threads still run kernels (e.g. histogram's InterlockedAdd) on
    // out-of-range reads.
    s.push_str("    if (idx.x >= params[0].domain.width || idx.y >= params[0].domain.height) { return; }\n\n");

    // Source inputs: each decodes from its codec buffer via the source View.
    // Steps reference these (`in_{i}`) by their `BaseInput::Source` slot.
    for (i, view) in builder.input_views.iter().enumerate() {
        let init = view.input_expr(&format!("src_{i}"), i);
        writeln!(s, "    {} in_{i} = {init};", view.slang).unwrap();
    }
    s.push('\n');

    // Steps in topo order. Each step `s` writes its own temp `work_{s}`; a later
    // step (or a downstream node) reads it via `BaseInput::Step(s)`. A node
    // reachable by several consumers was lowered once, so its temp is read by
    // index — diamonds need no special handling here.
    let n_steps = builder.steps.len();
    for (s_i, step) in builder.steps.iter().enumerate() {
        // Resolve this step's read arguments.
        let mut read_args: Vec<String> = Vec::with_capacity(step.inputs.len());
        for (k, inp) in step.inputs.iter().enumerate() {
            let var = format!("r_{s_i}_{k}");
            let (decl, expr) = read_expr(builder, inp, &var);
            s.push_str(&decl);
            read_args.push(expr);
        }

        // This step's output: its own temp, unless it's the final step of a
        // direct (atomic) output, which writes the target wrapper.
        let is_last = s_i + 1 == n_steps;
        let direct_final = is_last && !scratch;
        let out_var = format!("out_{s_i}");
        if direct_final {
            let init = output
                .arg
                .init_expr("target_buffer", "params", "params[0].region_out");
            writeln!(s, "    {} {out_var} = {init};", output.arg.slang).unwrap();
        } else {
            let wrapper = step.temp_elem.region_wrapper;
            writeln!(
                s,
                "    {wrapper} {out_var} = {{ work_{s_i}, params[0].domain }};"
            )
            .unwrap();
        }

        let mut args: Vec<String> = Vec::with_capacity(2 + read_args.len() + step.params.len());
        args.push("idx".into());
        args.extend(read_args);
        args.push(out_var.clone());
        for p in &step.params {
            args.push(format!("params[0].{p}"));
        }
        writeln!(s, "    {}({});\n", step.kernel, args.join(", ")).unwrap();

        // Codec sandwich close on the final step (image outputs only): encode
        // the final working temp into the target.
        if is_last && let Some(encode) = &output.encode {
            let init = encode.init_expr("target_buffer", "params", "params[0].region_out");
            writeln!(s, "    {} enc = {init};", encode.slang).unwrap();
            if let Some(adapted) = &builder.cur_output_adapter {
                // The DAG root is a pure view adapter (e.g. swizzle/remap) of
                // the final working temp: encode through the adapter instead
                // of the temp's raw value.
                let (decl, expr) = read_expr(builder, adapted, "out_adapt");
                s.push_str(&decl);
                writeln!(s, "    enc.write(idx, {expr}.read(idx));").unwrap();
            } else {
                writeln!(s, "    enc.write(idx, {out_var}.read(idx));").unwrap();
            }
        }
    }

    // A pure alias of a source with zero kernel steps (e.g. `extract_band` or
    // `flip` applied directly to a freshly-opened image — no other ops in the
    // chain). The per-step loop above never ran, so encode straight from the
    // resolved (and possibly adapted) source read here.
    if n_steps == 0 {
        if let Some(encode) = &output.encode {
            let init = encode.init_expr("target_buffer", "params", "params[0].region_out");
            writeln!(s, "    {} enc = {init};", encode.slang).unwrap();
            let input = builder
                .cur_output_adapter
                .clone()
                .or_else(|| builder.cur_inputs.first().cloned());
            if let Some(input) = input {
                let (decl, expr) = read_expr(builder, &input, "out_adapt");
                s.push_str(&decl);
                writeln!(s, "    enc.write(idx, {expr}.read(idx));").unwrap();
            }
        }
    }

    s.push_str("}\n");
    s
}

/// Hashes the generated Slang text to form the pipeline cache key.
pub fn hash_slang(slang: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    slang.hash(&mut hasher);
    hasher.finish()
}
