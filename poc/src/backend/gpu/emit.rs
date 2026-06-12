use super::{GpuBuilder, StepInput, TempElem};
use super::view::OutBuffer;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

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
    let output = builder.output.as_ref().expect("a fused pass needs an output");
    let scratch = output.arg_buffer == OutBuffer::Scratch;
    let n_work = builder.work_buffer_count();

    let mut s = String::with_capacity(2048);

    s.push_str("import lib.region;\n");
    s.push_str("import lib.io;\n");
    s.push_str("import lib.codecs;\n");
    s.push_str("import lib.pixel;\n");
    // POC: import every ops module the builder might reference. A production
    // emitter would carry each kernel's module path on `KernelCall`.
    s.push_str("import ops.invert;\n");
    s.push_str("import ops.gaussian_blur;\n");
    s.push_str("import ops.histogram;\n");
    s.push_str("import ops.arithmetic;\n");
    s.push_str("import ops.bands;\n");
    s.push_str("import ops.composite;\n");
    s.push_str("import ops.convolution;\n");
    s.push_str("import ops.exposure;\n");
    s.push_str("import ops.gamma;\n");
    s.push_str("import ops.passthrough;\n");
    s.push_str("import ops.vectorscope;\n");
    s.push_str("import ops.opacity;\n");
    s.push_str("import ops.saturation;\n");
    s.push_str("import ops.shrink;\n");
    s.push_str("import ops.unary;\n");
    s.push_str("import ops.data_driven;\n");
    s.push_str("import ops.resample;\n");
    s.push_str("import ops.geometry_extended;\n\n");

    // ── ChainParams (one SSBO: per-slot BufferRegions + op scalars) ──────────
    s.push_str("struct ChainParams {\n");
    let mut seen = HashSet::new();
    for (name, ty) in &builder.params.fields {
        if seen.insert(name.clone()) {
            let slang_ty = if *ty == "scalar" { "float" } else { ty };
            s.push_str(&format!("    {slang_ty} {name};\n"));
        }
    }
    s.push_str("};\n\n");

    // ── bindings ─────────────────────────────────────────────────────────────
    // The target's element type: the encode wrapper's (sandwich) or the direct
    // out-arg wrapper's.
    let target_elem = output.encode.as_ref().map(|e| e.buffer_type.as_ref())
        .unwrap_or(output.buffer_type.as_ref());
    let mut binding = 0u32;
    s.push_str(&format!("[[vk::binding({binding}, 0)]] RWStructuredBuffer<{target_elem}> target_buffer;\n"));
    binding += 1;
    s.push_str(&format!("[[vk::binding({binding}, 0)]] StructuredBuffer<ChainParams> params;\n"));
    binding += 1;
    for k in 0..n_work {
        let elem = builder.steps[k].temp_elem.buffer_ty;
        s.push_str(&format!("[[vk::binding({binding}, 0)]] RWStructuredBuffer<{elem}> work_{k};\n"));
        binding += 1;
    }
    for (i, view) in builder.input_views.iter().enumerate() {
        s.push_str(&format!("[[vk::binding({binding}, 0)]] StructuredBuffer<{}> src_{i};\n", view.buffer_type));
        binding += 1;
    }

    // ── main ─────────────────────────────────────────────────────────────────
    s.push_str(&format!("\n[shader(\"compute\")]\n[numthreads({wg_dim}, {wg_dim}, 1)]\n"));
    s.push_str("void main(uint3 dispatchThreadID : SV_DispatchThreadID) {\n");
    s.push_str("    uint2 idx = dispatchThreadID.xy;\n\n");

    // Source inputs: each decodes from its codec buffer via the source View.
    // Steps reference these (`in_{i}`) by their `StepInput::Source` slot.
    for (i, view) in builder.input_views.iter().enumerate() {
        let region = format!("params[0].region_in_{i}");
        let init = view.init_expr(&format!("src_{i}"), "params", &region);
        s.push_str(&format!("    {} in_{i} = {};\n", view.slang, init));
    }
    s.push('\n');

    // Steps in topo order. Each step `s` writes its own temp `work_{s}`; a later
    // step (or a downstream node) reads it via `StepInput::Step(s)`. A node
    // reachable by several consumers was lowered once, so its temp is read by
    // index — diamonds need no special handling here.
    let n_steps = builder.steps.len();
    for (s_i, step) in builder.steps.iter().enumerate() {
        // Resolve this step's read arguments.
        let mut read_args: Vec<String> = Vec::with_capacity(step.inputs.len());
        for (k, inp) in step.inputs.iter().enumerate() {
            match inp {
                StepInput::Source(i) => read_args.push(format!("in_{i}")),
                StepInput::Step(j) => {
                    // Prior step's temp is RW-bound → wrap in its region
                    // wrapper (an `IRegion`, so kernels accept it).
                    let wrapper = builder.steps[*j].temp_elem.region_wrapper;
                    let var = format!("r_{s_i}_{k}");
                    s.push_str(&format!(
                        "    {wrapper} {var} = {{ work_{j}, params[0].region_out }};\n"
                    ));
                    read_args.push(var);
                }
                StepInput::SwizzleSource(i, c) => {
                    // A "free" alias of a source decode (e.g. ExtractBand
                    // directly on a freshly-opened image): read one component
                    // of `in_{i}` through `SwizzleView`.
                    let var = format!("r_{s_i}_{k}");
                    s.push_str(&format!(
                        "    SwizzleView<{}> {var} = {{ in_{i}, {c}u }};\n",
                        builder.input_views[*i].slang
                    ));
                    read_args.push(var);
                }
                StepInput::SwizzleStep(j, c) => {
                    // A "free" alias (e.g. ExtractBand): read one component
                    // of a prior step's temp through `SwizzleView`.
                    let wrapper = builder.steps[*j].temp_elem.region_wrapper;
                    let var = format!("r_{s_i}_{k}");
                    s.push_str(&format!(
                        "    SwizzleView<{wrapper}> {var} = {{ {{ work_{j}, params[0].region_out }}, {c}u }};\n"
                    ));
                    read_args.push(var);
                }
                StepInput::RemapSource(i, kind, rp) => {
                    let var = format!("r_{s_i}_{k}");
                    let inner_view = &builder.input_views[*i];
                    s.push_str(&format!(
                        "    RemapView<{}> {var} = {{ in_{i}, {}u, {}u, {}u, {:?}, {:?}, {}u, {}u, {}, {} }};\n",
                        inner_view.slang, *kind as u32, rp.out_w, rp.out_h, rp.sx, rp.sy, rp.in_w, rp.in_h, rp.tx, rp.ty
                    ));
                    read_args.push(var);
                }
                StepInput::RemapStep(j, kind, rp) => {
                    let wrapper = builder.steps[*j].temp_elem.region_wrapper;
                    let var = format!("r_{s_i}_{k}");
                    s.push_str(&format!(
                        "    RemapView<{wrapper}> {var} = {{ {{ work_{j}, params[0].region_out }}, {}u, {}u, {}u, {:?}, {:?}, {}u, {}u, {}, {} }};\n",
                        *kind as u32, rp.out_w, rp.out_h, rp.sx, rp.sy, rp.in_w, rp.in_h, rp.tx, rp.ty
                    ));
                    read_args.push(var);
                }
            }
        }

        // This step's output: its own temp, unless it's the final step of a
        // direct (atomic) output, which writes the target wrapper.
        let is_last = s_i + 1 == n_steps;
        let direct_final = is_last && !scratch;
        let out_var = format!("out_{s_i}");
        if direct_final {
            let init = output.arg_ctor
                .replace("{buf}", "target_buffer")
                .replace("{params}", "params")
                .replace("{region}", "params[0].region_out");
            s.push_str(&format!("    {} {out_var} = {init};\n", output.arg_type));
        } else {
            let wrapper = step.temp_elem.region_wrapper;
            s.push_str(&format!(
                "    {wrapper} {out_var} = {{ work_{s_i}, params[0].region_out }};\n"
            ));
        }

        let mut args: Vec<String> = Vec::with_capacity(2 + read_args.len() + step.params.len());
        args.push("idx".into());
        args.extend(read_args);
        args.push(out_var.clone());
        for p in &step.params {
            args.push(format!("params[0].{p}"));
        }
        s.push_str(&format!("    {}({});\n\n", step.kernel, args.join(", ")));

        // Codec sandwich close on the final step (image outputs only): encode
        // the final working temp into the target.
        if is_last && let Some(encode) = &output.encode {
            let init = encode.init_expr("target_buffer", "params", "params[0].region_out");
            s.push_str(&format!("    {} enc = {};\n", encode.slang, init));
            if let Some(c) = builder.output_swizzle {
                // The DAG root is a pure alias: broadcast the swizzled
                // component back into this temp's full shape before encode.
                let comp = step.temp_elem.component(c);
                let scalar = format!("{out_var}.read(idx).{comp}");
                let broadcast = step.temp_elem.broadcast_expr(&scalar);
                s.push_str(&format!("    enc.write(idx, {broadcast});\n"));
            } else {
                s.push_str(&format!("    enc.write(idx, {out_var}.read(idx));\n"));
            }
        }
    }

    // A pure alias of a source with zero kernel steps (e.g. `extract_band`
    // applied directly to a freshly-opened image — no other ops in the
    // chain). The per-step loop above never ran, so encode straight from the
    // resolved source read here.
    if n_steps == 0 {
        if let Some(encode) = &output.encode {
            let init = encode.init_expr("target_buffer", "params", "params[0].region_out");
            s.push_str(&format!("    {} enc = {};\n", encode.slang, init));
            match (builder.cur_inputs.first(), builder.output_swizzle) {
                (Some(StepInput::SwizzleSource(i, c)), Some(_)) => {
                    let comp = TempElem::F4.component(*c);
                    let scalar = format!("in_{i}.read(idx).{comp}");
                    let broadcast = TempElem::F4.broadcast_expr(&scalar);
                    s.push_str(&format!("    enc.write(idx, {broadcast});\n"));
                }
                (Some(StepInput::RemapSource(i, kind, rp)), _) => {
                    let var = format!("remap_0");
                    let inner_view = &builder.input_views[*i];
                    s.push_str(&format!(
                        "    RemapView<{}> {var} = {{ in_{i}, {}u, {}u, {}u, {:?}, {:?}, {}u, {}u, {}, {} }};\n",
                        inner_view.slang, *kind as u32, rp.out_w, rp.out_h, rp.sx, rp.sy, rp.in_w, rp.in_h, rp.tx, rp.ty
                    ));
                    s.push_str(&format!("    enc.write(idx, {var}.read(idx));\n"));
                }
                (Some(StepInput::Source(i)), _) => {
                    s.push_str(&format!("    enc.write(idx, in_{i}.read(idx));\n"));
                }
                _ => {}
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
