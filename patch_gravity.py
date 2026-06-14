with open("src/operation/geometry.rs", "r") as f:
    code = f.read()

old_gravity = """        cx.param_block(
            crate::backend::gpu::view::ParamBlock::new()
                .param("ox", ox)
                .param("oy", oy)
                .param("in_w", in_spec.width as u32)
                .param("in_h", in_spec.height as u32)
                .param("extend_mode", extend_mode as u32)
                .param("bg_r", bg[0] as f32)
                .param("bg_g", bg[1] as f32)
                .param("bg_b", bg[2] as f32),
        );"""

new_gravity = """        let wu = cx.wu();
        let (out_x, out_y) = match wu {
            poc::work_unit::WorkUnit::Region(r) => (r.x, r.y),
            _ => (0, 0),
        };
        cx.param_block(
            crate::backend::gpu::view::ParamBlock::new()
                .param("ox", ox)
                .param("oy", oy)
                .param("out_x", out_x)
                .param("out_y", out_y)
                .param("in_w", in_spec.width as u32)
                .param("in_h", in_spec.height as u32)
                .param("extend_mode", extend_mode as u32)
                .param("bg_r", bg[0] as f32)
                .param("bg_g", bg[1] as f32)
                .param("bg_b", bg[2] as f32),
        );"""

code = code.replace(old_gravity, new_gravity)

with open("src/operation/geometry.rs", "w") as f:
    f.write(code)
