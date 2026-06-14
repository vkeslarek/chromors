with open("src/operation/geometry.rs", "r") as f:
    code = f.read()

code = code.replace("""        cx.param_block(
            crate::backend::gpu::view::ParamBlock::new()
                .param("ox", self.x)
                .param("oy", self.y)
                .param("in_w", in_spec.width as u32)
                .param("in_h", in_spec.height as u32)
                .param("extend_mode", extend_mode as u32)
                .param("bg_r", bg[0] as f32)
                .param("bg_g", bg[1] as f32)
                .param("bg_b", bg[2] as f32),
        );""", """        let wu = cx.wu();
        let (out_x, out_y) = match wu {
            poc::work_unit::WorkUnit::Region(r) => (r.x, r.y),
            _ => (0, 0),
        };
        cx.param_block(
            crate::backend::gpu::view::ParamBlock::new()
                .param("ox", self.x)
                .param("oy", self.y)
                .param("out_x", out_x)
                .param("out_y", out_y)
                .param("in_w", in_spec.width as u32)
                .param("in_h", in_spec.height as u32)
                .param("extend_mode", extend_mode as u32)
                .param("bg_r", bg[0] as f32)
                .param("bg_g", bg[1] as f32)
                .param("bg_b", bg[2] as f32),
        );""")

with open("src/operation/geometry.rs", "w") as f:
    f.write(code)

with open("shaders/ops/geometry.slang", "r") as f:
    code = f.read()

code = code.replace("""public void embed_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, int ox, int oy, uint in_w, uint in_h, uint extend_mode, float bg_r, float bg_g, float bg_b) {
    int sx = int(idx.x) - ox;
    int sy = int(idx.y) - oy;""", """public void embed_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, int ox, int oy, int out_x, int out_y, uint in_w, uint in_h, uint extend_mode, float bg_r, float bg_g, float bg_b) {
    int sx = out_x + int(idx.x) - ox;
    int sy = out_y + int(idx.y) - oy;""")

code = code.replace("input.read(uint2(sx, sy))", "input.read(idx)")
code = code.replace("input.read(uint2(wx, wy))", "input.read(uint2(wx - out_x + ox, wy - out_y + oy))")
code = code.replace("input.read(uint2(cx, cy))", "input.read(uint2(cx - out_x + ox, cy - out_y + oy))")

with open("shaders/ops/geometry.slang", "w") as f:
    f.write(code)
