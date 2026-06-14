with open("shaders/ops/geometry.slang", "r") as f:
    code = f.read()

bad_grid = """public void grid_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, uint in_w, uint tile_height, uint across) {
    uint col = idx.x / in_w;
    uint row = idx.y / tile_height;
    uint strip = row * across + col;

    uint sx = idx.x % in_w;
    uint sy = strip * tile_height + (idx.y % tile_height);

    output.write(idx, input.read(idx));
}"""

good_grid = """public void grid_kernel<R: IRegion>(uint2 idx, R input, RWRegion output, int out_x, int out_y, uint in_w, uint tile_height, uint across) {
    uint x = out_x + idx.x;
    uint y = out_y + idx.y;
    uint col = x / in_w;
    uint row = y / tile_height;
    uint strip = row * across + col;

    uint sx = x % in_w;
    uint sy = strip * tile_height + (y % tile_height);

    output.write(idx, input.read(uint2(sx, sy)));
}"""

code = code.replace(bad_grid, good_grid)

with open("shaders/ops/geometry.slang", "w") as f:
    f.write(code)

with open("src/operation/geometry.rs", "r") as f:
    code = f.read()

bad_rs = """    fn lower(&self, cx: &mut GpuBuilder) {
        let in_spec = &*self.input.spec;
        cx.param_block(
            ParamBlock::new()
                .param("in_w", in_spec.width as u32)
                .param("tile_height", self.tile_height as u32)
                .param("across", self.across as u32),
        );"""

good_rs = """    fn lower(&self, cx: &mut GpuBuilder) {
        let in_spec = &*self.input.spec;
        let wu = cx.wu();
        let (out_x, out_y) = match wu {
            poc::work_unit::WorkUnit::Region(r) => (r.x, r.y),
            _ => (0, 0),
        };
        cx.param_block(
            ParamBlock::new()
                .param("out_x", out_x)
                .param("out_y", out_y)
                .param("in_w", in_spec.width as u32)
                .param("tile_height", self.tile_height as u32)
                .param("across", self.across as u32),
        );"""

code = code.replace(bad_rs, good_rs)

with open("src/operation/geometry.rs", "w") as f:
    f.write(code)
