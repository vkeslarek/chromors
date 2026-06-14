import re

with open("src/backend/gpu/view.rs", "r") as f:
    code = f.read()

code = code.replace(
"""pub struct RegionParams {
    pub stride: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}""",
"""pub struct RegionParams {
    pub stride: u32,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub pad_x: i32,
    pub pad_y: i32,
}"""
)

code = code.replace(
"""    pub fn tight(w: u32, h: u32) -> Self {
        Self {
            stride: w,
            x: 0,
            y: 0,
            w,
            h,
        }
    }""",
"""    pub fn tight(w: u32, h: u32) -> Self {
        Self {
            stride: w,
            x: 0,
            y: 0,
            w,
            h,
            pad_x: 0,
            pad_y: 0,
        }
    }"""
)

code = code.replace(
"""    pub fn padded(stride: u32, x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { stride, x, y, w, h }
    }""",
"""    pub fn padded(stride: u32, x: u32, y: u32, w: u32, h: u32, pad_x: i32, pad_y: i32) -> Self {
        Self { stride, x, y, w, h, pad_x, pad_y }
    }"""
)

code = code.replace(
"""        block.fields.push((name.to_string(), "BufferRegion"));
        block.field_sizes.push(20);
        for v in [self.stride, self.x, self.y, self.w, self.h] {
            block.bytes.extend_from_slice(&v.to_le_bytes());
        }""",
"""        block.fields.push((name.to_string(), "BufferRegion"));
        block.field_sizes.push(28);
        for v in [self.stride, self.x, self.y, self.w, self.h] {
            block.bytes.extend_from_slice(&v.to_le_bytes());
        }
        block.bytes.extend_from_slice(&self.pad_x.to_le_bytes());
        block.bytes.extend_from_slice(&self.pad_y.to_le_bytes());"""
)

with open("src/backend/gpu/view.rs", "w") as f:
    f.write(code)
