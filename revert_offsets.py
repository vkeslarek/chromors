import re

with open("shaders/ops/convolution.slang", "r") as f:
    code = f.read()

code = code.replace("int halo = max(hw, hh);\n\n    float4 sum_a", "float4 sum_a")
code = code.replace("int2 src_pos = int2(idx) + int2(halo, halo) + int2(int(mx) - hw, int(my) - hh);", "int2 src_pos = int2(idx) + int2(int(mx) - hw, int(my) - hh);")
code = code.replace("int halo = max(hw, hh);\n    float4 sum = float4(0.0", "float4 sum = float4(0.0")
code = code.replace("int halo = max(hw, hh);\n    bool erode", "bool erode")

with open("shaders/ops/convolution.slang", "w") as f:
    f.write(code)

with open("shaders/ops/filters.slang", "r") as f:
    code = f.read()

code = code.replace("float4 orig = input.read_clamped(int2(idx) + int2(r, r));", "float4 orig = input.read_clamped(int2(idx));")
code = code.replace("acc += w * input.read_clamped(int2(idx) + int2(r, r) + int2(x, y));", "acc += w * input.read_clamped(int2(idx) + int2(x, y));")
code = code.replace("output.write(idx, input.read_clamped(int2(idx) + int2(radius, radius)));", "output.write(idx, input.read_clamped(int2(idx)));")
code = code.replace("acc += w * input.read_clamped(int2(idx) + int2(radius, radius) + int2(x, y));", "acc += w * input.read_clamped(int2(idx) + int2(x, y));")

code = code.replace("int2 src_pos = int2(idx) + int2(mx, my);", "int2 src_pos = int2(idx) + int2(int(mx) - half, int(my) - half);")
code = code.replace("float4 src = input.read_clamped(src_pos);", "float4 src = input.read(uint2(src_pos));")

with open("shaders/ops/filters.slang", "w") as f:
    f.write(code)
