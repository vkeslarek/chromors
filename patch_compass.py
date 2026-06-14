with open("shaders/ops/convolution.slang", "r") as f:
    code = f.read()

new_func = """uint2 compass_rotate90(uint mx, uint my, uint mh) {
    return uint2(my, mh - 1u - mx);
}

"""

code = code.replace("public void compass_kernel", new_func + "public void compass_kernel")

with open("shaders/ops/convolution.slang", "w") as f:
    f.write(code)
