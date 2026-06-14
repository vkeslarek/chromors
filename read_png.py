from PIL import Image
img = Image.open("tests/fixtures/rgba.png")
pixels = img.load()
for y in range(2):
    for x in range(2):
        print(f"({x}, {y}): {pixels[x, y]}")
