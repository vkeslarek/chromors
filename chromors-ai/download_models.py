import os
import urllib.request
import sys

# All ONNX models used by chromors-ai.
# URLs point to HuggingFace hosted ONNX files.
MODELS = {
    "sam2": {
        "sam2_hiera_tiny.encoder.onnx": "https://huggingface.co/chromors/sam2-onnx/resolve/main/sam2_hiera_tiny.encoder.onnx",
        "sam2_hiera_tiny.decoder.onnx": "https://huggingface.co/chromors/sam2-onnx/resolve/main/sam2_hiera_tiny.decoder.onnx",
    },
    "sam3": {
        "sam3_image_encoder.onnx": "https://huggingface.co/chromors/sam3-onnx/resolve/main/sam3_image_encoder.onnx",
        "sam3_decoder.onnx": "https://huggingface.co/chromors/sam3-onnx/resolve/main/sam3_decoder.onnx",
    },
    "cascadepsp": {
        "cascadepsp_base.onnx": "https://huggingface.co/chromors/cascadepsp-onnx/resolve/main/cascadepsp_base.onnx",
    },
    "modnet": {
        "modnet_photographic.onnx": "https://huggingface.co/Xenova/modnet/resolve/main/onnx/model.onnx",
    },
    "vitmatte": {
        "vitmatte_small.onnx": "https://huggingface.co/Xenova/vitmatte-base-composition-1k/resolve/main/onnx/model.onnx",
    },
    "realesrgan": {
        "realesrgan_x4plus.onnx": "https://huggingface.co/AXERA-TECH/Real-ESRGAN/resolve/main/onnx/realesrgan-x4.onnx",
    },
    "swinir": {
        "swinir_denoise_color_15.onnx": "https://huggingface.co/wuminghao/swinir/resolve/main/swin-ir-noise.onnx",
    },
    "depth_anything": {
        "depth_anything_v2_small.onnx": "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
    },
    "lama": {
        "lama_fp32.onnx": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
}

def download_file(url, dest_path):
    print(f"  ↓ {os.path.basename(dest_path)}")
    print(f"    from {url}")
    try:
        urllib.request.urlretrieve(url, dest_path)
        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"    ✓ {size_mb:.1f} MB")
    except Exception as e:
        print(f"    ✗ Failed: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)

def main():
    if len(sys.argv) == 1:
        active_features = set(MODELS.keys())
    elif sys.argv[1] == "none":
        active_features = set()
    elif sys.argv[1] == "--list":
        print("Available model categories:")
        for cat, files in MODELS.items():
            total = len(files)
            print(f"  {cat}: {total} file(s)")
            for f in files:
                print(f"    - {f}")
        return
    else:
        active_features = set(sys.argv[1:])

    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(base_dir, exist_ok=True)

    downloaded = 0
    skipped = 0

    for category, files in MODELS.items():
        if category not in active_features:
            continue

        cat_dir = os.path.join(base_dir, category)
        os.makedirs(cat_dir, exist_ok=True)

        print(f"\n[{category}]")
        for filename, url in files.items():
            file_path = os.path.join(cat_dir, filename)
            if url == "MANUAL_EXPORT_REQUIRED":
                print(f"  ⚠ {filename} requires manual PyTorch→ONNX export")
                skipped += 1
            elif os.path.exists(file_path):
                print(f"  ✓ {filename} (cached)")
                skipped += 1
            else:
                download_file(url, file_path)
                downloaded += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} cached.")

if __name__ == "__main__":
    main()
