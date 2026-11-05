"""
Download Pretrained Foundation Model Weights for PhotoRestore AI.
Downloads:
1. GFPGANv1.4 (Facial Generative Prior) -> models/GFPGANv1.4.pth
2. RetinaFace ResNet50 (Face Detection) -> gfpgan/weights/detection_Resnet50_Final.pth
3. ParseNet (Face Parsing) -> gfpgan/weights/parsing_parsenet.pth
4. Big-LaMa (Fast Fourier Convolution Inpainting) -> models/big-lama.pt
5. Deep Colorization Model (Zhang et al. ECCV 2016) -> models/colorization_release_v2.caffemodel
"""

import os
import urllib.request
import ssl
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

WEIGHTS = [
    {
        "name": "GFPGAN v1.4",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "path": BASE_DIR / "models" / "GFPGANv1.4.pth",
        "min_size_mb": 300,
    },
    {
        "name": "RetinaFace ResNet50",
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth",
        "path": BASE_DIR / "gfpgan" / "weights" / "detection_Resnet50_Final.pth",
        "min_size_mb": 90,
    },
    {
        "name": "ParseNet",
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "path": BASE_DIR / "gfpgan" / "weights" / "parsing_parsenet.pth",
        "min_size_mb": 75,
    },
    {
        "name": "Big-LaMa Inpainting",
        "url": "https://huggingface.co/anyisalin/big-lama/resolve/main/big-lama.pt",
        "path": BASE_DIR / "models" / "big-lama.pt",
        "min_size_mb": 180,
    },
    {
        "name": "Deep Colorization Caffe Model",
        "url": "https://huggingface.co/kaitlynbassford/face-analyzer-weights/resolve/main/colorization_release_v2.caffemodel",
        "path": BASE_DIR / "models" / "colorization_release_v2.caffemodel",
        "min_size_mb": 115,
    },
]


def download_file(item: dict):
    path = item["path"]
    min_size = item["min_size_mb"] * 1024 * 1024
    if path.exists() and path.stat().st_size >= min_size:
        print(f"[OK] {item['name']} already exists ({path.stat().st_size / (1024*1024):.1f} MB): {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    url = item["url"]
    print(f"Downloading {item['name']} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ctx) as response, open(path, "wb") as f:
        total_size = int(response.info().get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 512
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                print(
                    f"\rProgress: {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
    print(f"\n[Done] Saved {path} ({path.stat().st_size / (1024*1024):.1f} MB)")


def main():
    print("=" * 60)
    print("PhotoRestore AI: Pretrained Model Weights Downloader")
    print("=" * 60)
    for item in WEIGHTS:
        download_file(item)
    print("All weights verified.")


if __name__ == "__main__":
    main()

