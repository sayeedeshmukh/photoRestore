"""
Deep Learning Photographic Colorization Module for PhotoRestore AI.
Uses Zhang et al. (ECCV 2016) Deep Convolutional Colorization network via OpenCV DNN
to infer authentic, realistic chrominance (skin tones, hair, clothing, environment)
for black-and-white and faded vintage photographs.
"""

from typing import Optional
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

from config import BASE_DIR

MODELS_DIR = BASE_DIR / "models"
PROTOTXT_PATH = MODELS_DIR / "colorization_deploy_v2.prototxt"
MODEL_PATH = MODELS_DIR / "colorization_release_v2.caffemodel"
PTS_PATH = MODELS_DIR / "pts_in_hull.npy"

_cached_colorizer_net = None


def is_colorizer_available() -> bool:
    """Verifies that all three colorization model assets exist on disk."""
    return (
        PROTOTXT_PATH.exists()
        and MODEL_PATH.exists()
        and PTS_PATH.exists()
        and MODEL_PATH.stat().st_size > 10 * 1024 * 1024
    )


def get_colorizer_net() -> Optional[cv2.dnn.Net]:
    """Initializes and caches the OpenCV DNN colorization network."""
    global _cached_colorizer_net
    if _cached_colorizer_net is not None:
        return _cached_colorizer_net

    if not is_colorizer_available():
        return None

    try:
        net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_PATH), str(MODEL_PATH))
        pts = np.load(str(PTS_PATH))

        # Populate cluster centers in color decoder layer
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype(np.float32)]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

        _cached_colorizer_net = net
        return _cached_colorizer_net
    except Exception as e:
        print(f"Warning: Failed to load colorization network ({type(e).__name__}: {e})")
        return None


def is_image_monochrome(image: Image.Image, threshold: float = 4.0) -> bool:
    """
    Detects whether an image is monochrome / grayscale or faded sepia.
    Computes mean absolute difference between R, G, and B channels.
    """
    rgb = image.convert("RGB")
    arr = np.array(rgb, dtype=np.float32)
    diff_rg = np.mean(np.abs(arr[:, :, 0] - arr[:, :, 1]))
    diff_gb = np.mean(np.abs(arr[:, :, 1] - arr[:, :, 2]))
    return bool((diff_rg < threshold) and (diff_gb < threshold))


def colorize_image(image: Image.Image) -> Image.Image:
    """
    Applies Deep Learning Colorization to an image.
    
    1. Converts image to CIE Lab color space.
    2. Feeds normalized L luminance channel to deep CNN (224x224 input).
    3. Predicts 'ab' chrominance channels across 313 quantized color bins.
    4. Upsamples predicted 'ab' chrominance to match original image dimensions.
    5. Combines with original uncompressed L luminance to preserve 100% of photographic sharpness.
    6. Converts back to RGB color space.
    """
    net = get_colorizer_net()
    if net is None:
        return image

    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size

    rgb_arr = np.array(img_rgb)
    scaled = rgb_arr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2Lab)

    # 224x224 normalized L input for CNN forward pass
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50.0  # Empirical ImageNet luminance centering

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Bilinear upsample ab channels to match exact original dimensions
    ab = cv2.resize(ab, (orig_w, orig_h))

    # Keep original uncompressed L channel for pristine resolution and sharpness
    L_orig = cv2.split(lab)[0]
    colorized_lab = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)

    colorized_rgb = cv2.cvtColor(colorized_lab, cv2.COLOR_Lab2RGB)
    colorized_rgb = np.clip(colorized_rgb, 0.0, 1.0)
    colorized_arr = (colorized_rgb * 255.0).astype(np.uint8)

    return Image.fromarray(colorized_arr)

