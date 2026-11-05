"""
Neural Inpainting Module for PhotoRestore AI using LaMa (Large Mask Inpainting).
Removes physical scratches, cracks, tears, dust specks, and stains from old photographs
using Fast Fourier Convolutions to realistically synthesize damaged substrate pixels.
"""

from typing import Optional, Tuple
from pathlib import Path
import os
import sys
import numpy as np
from PIL import Image
import torch

# NumPy compatibility patch
if not hasattr(np, "long"):
    np.long = int

from config import BASE_DIR, DEVICE

# Checkpoint paths
LAMA_MODEL_PATH = BASE_DIR / "models" / "big-lama.pt"
CACHE_LAMA_PATH = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "big-lama.pt"

_cached_lama_model = None


def is_lama_available() -> bool:
    """Checks whether the LaMa pretrained model weights exist."""
    return (
        (LAMA_MODEL_PATH.exists() and LAMA_MODEL_PATH.stat().st_size > 10 * 1024 * 1024)
        or (CACHE_LAMA_PATH.exists() and CACHE_LAMA_PATH.stat().st_size > 10 * 1024 * 1024)
    )


def get_lama_model(device: Optional[torch.device] = None):
    """Initializes and caches the SimpleLama inpainting model."""
    global _cached_lama_model
    if _cached_lama_model is not None:
        return _cached_lama_model

    try:
        from simple_lama_inpainting import SimpleLama
        dev = device or DEVICE
        model = SimpleLama(device=dev)
        _cached_lama_model = model
        return _cached_lama_model
    except Exception as e:
        print(f"Warning: Could not initialize LaMa inpainting model ({type(e).__name__}: {e})")
        return None


def inpaint_damaged_photograph(
    image: Image.Image,
    mask: Image.Image,
    device: Optional[torch.device] = None,
) -> Image.Image:
    """
    Applies LaMa neural inpainting to repair all damaged pixels indicated by the mask.
    
    Guarantees:
    - 100% preservation of original photograph width and height.
    - Preserves native aspect ratio.
    - Preserves monochrome/grayscale integrity without introducing blue or cyan color tints.
    - Seamlessly fills missing pixels with authentic image textures.
    
    Parameters:
      image: Original damaged PIL Image.
      mask: Binary damage mask (PIL Image mode 'L', 255 = damage, 0 = intact).
      device: Computation device.
      
    Returns:
      Inpainted PIL Image.
    """
    lama = get_lama_model(device=device)
    if lama is None:
        print("LaMa model unavailable. Returning original image.")
        return image

    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size
    
    # Detect if image is grayscale/monochrome
    rgb_arr = np.array(img_rgb).astype(np.float32)
    diff_rg = np.mean(np.abs(rgb_arr[:, :, 0] - rgb_arr[:, :, 1]))
    diff_gb = np.mean(np.abs(rgb_arr[:, :, 1] - rgb_arr[:, :, 2]))
    is_monochrome = (diff_rg < 3.0) and (diff_gb < 3.0)

    # Ensure mask is L mode and matches exact dimensions
    mask_l = mask.convert("L")
    if mask_l.size != (orig_w, orig_h):
        mask_l = mask_l.resize((orig_w, orig_h), Image.Resampling.NEAREST)

    # Check if mask has any positive pixels
    mask_arr = np.array(mask_l)
    if not np.any(mask_arr > 0):
        # No damage detected, return original
        return image

    try:
        inpainted_pil = lama(img_rgb, mask_l)
        
        # Verify and guarantee exact original dimensions
        if inpainted_pil.size != (orig_w, orig_h):
            inpainted_pil = inpainted_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            
        # Prevent unwanted blue/cyan or color casts on black-and-white photographs
        if is_monochrome:
            inpainted_gray = inpainted_pil.convert("L")
            inpainted_pil = inpainted_gray.convert("RGB")
            
        return inpainted_pil
    except Exception as e:
        print(f"Inpainting error ({type(e).__name__}: {e}). Returning original image.")
        return image
