"""
Face and Photographic Detail Restoration Module for PhotoRestore AI.
Uses GFPGAN / facial prior to recover facial features (eyes, nose, mouth, hair)
from heavily damaged photographs while preserving authentic personal identities.
"""

from typing import Tuple, Optional, List, Sequence
from pathlib import Path
import os
import sys
import numpy as np
from PIL import Image
import torch
import cv2

# NumPy compatibility patch for legacy facexlib/basicsr
for _attr, _type in [("bool", bool), ("int", int), ("float", float), ("long", int), ("ulong", int)]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _type)

# Modern torchvision compatibility patch for basicsr / gfpgan
try:
    import torchvision.transforms.functional as functional_tensor
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
except Exception:
    pass

from config import BASE_DIR, DEVICE

# Pretrained model weight paths
GFPGAN_MODEL_PATH = BASE_DIR / "models" / "GFPGANv1.4.pth"
WEIGHTS_DIR = BASE_DIR / "gfpgan" / "weights"

# Singleton instance for cached restorer
_cached_face_restorer = None


def is_face_restorer_available() -> bool:
    """Checks whether the GFPGAN model weights exist on disk."""
    return GFPGAN_MODEL_PATH.exists() and GFPGAN_MODEL_PATH.stat().st_size > 10 * 1024 * 1024


def get_face_restorer(device: Optional[torch.device] = None):
    """
    Initializes and caches the GFPGANer engine.
    Ensures upscale=1 so original photograph dimensions and aspect ratio
    are strictly preserved without uncontrolled upscaling.
    """
    global _cached_face_restorer
    if _cached_face_restorer is not None:
        return _cached_face_restorer

    if not is_face_restorer_available():
        return None

    try:
        from gfpgan import GFPGANer

        dev = device or DEVICE
        restorer = GFPGANer(
            model_path=str(GFPGAN_MODEL_PATH),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=dev,
        )
        _cached_face_restorer = restorer
        return _cached_face_restorer
    except Exception as e:
        print(f"Warning: Could not initialize GFPGANer ({type(e).__name__}: {e})")
        return None


def detect_face_boxes(image: Image.Image, device: Optional[torch.device] = None) -> List[np.ndarray]:
    """
    Detects all face bounding boxes in an image using RetinaFace from GFPGAN face_helper.
    Returns list of bounding boxes [x1, y1, x2, y2, confidence].
    """
    restorer = get_face_restorer(device=device)
    if restorer is None:
        return []

    try:
        img_rgb = image.convert("RGB")
        bgr_arr = np.array(img_rgb)[:, :, ::-1].copy()
        restorer.face_helper.clean_all()
        restorer.face_helper.read_image(bgr_arr)
        restorer.face_helper.get_face_landmarks_5(only_center_face=False)
        return list(restorer.face_helper.det_faces)
    except Exception as e:
        print(f"Warning: Face detection failed ({type(e).__name__}: {e})")
        return []


def restore_faces_in_image(
    image: Image.Image,
    fidelity_weight: float = 0.5,
    device: Optional[torch.device] = None,
    preserve_monochrome: bool = False,
) -> Tuple[Image.Image, int]:
    """
    Restores all faces present in the photograph using deep facial priors.
    Reconstructs complete facial features (eyes, pupils, nose, mouth, hair, teeth).
    
    Parameters:
      image: Input PIL Image (damaged or pre-cleaned photograph)
      fidelity_weight: Balance between restored facial features (0.0) and original identity (1.0).
                       0.5 generates sharp facial structures while maintaining identity.
      device: Torch computation device.
      preserve_monochrome: Whether to strictly convert output back to monochrome.
      
    Returns:
      (restored_pil_image, face_count): Restored image at exact same dimensions and number of restored faces.
    """
    restorer = get_face_restorer(device=device)
    if restorer is None:
        return image, 0

    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size
    
    # GFPGAN works with BGR numpy array (OpenCV format)
    bgr_arr = np.array(img_rgb)[:, :, ::-1].copy()

    try:
        # Enhance all faces in the photo, paste back onto original canvas
        cropped_faces, restored_faces, enhanced_bgr = restorer.enhance(
            bgr_arr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=fidelity_weight,
        )
        
        face_count = len(restored_faces) if restored_faces is not None else 0
        
        if enhanced_bgr is not None:
            enhanced_rgb = enhanced_bgr[:, :, ::-1]
            if enhanced_rgb.shape[1] != orig_w or enhanced_rgb.shape[0] != orig_h:
                enhanced_pil = Image.fromarray(enhanced_rgb).resize((orig_w, orig_h), Image.Resampling.LANCZOS)
            else:
                enhanced_pil = Image.fromarray(enhanced_rgb)
                
            if preserve_monochrome:
                enhanced_pil = enhanced_pil.convert("L").convert("RGB")
                
            return enhanced_pil, face_count
        else:
            return image, 0
            
    except Exception as e:
        print(f"Warning: Face restoration failed ({type(e).__name__}: {e}). Returning input.")
        return image, 0
