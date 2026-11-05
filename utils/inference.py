"""
Unified Deep Learning Restoration Engine for PhotoRestore AI.
Combines:
1. RetinaFace Multi-Face Detection to locate all subjects and protect their facial features.
2. Pretrained LaMa (Large Mask Inpainting) for physical damage, scratch, and crack removal
   without ever erasing or blurring facial details.
3. Pretrained GFPGAN facial prior for high-resolution, identity-preserving facial reconstruction
   (eyes, nose, mouth, teeth, hair) so no face remains empty.
4. Deep Learning Colorization (Zhang et al. ECCV 2016) to infer natural, authentic skin tones,
   clothing, and environmental colors for black-and-white photographs.
5. In-house trained Convolutional Autoencoder (CAE) / Variational Autoencoder (VAE) for substrate regularization.
6. Strict dimension, aspect-ratio, and image-quality preservation.
"""

from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch

# NumPy compatibility patch
for _attr, _type in [("bool", bool), ("int", int), ("float", float), ("long", int), ("ulong", int)]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _type)

from config import (
    CAE_CHECKPOINT_PATH,
    VAE_CHECKPOINT_PATH,
    DEVICE,
    CAE_CONFIG,
    VAE_CONFIG,
)
from models.autoencoder import ConvAutoencoder
from models.vae import VariationalAutoencoder
from utils.image_processing import tensor_to_pil_image
from utils.damage_detection import detect_damage_mask
from utils.lama_inpainting import inpaint_damaged_photograph, is_lama_available, LAMA_MODEL_PATH
from utils.face_enhancer import (
    restore_faces_in_image,
    detect_face_boxes,
    is_face_restorer_available,
    GFPGAN_MODEL_PATH,
)
from utils.colorizer import (
    colorize_image,
    is_colorizer_available,
    is_image_monochrome,
    MODEL_PATH as COLOR_MODEL_PATH,
)


def check_checkpoint_status() -> Dict[str, Dict[str, Any]]:
    """Checks whether in-house trained checkpoints and pretrained foundation models exist on disk."""
    cae_exists = CAE_CHECKPOINT_PATH.exists() and CAE_CHECKPOINT_PATH.stat().st_size > 1024
    vae_exists = VAE_CHECKPOINT_PATH.exists() and VAE_CHECKPOINT_PATH.stat().st_size > 1024
    lama_exists = is_lama_available()
    face_exists = is_face_restorer_available()
    color_exists = is_colorizer_available()
    
    return {
        "cae": {
            "exists": cae_exists,
            "path": str(CAE_CHECKPOINT_PATH),
            "size_mb": round(CAE_CHECKPOINT_PATH.stat().st_size / (1024 * 1024), 2) if cae_exists else 0.0,
            "role": "In-house trained Convolutional Autoencoder (substrate & patch regularization)",
        },
        "vae": {
            "exists": vae_exists,
            "path": str(VAE_CHECKPOINT_PATH),
            "size_mb": round(VAE_CHECKPOINT_PATH.stat().st_size / (1024 * 1024), 2) if vae_exists else 0.0,
            "role": "In-house trained Variational Autoencoder (latent distribution regularization)",
        },
        "lama": {
            "exists": lama_exists,
            "path": str(LAMA_MODEL_PATH),
            "size_mb": round(LAMA_MODEL_PATH.stat().st_size / (1024 * 1024), 2) if LAMA_MODEL_PATH.exists() else 0.0,
            "role": "Pretrained LaMa Inpainting (physical scratch, crack, stain, and tear removal)",
        },
        "face_prior": {
            "exists": face_exists,
            "path": str(GFPGAN_MODEL_PATH),
            "size_mb": round(GFPGAN_MODEL_PATH.stat().st_size / (1024 * 1024), 2) if face_exists else 0.0,
            "role": "Pretrained GFPGAN (semantic facial geometry and identity preservation)",
        },
        "colorizer": {
            "exists": color_exists,
            "path": str(COLOR_MODEL_PATH),
            "size_mb": round(COLOR_MODEL_PATH.stat().st_size / (1024 * 1024), 2) if color_exists else 0.0,
            "role": "Pretrained Deep Learning Colorization (Zhang et al. ECCV 2016)",
        },
    }


def load_trained_model(
    model_type: str = "cae",
    device: Optional[torch.device] = None
) -> Tuple[Optional[torch.nn.Module], str]:
    """
    Loads trained CAE or VAE weights from checkpoints directory with strict=True.
    Returns:
      (model, status_message)
    """
    dev = device or DEVICE
    model_type = model_type.lower()
    
    if model_type in ["cae", "autoencoder"]:
        checkpoint_path = CAE_CHECKPOINT_PATH
        if not checkpoint_path.exists():
            return None, f"CAE checkpoint not found at '{checkpoint_path}'."
            
        try:
            model = ConvAutoencoder(
                in_channels=CAE_CONFIG["input_channels"],
                base_channels=CAE_CONFIG["base_channels"]
            )
            checkpoint = torch.load(checkpoint_path, map_location=dev)
            state_dict = (
                checkpoint["model_state_dict"]
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(dev)
            model.eval()
            return model, f"Successfully loaded CAE model onto {dev} with strict=True."
        except Exception as e:
            return None, f"Failed to load CAE checkpoint ({type(e).__name__}: {str(e)})"
            
    elif model_type in ["vae", "variational_autoencoder"]:
        checkpoint_path = VAE_CHECKPOINT_PATH
        if not checkpoint_path.exists():
            return None, f"VAE checkpoint not found at '{checkpoint_path}'."
            
        try:
            model = VariationalAutoencoder(
                in_channels=VAE_CONFIG["input_channels"],
                latent_dim=VAE_CONFIG["latent_dim"],
                base_channels=VAE_CONFIG["base_channels"]
            )
            checkpoint = torch.load(checkpoint_path, map_location=dev)
            state_dict = (
                checkpoint["model_state_dict"]
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                else checkpoint
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(dev)
            model.eval()
            return model, f"Successfully loaded VAE model onto {dev} with strict=True."
        except Exception as e:
            return None, f"Failed to load VAE checkpoint ({type(e).__name__}: {str(e)})"
            
    else:
        return None, f"Unknown model type '{model_type}'."


@torch.no_grad()
def _patch_based_substrate_restore(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> Image.Image:
    """
    Executes in-house neural network inference via overlapping 128x128 patches
    with continuous 2D Hann window blending.
    """
    model.eval()
    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size
    patch_size = 128
    
    if orig_w == patch_size and orig_h == patch_size:
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        inp = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        if isinstance(model, VariationalAutoencoder):
            out, _, _ = model(inp)
        else:
            out = model(inp)
        return tensor_to_pil_image(out)
        
    max_dim = max(orig_w, orig_h)
    stride = 96 if max_dim > 800 else 64
    
    img_arr = np.array(img_rgb, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    
    pad_h = max(0, patch_size - orig_h)
    pad_w = max(0, patch_size - orig_w)
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
        
    _, _, curr_h, curr_w = tensor.shape
    
    hann_1d = np.hanning(patch_size).astype(np.float32)
    window_2d = np.clip(np.outer(hann_1d, hann_1d), 0.05, 1.0)
    win_tensor = torch.from_numpy(window_2d).unsqueeze(0).unsqueeze(0).to(device)
    
    y_steps = list(range(0, curr_h - patch_size + 1, stride))
    if y_steps[-1] != curr_h - patch_size:
        y_steps.append(curr_h - patch_size)
    x_steps = list(range(0, curr_w - patch_size + 1, stride))
    if x_steps[-1] != curr_w - patch_size:
        x_steps.append(curr_w - patch_size)
        
    coords = [(y, x) for y in y_steps for x in x_steps]
    canvas = torch.zeros((1, 3, curr_h, curr_w), dtype=torch.float32, device=device)
    weight = torch.zeros((1, 1, curr_h, curr_w), dtype=torch.float32, device=device)
    
    batch_size = 16 if device.type == "cpu" else 32
    is_vae = isinstance(model, VariationalAutoencoder)
    
    for i in range(0, len(coords), batch_size):
        b_coords = coords[i:i + batch_size]
        patches = torch.cat([tensor[:, :, y:y + patch_size, x:x + patch_size] for y, x in b_coords], dim=0)
        if is_vae:
            restored_patches, _, _ = model(patches)
        else:
            restored_patches = model(patches)
            
        for (y, x), res_p in zip(b_coords, restored_patches):
            canvas[:, :, y:y + patch_size, x:x + patch_size] += res_p.unsqueeze(0) * win_tensor
            weight[:, :, y:y + patch_size, x:x + patch_size] += win_tensor
            
    canvas /= weight
    res_arr = canvas[:, :, :orig_h, :orig_w].squeeze(0).permute(1, 2, 0).cpu().numpy()
    res_arr = np.clip(res_arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(res_arr)


def restore_image(
    model: Optional[torch.nn.Module] = None,
    image: Image.Image = None,
    device: Optional[torch.device] = None,
    restore_original_aspect: bool = True,
    run_damage_removal: bool = True,
    run_face_restoration: bool = True,
    run_colorization: bool = True,
    force_colorization: bool = False,
    run_substrate_cae: bool = False,
    fidelity_weight: float = 0.5,
) -> Tuple[Image.Image, float, int, float, bool]:
    """
    Executes the multi-stage deep learning photograph restoration and colorization pipeline.
    
    Stages:
    1. Face Detection & Protection: Identifies all face coordinates upfront to prevent
       damage masking from erasing eyes, noses, or mouths.
    2. Physical Damage Removal (LaMa Inpainting): Detects and heals background scratches,
       cracks, stains, tears, and dust specks while strictly safeguarding facial regions.
    3. In-House Substrate Regularization (CAE / VAE): Optional neural patch smoothing.
    4. Semantic Face Restoration (GFPGAN): Reconstructs eyes, pupils, eyebrows, nose,
       mouth, teeth, hair, and facial geometry so no face remains empty.
    5. Deep Learning Colorization (Zhang et al.): Infers authentic, realistic skin tones,
       clothing colors, and background scenery for black-and-white photographs.
    6. Quality & Dimension Enforcement:
       - 100% preservation of original photograph width and height.
       
    Returns:
      (restored_image, latency_ms, faces_restored, damaged_pixel_pct, colorized_applied)
    """
    if image is None:
        raise ValueError("Image must be provided for restoration.")

    dev = device or DEVICE
    start_time = time.perf_counter()
    
    img_rgb = image.convert("RGB")
    orig_w, orig_h = img_rgb.size
    
    # Check if photograph is monochrome/grayscale
    is_monochrome = is_image_monochrome(img_rgb)
    
    current_img = img_rgb
    damaged_pixel_pct = 0.0
    faces_restored = 0
    colorized_applied = False

    # STAGE 1: Face Detection Upfront for Mask Protection
    face_boxes: List[np.ndarray] = []
    if run_face_restoration and is_face_restorer_available():
        face_boxes = detect_face_boxes(current_img, device=dev)

    # STAGE 2: Physical Damage & Scratch Removal (LaMa Inpainting)
    if run_damage_removal and is_lama_available():
        damage_mask = detect_damage_mask(
            current_img,
            sensitivity=0.50,
            protect_faces=True,
            face_boxes=face_boxes,
        )
        mask_arr = np.array(damage_mask)
        dmg_pixels = int(np.sum(mask_arr > 0))
        damaged_pixel_pct = round((dmg_pixels / (orig_w * orig_h)) * 100.0, 2)
        
        if dmg_pixels > 0:
            current_img = inpaint_damaged_photograph(current_img, damage_mask, device=dev)

    # STAGE 3: In-House Substrate Regularization (CAE / VAE)
    if run_substrate_cae and model is not None:
        current_img = _patch_based_substrate_restore(model, current_img, dev)

    # STAGE 4: Semantic Face Restoration (GFPGAN Facial Prior)
    if run_face_restoration and is_face_restorer_available():
        current_img, faces_restored = restore_faces_in_image(
            image=current_img,
            fidelity_weight=fidelity_weight,
            device=dev,
            preserve_monochrome=False,
        )

    # STAGE 5: Deep Learning Colorization (Zhang et al. ECCV 2016)
    should_colorize = run_colorization and is_colorizer_available() and (is_monochrome or force_colorization)
    if should_colorize:
        current_img = colorize_image(current_img)
        colorized_applied = True

    # STAGE 6: Dimensional & Quality Enforcement
    if current_img.size != (orig_w, orig_h):
        current_img = current_img.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return current_img, round(elapsed_ms, 2), faces_restored, damaged_pixel_pct, colorized_applied
