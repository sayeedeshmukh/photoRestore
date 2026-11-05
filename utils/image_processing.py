"""
Image processing utilities for PhotoRestore AI.
Handles loading, validation, tensor conversions, postprocessing, and side-by-side visualizations.
"""

from typing import Tuple, Optional, Union
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch

from config import IMAGE_SIZE


# Supported image extensions and size constraints
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 15.0


def validate_image_file(file_obj_or_path: Union[io.BytesIO, bytes, str, Path]) -> Tuple[bool, str]:
    """
    Validates uploaded photograph for valid format, integrity, and reasonable file size.
    Returns:
      (is_valid: bool, error_message: str)
    """
    try:
        if isinstance(file_obj_or_path, (str, Path)):
            p = Path(file_obj_or_path)
            if not p.exists():
                return False, f"File does not exist: {p}"
            if p.suffix.lower() not in ALLOWED_EXTENSIONS:
                return False, f"Unsupported format '{p.suffix}'. Only JPG, JPEG, and PNG are supported."
            size_mb = p.stat().st_size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                return False, f"File size ({size_mb:.1f} MB) exceeds maximum limit of {MAX_FILE_SIZE_MB} MB."
            img = Image.open(p)
        else:
            # Bytes or BytesIO
            if isinstance(file_obj_or_path, bytes):
                buf = io.BytesIO(file_obj_or_path)
                size_mb = len(file_obj_or_path) / (1024 * 1024)
            else:
                file_obj_or_path.seek(0, io.SEEK_END)
                size_mb = file_obj_or_path.tell() / (1024 * 1024)
                file_obj_or_path.seek(0)
                buf = file_obj_or_path
                
            if size_mb > MAX_FILE_SIZE_MB:
                return False, f"File size ({size_mb:.1f} MB) exceeds maximum limit of {MAX_FILE_SIZE_MB} MB."
                
            img = Image.open(buf)

        # Verify image can be parsed
        img.verify()
        
        # Reopen to check dimensions (verify() invalidates image object)
        if isinstance(file_obj_or_path, (str, Path)):
            img = Image.open(file_obj_or_path)
        else:
            buf.seek(0)
            img = Image.open(buf)
            
        w, h = img.size
        if w < 16 or h < 16:
            return False, f"Image dimensions too small ({w}x{h}). Minimum supported is 16x16."
        if w > 8192 or h > 8192:
            return False, f"Image dimensions excessively large ({w}x{h})."
            
        return True, ""
    except Exception as e:
        return False, f"Corrupted or invalid image file ({type(e).__name__}: {str(e)})"


def load_image_to_pil(file_obj_or_path: Union[io.BytesIO, bytes, str, Path]) -> Image.Image:
    """Loads image and converts strictly to RGB space (stripping RGBA alpha or grayscale)."""
    if isinstance(file_obj_or_path, (str, Path)):
        img = Image.open(file_obj_or_path)
    elif isinstance(file_obj_or_path, bytes):
        img = Image.open(io.BytesIO(file_obj_or_path))
    else:
        file_obj_or_path.seek(0)
        img = Image.open(file_obj_or_path)
        
    # Auto-orient based on EXIF tag if present
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def preprocess_image_for_model(
    image: Image.Image,
    target_size: Tuple[int, int] = IMAGE_SIZE
) -> torch.Tensor:
    """
    Prepares a PIL Image for model inference:
    - Resizes to target_size (default 128x128) using Bilinear interpolation.
    - Converts to normalized torch.FloatTensor in range [0.0, 1.0].
    - Adds batch dimension: shape (1, 3, H, W).
    """
    img_rgb = image.convert("RGB")
    if img_rgb.size != target_size:
        img_rgb = img_rgb.resize(target_size, Image.Resampling.BILINEAR)
        
    arr = np.array(img_rgb, dtype=np.float32) / 255.0
    # Convert HWC to CHW
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.contiguous()


def tensor_to_pil_image(
    tensor: torch.Tensor,
    restore_original_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Converts a model output tensor (1, 3, H, W) or (3, H, W) in [0, 1] back to a PIL Image.
    Optionally upsamples to restore_original_size (e.g., user upload size).
    """
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t.squeeze(0)  # Remove batch dim -> (3, H, W)
        
    t = torch.clamp(t, 0.0, 1.0)
    arr = t.permute(1, 2, 0).numpy()  # (H, W, 3)
    arr = (arr * 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    
    if restore_original_size is not None and restore_original_size != img.size:
        img = img.resize(restore_original_size, Image.Resampling.BICUBIC)
        
    return img


def create_side_by_side_comparison(
    image_left: Image.Image,
    image_right: Image.Image,
    label_left: str = "Damaged",
    label_right: str = "Restored"
) -> Image.Image:
    """Combines two images horizontally with equal heights for side-by-side presentation."""
    target_h = max(image_left.height, image_right.height)
    # Maintain aspect ratios
    w_l = int(image_left.width * (target_h / image_left.height))
    w_r = int(image_right.width * (target_h / image_right.height))
    
    left_resized = image_left.resize((w_l, target_h), Image.Resampling.BICUBIC)
    right_resized = image_right.resize((w_r, target_h), Image.Resampling.BICUBIC)
    
    combined = Image.new("RGB", (w_l + w_r + 10, target_h), (240, 242, 246))
    combined.paste(left_resized, (0, 0))
    combined.paste(right_resized, (w_l + 10, 0))
    return combined


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Converts a PIL Image into bytes for Streamlit download buttons."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()
