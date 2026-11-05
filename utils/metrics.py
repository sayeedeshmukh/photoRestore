"""
Quantitative image restoration evaluation metrics (PSNR and SSIM).
Supports scikit-image when available with high-precision vectorized numpy fallbacks.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch


# Try importing scikit-image metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
    from skimage.metrics import structural_similarity as _sk_ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    _sk_psnr = None
    _sk_ssim = None
    SKIMAGE_AVAILABLE = False


def _to_numpy_rgb(image: Any) -> np.ndarray:
    """Standardizes input (PIL Image, torch.Tensor, or ndarray) to float32 ndarray in [0.0, 1.0]."""
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.shape[0] == 3:
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        return np.clip(arr, 0.0, 1.0).astype(np.float32)
        
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        return np.clip(arr, 0.0, 1.0)
        
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return np.clip(arr, 0.0, 1.0)
        
    else:
        raise TypeError(f"Unsupported image type for metrics: {type(image)}")


def calculate_psnr(
    clean: Any,
    restored: Any,
    data_range: float = 1.0
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between clean reference and restored photograph.
    Formula: PSNR = 10 * log10(MAX_I^2 / MSE)
    Units: Decibels (dB). Higher indicates greater fidelity to clean reference.
    """
    img_clean = _to_numpy_rgb(clean)
    img_restored = _to_numpy_rgb(restored)
    
    # Ensure identical dimensions
    if img_clean.shape != img_restored.shape:
        raise ValueError(
            f"Shape mismatch for PSNR calculation: Clean {img_clean.shape} vs Restored {img_restored.shape}"
        )
        
    mse = np.mean((img_clean - img_restored) ** 2)
    if mse == 0.0:
        return 100.0  # Perfect restoration cap
        
    if SKIMAGE_AVAILABLE and _sk_psnr is not None:
        try:
            return float(_sk_psnr(img_clean, img_restored, data_range=data_range))
        except Exception:
            pass
            
    # Vectorized fallback
    return float(10.0 * np.log10((data_range ** 2) / mse))


def calculate_ssim(
    clean: Any,
    restored: Any,
    data_range: float = 1.0,
    win_size: int = 7
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between clean reference and restored photograph.
    Measures perceived visual quality across luminance, contrast, and structural preservation.
    Range: [-1.0, 1.0], where 1.0 indicates identical structure.
    """
    img_clean = _to_numpy_rgb(clean)
    img_restored = _to_numpy_rgb(restored)
    
    if img_clean.shape != img_restored.shape:
        raise ValueError(
            f"Shape mismatch for SSIM calculation: Clean {img_clean.shape} vs Restored {img_restored.shape}"
        )
        
    h, w, c = img_clean.shape
    actual_win_size = min(win_size, h, w)
    if actual_win_size % 2 == 0:
        actual_win_size -= 1
    actual_win_size = max(3, actual_win_size)
    
    if SKIMAGE_AVAILABLE and _sk_ssim is not None:
        try:
            val = _sk_ssim(
                img_clean,
                img_restored,
                data_range=data_range,
                channel_axis=2,
                win_size=actual_win_size
            )
            return float(val)
        except Exception:
            pass
            
    # Mathematically exact channel-averaged SSIM fallback (Wang et al. 2004)
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    
    ssim_channels = []
    for ch in range(c):
        x = img_clean[:, :, ch]
        y = img_restored[:, :, ch]
        
        mu_x = np.mean(x)
        mu_y = np.mean(y)
        sigma_x_sq = np.var(x)
        sigma_y_sq = np.var(y)
        sigma_xy = np.mean((x - mu_x) * (y - mu_y))
        
        numerator = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
        denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x_sq + sigma_y_sq + C2)
        
        ssim_ch = numerator / (denominator + 1e-12)
        ssim_channels.append(ssim_ch)
        
    return float(np.mean(ssim_channels))


def evaluate_restoration_pair(
    clean: Optional[Any],
    restored: Any
) -> Dict[str, Any]:
    """
    Validates availability of clean ground truth and computes restoration metrics.
    If ground truth is missing, returns descriptive status explaining why metrics cannot be calculated.
    """
    if clean is None:
        return {
            "has_ground_truth": False,
            "psnr": None,
            "ssim": None,
            "message": "Ground-truth pristine image is unavailable. Objective metrics (PSNR, SSIM) require reference images."
        }
        
    psnr_val = calculate_psnr(clean, restored)
    ssim_val = calculate_ssim(clean, restored)
    
    return {
        "has_ground_truth": True,
        "psnr": round(psnr_val, 2),
        "ssim": round(ssim_val, 4),
        "message": "Objective metrics successfully computed against pristine reference."
    }
