"""
Automatic photographic damage and scratch detection module for PhotoRestore AI.
Generates binary defect masks identifying scratches, cracks, tears, dust specks,
and physical paper abrasions for neural inpainting while strictly protecting
facial features (eyes, nose, mouth) from being falsely masked or erased.
"""

from typing import Tuple, Optional, List, Sequence, Union
import numpy as np
from PIL import Image
import cv2


def detect_damage_mask(
    image: Image.Image,
    sensitivity: float = 0.5,
    min_scratch_area: int = 4,
    dilate_radius: int = 2,
    protect_faces: bool = True,
    face_boxes: Optional[Sequence[Union[Sequence[float], np.ndarray]]] = None,
) -> Image.Image:
    """
    Detects physical damage (scratches, cracks, dust specks, stains, tears) in old photographs.
    
    Uses multi-scale morphological top-hat and black-hat operators, multi-directional
    linear kernel filtering, and localized contrast anomaly analysis.
    
    Parameters:
      image: Input PIL Image
      sensitivity: Sensitivity factor in [0.1, 1.0]. Default 0.5 balances detecting fine
                   cracks while rejecting photographic grain.
      min_scratch_area: Minimum connected pixel count to qualify as damage.
      dilate_radius: Dilation radius to ensure mask covers boundary penumbra of scratches.
      protect_faces: Whether to avoid masking primary facial regions.
      face_boxes: Optional sequence of face bounding boxes [x1, y1, x2, y2] to strictly
                  protect from inpainting.
      
    Returns:
      Binary damage mask as PIL Image in mode 'L' (255 = damage, 0 = intact photo).
    """
    rgb_arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    # 1. Multi-scale Top-Hat (bright scratches, cracks, white flaking)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    
    tophat_small = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_small)
    tophat_med = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_medium)
    tophat_cross = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel_cross)

    # 2. Multi-scale Black-Hat (dark cracks, dirt, creases, stains)
    blackhat_small = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_small)
    blackhat_med = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_medium)
    blackhat_cross = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_cross)

    # 3. Multi-directional linear kernels to capture elongated scratches
    line_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    line_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    tophat_line_h = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, line_h)
    tophat_line_v = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, line_v)
    blackhat_line_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, line_h)
    blackhat_line_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, line_v)

    # 4. Combine morphological responses
    bright_defects = np.maximum.reduce([tophat_small, tophat_med, tophat_cross, tophat_line_h, tophat_line_v])
    dark_defects = np.maximum.reduce([blackhat_small, blackhat_med, blackhat_cross, blackhat_line_h, blackhat_line_v])
    
    combined_response = np.maximum(bright_defects, dark_defects).astype(np.float32)

    # 5. Local variance contrast normalization to reduce false positives in high-texture areas
    blurred = cv2.GaussianBlur(gray, (9, 9), 2.0)
    high_freq_contrast = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))

    # Weight response by high-frequency deviation
    defect_signal = (combined_response * 0.7) + (high_freq_contrast * 0.3)

    # 6. Adaptive Thresholding based on image statistics and sensitivity
    mean_val = np.mean(defect_signal)
    std_val = np.std(defect_signal)
    thresh_multiplier = 3.5 - (sensitivity * 2.2)
    thresh_val = mean_val + thresh_multiplier * std_val
    thresh_val = max(18.0, thresh_val)

    binary_mask = (defect_signal > thresh_val).astype(np.uint8) * 255

    # 7. Filter out isolated single-pixel noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    filtered_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_scratch_area:
            filtered_mask[labels == i] = 255

    # 8. Dilate defect mask slightly to encompass boundary penumbra of cracks
    if dilate_radius > 0:
        kernel_dilate = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_radius * 2 + 1, dilate_radius * 2 + 1)
        )
        filtered_mask = cv2.dilate(filtered_mask, kernel_dilate, iterations=1)

    # 9. Face Protection: strictly zero out all facial regions so inpainting NEVER erases facial features
    if protect_faces:
        active_boxes = []
        if face_boxes is not None and len(face_boxes) > 0:
            active_boxes = [box[:4] for box in face_boxes]
        else:
            # Fallback face detection via OpenCV Haar cascade if no boxes were passed
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
                active_boxes = [(x, y, x + fw, y + fh) for (x, y, fw, fh) in detected]
            except Exception:
                active_boxes = []

        for box in active_boxes:
            x1, y1, x2, y2 = box[:4]
            # Safety margin of 10%
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            bx1 = max(0, int(x1 - pad_x))
            by1 = max(0, int(y1 - pad_y))
            bx2 = min(w, int(x2 + pad_x))
            by2 = min(h, int(y2 + pad_y))
            filtered_mask[by1:by2, bx1:bx2] = 0

    return Image.fromarray(filtered_mask, mode="L")
