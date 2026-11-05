"""
Unit tests for image processing utilities and quantitative evaluation metrics.
Verifies preprocessing pipeline, tensor conversions, PSNR, and SSIM calculations.
"""

import unittest
import numpy as np
from PIL import Image
import torch

from utils.image_processing import (
    validate_image_file,
    preprocess_image_for_model,
    tensor_to_pil_image,
)
from utils.metrics import calculate_psnr, calculate_ssim, evaluate_restoration_pair
from training.dataset import degrade_photograph, generate_procedural_photograph


class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        # Create a sample clean image
        self.clean_pil = generate_procedural_photograph(seed=101, width=128, height=128)
        self.damaged_pil = degrade_photograph(self.clean_pil)

    def test_preprocessing_and_postprocessing(self):
        """Verifies roundtrip conversion between PIL and PyTorch tensors."""
        tensor = preprocess_image_for_model(self.clean_pil, target_size=(128, 128))
        
        # Verify shape and ranges
        self.assertEqual(tensor.shape, (1, 3, 128, 128))
        self.assertTrue(torch.all(tensor >= 0.0) and torch.all(tensor <= 1.0))
        
        # Postprocess back to PIL
        reconstructed_pil = tensor_to_pil_image(tensor)
        self.assertEqual(reconstructed_pil.size, (128, 128))
        self.assertEqual(reconstructed_pil.mode, "RGB")

    def test_metrics_identical_images(self):
        """Tests that identical images yield perfect fidelity metrics."""
        psnr_val = calculate_psnr(self.clean_pil, self.clean_pil)
        ssim_val = calculate_ssim(self.clean_pil, self.clean_pil)
        
        self.assertEqual(psnr_val, 100.0)
        self.assertAlmostEqual(ssim_val, 1.0, places=3)

    def test_metrics_degraded_images(self):
        """Tests that damaged images exhibit realistic lower PSNR and SSIM than clean originals."""
        psnr_val = calculate_psnr(self.clean_pil, self.damaged_pil)
        ssim_val = calculate_ssim(self.clean_pil, self.damaged_pil)
        
        # Damaged images typically have PSNR between 10 dB and 35 dB
        self.assertGreater(psnr_val, 5.0)
        self.assertLess(psnr_val, 45.0)
        # SSIM should be strictly less than 1.0
        self.assertLess(ssim_val, 0.99)
        self.assertGreater(ssim_val, 0.0)

    def test_evaluate_restoration_pair_no_ground_truth(self):
        """Verifies graceful handling when clean reference is missing."""
        result = evaluate_restoration_pair(None, self.damaged_pil)
        self.assertFalse(result["has_ground_truth"])
        self.assertIsNone(result["psnr"])
        self.assertIsNone(result["ssim"])
        self.assertIn("unavailable", result["message"].lower())


if __name__ == "__main__":
    unittest.main()
