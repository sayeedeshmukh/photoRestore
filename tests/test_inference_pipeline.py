"""
Unit tests for the complete inference and restoration pipeline.
Tests checkpoint verification, damage detection, LaMa inpainting, face enhancement integration,
deep learning colorization, and dimension/channel preservation.
"""

import unittest
from PIL import Image
import numpy as np
import torch

from utils.inference import (
    check_checkpoint_status,
    load_trained_model,
    restore_image,
)
from utils.damage_detection import detect_damage_mask
from utils.lama_inpainting import is_lama_available
from utils.face_enhancer import is_face_restorer_available
from utils.colorizer import is_colorizer_available, colorize_image


class TestInferencePipeline(unittest.TestCase):
    def test_checkpoint_status_keys(self):
        """Verifies checkpoint status dictionary contains all expected components."""
        status = check_checkpoint_status()
        self.assertIn("cae", status)
        self.assertIn("vae", status)
        self.assertIn("lama", status)
        self.assertIn("face_prior", status)
        self.assertIn("colorizer", status)
        self.assertTrue(status["cae"]["exists"])
        self.assertTrue(status["vae"]["exists"])
        self.assertTrue(status["lama"]["exists"])
        self.assertTrue(status["face_prior"]["exists"])
        self.assertTrue(status["colorizer"]["exists"])

    def test_model_loading_strict(self):
        """Verifies CAE model loads with strict=True without state_dict mismatches."""
        cae_model, msg = load_trained_model("cae")
        self.assertIsNotNone(cae_model)
        self.assertIn("strict=True", msg)

        vae_model, msg = load_trained_model("vae")
        self.assertIsNotNone(vae_model)
        self.assertIn("strict=True", msg)

    def test_damage_detection(self):
        """Verifies automatic damage mask generation produces valid binary mask."""
        test_img = Image.new("RGB", (100, 100), color=(150, 150, 150))
        # Draw a scratch
        for i in range(20, 80):
            test_img.putpixel((i, i), (255, 255, 255))
        mask = detect_damage_mask(test_img, sensitivity=0.5)
        self.assertEqual(mask.size, (100, 100))
        self.assertEqual(mask.mode, "L")
        mask_arr = np.array(mask)
        self.assertTrue(np.any(mask_arr > 0))

    def test_restore_image_dimension_preservation(self):
        """Verifies arbitrary image dimensions and channels are preserved exactly."""
        model, _ = load_trained_model("cae")
        test_w, test_h = 160, 140
        dummy_img = Image.new("RGB", (test_w, test_h), color=(120, 140, 160))

        # Test full restoration
        restored, latency, face_count, dmg_pct, was_colorized = restore_image(
            model=model,
            image=dummy_img,
            device=torch.device("cpu"),
            run_damage_removal=False,
            run_face_restoration=False,
            run_colorization=False,
            run_substrate_cae=True,
        )
        self.assertEqual(restored.size, (test_w, test_h))
        self.assertEqual(restored.mode, "RGB")
        self.assertGreater(latency, 0.0)
        self.assertEqual(face_count, 0)
        self.assertEqual(dmg_pct, 0.0)
        self.assertFalse(was_colorized)

    def test_pretrained_models_available(self):
        """Verifies LaMa, GFPGAN, and Colorizer weights are downloaded and available."""
        self.assertTrue(is_lama_available())
        self.assertTrue(is_face_restorer_available())
        self.assertTrue(is_colorizer_available())

    def test_colorizer_execution(self):
        """Verifies colorizer successfully processes grayscale PIL image."""
        gray_img = Image.new("L", (128, 128), color=120)
        colorized = colorize_image(gray_img)
        self.assertEqual(colorized.size, (128, 128))
        self.assertEqual(colorized.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
