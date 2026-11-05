"""
Unit tests for Deep Learning Models (CAE and VAE).
Verifies tensor dimensions, forward passes, reparameterization trick, and loss functions.
"""

import unittest
import torch
from models.autoencoder import ConvAutoencoder
from models.vae import VariationalAutoencoder, vae_loss_function
from models import count_parameters


class TestAutoencoders(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 3
        self.height = 128
        self.width = 128
        self.dummy_input = torch.rand(self.batch_size, self.channels, self.height, self.width)

    def test_cae_forward_and_shapes(self):
        """Tests Convolutional Autoencoder output dimensions and pixel value ranges."""
        model = ConvAutoencoder(in_channels=3, base_channels=32)
        model.eval()
        
        with torch.no_grad():
            latent = model.encode(self.dummy_input)
            restored = model(self.dummy_input)
            
        # Check latent shape: (B, 64, 8, 8)
        self.assertEqual(latent.shape, (self.batch_size, 64, 8, 8))
        # Check restored shape: matches input dimensions
        self.assertEqual(restored.shape, (self.batch_size, self.channels, self.height, self.width))
        # Check value bounds: Sigmoid guarantees [0.0, 1.0]
        self.assertTrue(torch.all(restored >= 0.0) and torch.all(restored <= 1.0))
        # Check parameter count is reasonable (> 50k)
        self.assertGreater(count_parameters(model), 50000)

    def test_vae_forward_and_shapes(self):
        """Tests Variational Autoencoder latent statistics, output shapes, and reparameterization."""
        latent_dim = 256
        model = VariationalAutoencoder(in_channels=3, latent_dim=latent_dim, base_channels=32)
        
        # Test training mode (stochastic reparameterization)
        model.train()
        restored, mu, logvar = model(self.dummy_input)
        
        self.assertEqual(restored.shape, (self.batch_size, self.channels, self.height, self.width))
        self.assertEqual(mu.shape, (self.batch_size, latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, latent_dim))
        self.assertTrue(torch.all(restored >= 0.0) and torch.all(restored <= 1.0))
        
        # Test eval mode (deterministic forward)
        model.eval()
        with torch.no_grad():
            eval_restored, eval_mu, eval_logvar = model(self.dummy_input)
            self.assertEqual(eval_restored.shape, (self.batch_size, self.channels, self.height, self.width))

    def test_vae_loss_calculation(self):
        """Tests VAE composite loss computation (reconstruction loss + KL divergence)."""
        latent_dim = 256
        restored = torch.rand(self.batch_size, 3, 128, 128)
        target = torch.rand(self.batch_size, 3, 128, 128)
        mu = torch.randn(self.batch_size, latent_dim)
        logvar = torch.randn(self.batch_size, latent_dim)
        
        loss_dict = vae_loss_function(
            recon_x=restored,
            target_clean=target,
            mu=mu,
            logvar=logvar,
            kl_weight=0.001,
            recon_loss_type="l1"
        )
        
        self.assertIn("loss", loss_dict)
        self.assertIn("recon_loss", loss_dict)
        self.assertIn("kld_loss", loss_dict)
        
        # Losses must be finite, non-negative, and scalar
        self.assertFalse(torch.isnan(loss_dict["loss"]))
        self.assertFalse(torch.isinf(loss_dict["loss"]))
        self.assertGreater(loss_dict["loss"].item(), 0.0)
        self.assertGreater(loss_dict["recon_loss"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
