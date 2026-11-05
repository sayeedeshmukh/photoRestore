"""
Variational Autoencoder (VAE) Architecture for Photograph Restoration.
Learns continuous probabilistic latent distributions with multi-scale skip guidance
to restore damaged photographs.
"""

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """
    Convolutional Variational Autoencoder (VAE) with Feature Guidance
    for Damaged Photograph Restoration.
    State dict keys: encoder_conv.0, encoder_conv.3, ..., fc_mu, fc_logvar, decoder_input, decoder_conv.0, ...
    """
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 256,
        base_channels: int = 32
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.in_channels = in_channels
        
        # 1. Convolutional Encoder: (B, 3, 128, 128) -> (B, 256, 8, 8)
        self.encoder_conv = nn.Sequential(
            # 0, 1, 2: 128x128 -> 64x64
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 3, 4, 5: 64x64 -> 32x32
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 6, 7, 8: 32x32 -> 16x16
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 9, 10, 11: 16x16 -> 8x8
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.flatten_dim = (base_channels * 8) * 8 * 8
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # 2. Decoder projection
        self.decoder_input = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 3. Convolutional Decoder
        self.decoder_conv = nn.Sequential(
            # 0, 1, 2: 8x8 -> 16x16
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # 3, 4, 5: 16x16 -> 32x32
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # 6, 7, 8: 32x32 -> 64x64
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 9, 10, 11: 64x64 -> 128x128
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 12, 13: Output mapping
            nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self._cached_skips = None

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode damaged image into latent distribution parameters mu and logvar."""
        e1 = self.encoder_conv[0:3](x)
        e2 = self.encoder_conv[3:6](e1)
        e3 = self.encoder_conv[6:9](e2)
        e4 = self.encoder_conv[9:12](e3)
        flat = torch.flatten(e4, start_dim=1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        self._cached_skips = (e1, e2, e3)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization Trick: z = mu + eps * exp(0.5 * logvar)."""
        if self.training:
            logvar = torch.clamp(logvar, -15.0, 15.0)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z back into restored RGB photograph."""
        projected = self.decoder_input(z)
        reshaped = projected.view(-1, self.base_channels * 8, 8, 8)
        skips = getattr(self, "_cached_skips", None)
        
        d1 = self.decoder_conv[0:3](reshaped)
        if skips is not None and d1.shape == skips[2].shape:
            d1 = d1 + skips[2]
        d2 = self.decoder_conv[3:6](d1)
        if skips is not None and d2.shape == skips[1].shape:
            d2 = d2 + skips[1]
        d3 = self.decoder_conv[6:9](d2)
        if skips is not None and d3.shape == skips[0].shape:
            d3 = d3 + skips[0]
        d4 = self.decoder_conv[9:12](d3)
        out = self.decoder_conv[12:14](d4)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass mapping damaged image x to restored image and latent statistics."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        restored = self.decode(z)
        return restored, mu, logvar


def vae_loss_function(
    recon_x: torch.Tensor,
    target_clean: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.0001,
    recon_loss_type: str = "l1"
) -> Dict[str, torch.Tensor]:
    """Computes composite VAE Loss: Reconstruction Loss (L1, MSE, or L1+SSIM) + KL Divergence."""
    if recon_loss_type.lower() == "mse":
        recon_loss = F.mse_loss(recon_x, target_clean, reduction="mean")
    elif recon_loss_type.lower() == "l1_ssim":
        l1 = F.l1_loss(recon_x, target_clean, reduction="mean")
        mu1 = F.avg_pool2d(recon_x, 3, 1, 1)
        mu2 = F.avg_pool2d(target_clean, 3, 1, 1)
        sigma12 = F.avg_pool2d(recon_x * target_clean, 3, 1, 1) - mu1 * mu2
        sigma1_sq = F.avg_pool2d(recon_x * recon_x, 3, 1, 1) - mu1.pow(2)
        sigma2_sq = F.avg_pool2d(target_clean * target_clean, 3, 1, 1) - mu2.pow(2)
        c1, c2 = 0.01**2, 0.03**2
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1.pow(2) + mu2.pow(2) + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-7)
        recon_loss = 0.8 * l1 + 0.2 * (1.0 - ssim.mean())
    else:
        recon_loss = F.l1_loss(recon_x, target_clean, reduction="mean")
        
    logvar_clamped = torch.clamp(logvar, -15.0, 15.0)
    kld_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar_clamped - mu.pow(2) - logvar_clamped.exp(), dim=1)
    )
    
    total_loss = recon_loss + (kl_weight * kld_loss)
    
    return {
        "loss": total_loss,
        "recon_loss": recon_loss,
        "kld_loss": kld_loss
    }
