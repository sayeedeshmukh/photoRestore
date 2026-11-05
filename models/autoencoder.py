"""
Convolutional Autoencoder (CAE) Architecture for Photograph Restoration.
Maps damaged photographs (scratches, noise, fading, blur, missing regions) to clean counterparts.
Uses standard sequential encoder and decoder with multi-scale skip connections.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    """
    Deep Convolutional Autoencoder with Multi-Scale Skip Connections for Image Restoration.
    State dict keys: encoder.0, encoder.2, ..., decoder.0, decoder.3, ...
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        
        # Encoder: 128x128 -> 8x8 compact latent representation
        self.encoder = nn.Sequential(
            # 0, 1: 128x128 -> 128x128
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 2, 3, 4: 128x128 -> 64x64
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 5, 6, 7: 64x64 -> 32x32
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8, 9, 10: 32x32 -> 16x16
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 11, 12, 13: 16x16 -> 8x8
            nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 14, 15, 16: Bottleneck compact latent representation: 8x8 -> 8x8
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder: 8x8 -> 128x128 restored photograph
        self.decoder = nn.Sequential(
            # 0, 1, 2: Bottleneck expansion: 8x8 -> 8x8
            nn.ConvTranspose2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # 3, 4, 5: Up 1: 8x8 -> 16x16
            nn.ConvTranspose2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            
            # 6, 7, 8: Up 2: 16x16 -> 32x32
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # 9, 10, 11: Up 3: 32x32 -> 64x64
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # 12, 13, 14: Up 4: 64x64 -> 128x128
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # 15, 16: Reconstruction layer: 128x128 -> 128x128 RGB
            nn.Conv2d(base_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self._cached_skips = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract compact latent feature map from input image: (B, 64, 8, 8)."""
        e1 = self.encoder[0:2](x)       # 128x128 (base)
        e2 = self.encoder[2:5](e1)      # 64x64 (base*2)
        e3 = self.encoder[5:8](e2)      # 32x32 (base*4)
        e4 = self.encoder[8:11](e3)     # 16x16 (base*8)
        e5 = self.encoder[11:14](e4)    # 8x8 (base*4)
        z = self.encoder[14:17](e5)     # 8x8 (base*2)
        self._cached_skips = (e1, e2, e3, e4, e5)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation into restored image."""
        skips = getattr(self, "_cached_skips", None)
        d0 = self.decoder[0:3](z)
        if skips is not None and d0.shape == skips[4].shape:
            d0 = d0 + skips[4]
        d1 = self.decoder[3:6](d0)
        if skips is not None and d1.shape == skips[3].shape:
            d1 = d1 + skips[3]
        d2 = self.decoder[6:9](d1)
        if skips is not None and d2.shape == skips[2].shape:
            d2 = d2 + skips[2]
        d3 = self.decoder[9:12](d2)
        if skips is not None and d3.shape == skips[1].shape:
            d3 = d3 + skips[1]
        d4 = self.decoder[12:15](d3)
        if skips is not None and d4.shape == skips[0].shape:
            d4 = d4 + skips[0]
        out = self.decoder[15:17](d4)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass mapping damaged image x to restored image."""
        latent = self.encode(x)
        restored = self.decode(latent)
        return restored
