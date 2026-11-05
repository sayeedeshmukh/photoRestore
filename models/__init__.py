"""
Models package for PhotoRestore AI.
Exports Convolutional Autoencoder (CAE) and Variational Autoencoder (VAE).
"""

from models.autoencoder import ConvAutoencoder
from models.vae import VariationalAutoencoder, vae_loss_function


def count_parameters(model) -> int:
    """Returns the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "ConvAutoencoder",
    "VariationalAutoencoder",
    "vae_loss_function",
    "count_parameters",
]
