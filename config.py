"""
Configuration settings for PhotoRestore AI:
Deep Learning-Based Restoration of Damaged Photographs Using Autoencoders and Variational Autoencoders.
"""

from pathlib import Path
import torch

# Base Project Directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CHECKPOINTS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

# Ensure runtime directories exist
for directory in [MODELS_DIR, CHECKPOINTS_DIR, ARTIFACTS_DIR, OUTPUTS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Image Configurations
IMAGE_SIZE = (128, 128)
IMAGE_CHANNELS = 3

# Model Architecture Configurations
CAE_CONFIG = {
    "name": "Convolutional Autoencoder (CAE)",
    "latent_channels": 64,
    "input_channels": 3,
    "output_channels": 3,
    "base_channels": 32,
}

VAE_CONFIG = {
    "name": "Variational Autoencoder (VAE)",
    "latent_dim": 256,
    "input_channels": 3,
    "output_channels": 3,
    "base_channels": 32,
    "kl_weight": 0.0001,
    "recon_loss_type": "l1",
}

# Training Hyperparameters
TRAIN_CONFIG = {
    "batch_size": 16,
    "learning_rate": 1e-3,
    "epochs": 20,
    "weight_decay": 1e-5,
    "subset_size": None,
    "smoke_test_samples": 32,
    "smoke_test_epochs": 1,
    "seed": 42,
    "val_split": 0.1,
    "test_split": 0.1,
}

# Checkpoint Filenames
CAE_CHECKPOINT_PATH = CHECKPOINTS_DIR / "cae_best.pth"
VAE_CHECKPOINT_PATH = CHECKPOINTS_DIR / "vae_best.pth"

# Training History & Evaluation Metrics Artifacts
CAE_HISTORY_PATH = ARTIFACTS_DIR / "cae_history.json"
VAE_HISTORY_PATH = ARTIFACTS_DIR / "vae_history.json"
EVALUATION_RESULTS_PATH = ARTIFACTS_DIR / "evaluation_results.json"
CAE_LOSS_PLOT_PATH = ARTIFACTS_DIR / "cae_loss_curves.png"
VAE_LOSS_PLOT_PATH = ARTIFACTS_DIR / "vae_loss_curves.png"

# Dataset Identifier on Hugging Face
HF_DATASET_NAME = "joshuachin/openphoto-restore-dataset"

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
