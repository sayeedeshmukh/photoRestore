"""
Comprehensive training pipeline for PhotoRestore AI.
Trains Convolutional Autoencoder (CAE) and Variational Autoencoder (VAE)
with genuine loss tracking, best checkpoint persistence, history serialization, and loss plots.
"""

from typing import Dict, Any, List, Optional
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import (
    CAE_CONFIG,
    VAE_CONFIG,
    TRAIN_CONFIG,
    CAE_CHECKPOINT_PATH,
    VAE_CHECKPOINT_PATH,
    CAE_HISTORY_PATH,
    VAE_HISTORY_PATH,
    CAE_LOSS_PLOT_PATH,
    VAE_LOSS_PLOT_PATH,
    DEVICE,
)
from models.autoencoder import ConvAutoencoder
from models.vae import VariationalAutoencoder, vae_loss_function
from training.dataset import get_dataloaders


class CombinedL1SSIMLoss(nn.Module):
    """
    Composite loss combining L1 Pixel Loss and Structural Similarity (SSIM) Loss.
    Penalizes both numerical pixel deviations and structural / edge degradation.
    """
    def __init__(self, alpha: float = 0.8, window_size: int = 11):
        super().__init__()
        self.alpha = alpha
        self.window_size = window_size
        self.l1 = nn.L1Loss()
        
        sigma = 1.5
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel_2d = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel_2d.repeat(3, 1, 1, 1))

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        k = self.kernel.to(img1.device)
        mu1 = F.conv2d(img1, k, padding=pad, groups=3)
        mu2 = F.conv2d(img2, k, padding=pad, groups=3)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, k, padding=pad, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, k, padding=pad, groups=3) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, k, padding=pad, groups=3) - mu1_mu2
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-7)
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_loss = self.l1(pred, target)
        ssim_loss = 1.0 - self._ssim(pred, target)
        return self.alpha * l1_loss + (1.0 - self.alpha) * ssim_loss


def set_seed(seed: int = 42):
    """Sets reproducible random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Path,
    model_name: str
):
    """Generates and saves professional training and validation loss curves."""
    epochs = list(range(1, len(train_losses) + 1))
    
    plt.figure(figsize=(8, 5), dpi=150)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    plt.plot(epochs, train_losses, label="Training Loss", color="#1E88E5", linewidth=2.2, marker="o", markersize=4)
    plt.plot(epochs, val_losses, label="Validation Loss", color="#FF7043", linewidth=2.2, linestyle="--", marker="s", markersize=4)
    
    plt.title(f"{model_name} — Training & Validation Loss", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=11, fontweight="medium")
    plt.ylabel("Loss", fontsize=11, fontweight="medium")
    plt.legend(frameon=True, facecolor="white", framealpha=0.9, fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"[Plot] Saved loss curve to: {save_path}")


# =====================================================================
# Training: Convolutional Autoencoder (CAE)
# =====================================================================

def train_cae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    resume: bool = False,
    device: torch.device = DEVICE
) -> Dict[str, Any]:
    """Trains the Convolutional Autoencoder to map damaged photos to clean targets."""
    print("\n" + "=" * 65)
    print("  TRAINING CONVOLUTIONAL AUTOENCODER (CAE)")
    print(f"  Target Device: {device} | Epochs: {epochs} | LR: {lr}")
    print("=" * 65)
    
    model = ConvAutoencoder(
        in_channels=CAE_CONFIG["input_channels"],
        base_channels=CAE_CONFIG["base_channels"]
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = CombinedL1SSIMLoss(alpha=0.8).to(device)  # L1 + SSIM loss preserves high-frequency edges and fine details
    
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    
    if resume and CAE_CHECKPOINT_PATH.exists():
        print(f"[Resume] Loading checkpoint from {CAE_CHECKPOINT_PATH}...")
        ckpt = torch.load(CAE_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if CAE_HISTORY_PATH.exists():
            with open(CAE_HISTORY_PATH, "r") as f:
                history = json.load(f).get("history", history)
        print(f"[Resume] Resumed from epoch {start_epoch} with best val loss: {best_val_loss:.5f}")
        
    start_total_time = time.time()
    
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_batches = 0
        
        for batch in train_loader:
            damaged = batch["damaged"].to(device)
            pristine = batch["pristine"].to(device)
            
            optimizer.zero_grad()
            restored = model(damaged)
            loss = criterion(restored, pristine)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            train_batches += 1
            
        avg_train_loss = running_train_loss / max(1, train_batches)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                damaged = batch["damaged"].to(device)
                pristine = batch["pristine"].to(device)
                restored = model(damaged)
                val_loss = criterion(restored, pristine)
                running_val_loss += val_loss.item()
                val_batches += 1
                
        avg_val_loss = running_val_loss / max(1, val_batches)
        epoch_duration = time.time() - epoch_start
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epoch_times"].append(round(epoch_duration, 2))
        
        is_best = avg_val_loss < best_val_loss
        status_marker = " [BEST]" if is_best else ""
        
        print(
            f"Epoch [{epoch + 1:02d}/{start_epoch + epochs:02d}] "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | "
            f"Time: {epoch_duration:.2f}s{status_marker}"
        )
        
        if is_best:
            best_val_loss = avg_val_loss
            CAE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_type": "cae",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
                "config": CAE_CONFIG,
                "timestamp": datetime.now().isoformat()
            }, CAE_CHECKPOINT_PATH)
            
    total_duration = time.time() - start_total_time
    
    # Save training history
    summary = {
        "model_name": CAE_CONFIG["name"],
        "model_type": "cae",
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": round(history["train_loss"][-1], 6) if history["train_loss"] else None,
        "final_val_loss": round(history["val_loss"][-1], 6) if history["val_loss"] else None,
        "total_duration_seconds": round(total_duration, 2),
        "checkpoint_path": str(CAE_CHECKPOINT_PATH),
        "loss_plot_path": str(CAE_LOSS_PLOT_PATH),
        "device": str(device),
        "history": history,
        "completed_at": datetime.now().isoformat()
    }
    
    with open(CAE_HISTORY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
        
    # Generate loss curve plot
    if history["train_loss"]:
        plot_loss_curves(
            history["train_loss"],
            history["val_loss"],
            CAE_LOSS_PLOT_PATH,
            CAE_CONFIG["name"]
        )
        
    print(f"[CAE Completed] Best Val Loss: {best_val_loss:.5f} in {total_duration:.1f}s.")
    return summary


# =====================================================================
# Training: Variational Autoencoder (VAE)
# =====================================================================

def train_vae(
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    kl_weight: float = 0.0005,
    recon_loss_type: str = "l1",
    resume: bool = False,
    device: torch.device = DEVICE
) -> Dict[str, Any]:
    """Trains the Variational Autoencoder with composite Reconstruction + KLD loss."""
    print("\n" + "=" * 65)
    print("  TRAINING VARIATIONAL AUTOENCODER (VAE)")
    print(f"  Target Device: {device} | Epochs: {epochs} | LR: {lr} | Beta (KL): {kl_weight}")
    print("=" * 65)
    
    model = VariationalAutoencoder(
        in_channels=VAE_CONFIG["input_channels"],
        latent_dim=VAE_CONFIG["latent_dim"],
        base_channels=VAE_CONFIG["base_channels"]
    ).to(device)
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    start_epoch = 0
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "recon_loss": [], "kld_loss": [], "epoch_times": []}
    
    if resume and VAE_CHECKPOINT_PATH.exists():
        print(f"[Resume] Loading checkpoint from {VAE_CHECKPOINT_PATH}...")
        ckpt = torch.load(VAE_CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("val_loss", float("inf"))
        if VAE_HISTORY_PATH.exists():
            with open(VAE_HISTORY_PATH, "r") as f:
                history = json.load(f).get("history", history)
        print(f"[Resume] Resumed from epoch {start_epoch} with best val loss: {best_val_loss:.5f}")
        
    start_total_time = time.time()
    
    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = time.time()
        model.train()
        running_total_loss = 0.0
        running_recon = 0.0
        running_kld = 0.0
        train_batches = 0
        
        for batch in train_loader:
            damaged = batch["damaged"].to(device)
            pristine = batch["pristine"].to(device)
            
            optimizer.zero_grad()
            restored, mu, logvar = model(damaged)
            
            loss_dict = vae_loss_function(
                recon_x=restored,
                target_clean=pristine,
                mu=mu,
                logvar=logvar,
                kl_weight=kl_weight,
                recon_loss_type=recon_loss_type
            )
            
            total_loss = loss_dict["loss"]
            total_loss.backward()
            optimizer.step()
            
            running_total_loss += total_loss.item()
            running_recon += loss_dict["recon_loss"].item()
            running_kld += loss_dict["kld_loss"].item()
            train_batches += 1
            
        avg_train_loss = running_total_loss / max(1, train_batches)
        avg_recon_loss = running_recon / max(1, train_batches)
        avg_kld_loss = running_kld / max(1, train_batches)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                damaged = batch["damaged"].to(device)
                pristine = batch["pristine"].to(device)
                restored, mu, logvar = model(damaged)
                val_dict = vae_loss_function(
                    recon_x=restored,
                    target_clean=pristine,
                    mu=mu,
                    logvar=logvar,
                    kl_weight=kl_weight,
                    recon_loss_type=recon_loss_type
                )
                running_val_loss += val_dict["loss"].item()
                val_batches += 1
                
        avg_val_loss = running_val_loss / max(1, val_batches)
        epoch_duration = time.time() - epoch_start
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["recon_loss"].append(avg_recon_loss)
        history["kld_loss"].append(avg_kld_loss)
        history["epoch_times"].append(round(epoch_duration, 2))
        
        is_best = avg_val_loss < best_val_loss
        status_marker = " [BEST]" if is_best else ""
        
        print(
            f"Epoch [{epoch + 1:02d}/{start_epoch + epochs:02d}] "
            f"Train Loss: {avg_train_loss:.5f} (Recon: {avg_recon_loss:.4f}, KLD: {avg_kld_loss:.4f}) | "
            f"Val Loss: {avg_val_loss:.5f} | "
            f"Time: {epoch_duration:.2f}s{status_marker}"
        )
        
        if is_best:
            best_val_loss = avg_val_loss
            VAE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_type": "vae",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_loss": best_val_loss,
                "config": VAE_CONFIG,
                "timestamp": datetime.now().isoformat()
            }, VAE_CHECKPOINT_PATH)
            
    total_duration = time.time() - start_total_time
    
    # Save training history
    summary = {
        "model_name": VAE_CONFIG["name"],
        "model_type": "vae",
        "epochs_trained": len(history["train_loss"]),
        "best_val_loss": round(best_val_loss, 6),
        "final_train_loss": round(history["train_loss"][-1], 6) if history["train_loss"] else None,
        "final_val_loss": round(history["val_loss"][-1], 6) if history["val_loss"] else None,
        "total_duration_seconds": round(total_duration, 2),
        "checkpoint_path": str(VAE_CHECKPOINT_PATH),
        "loss_plot_path": str(VAE_LOSS_PLOT_PATH),
        "device": str(device),
        "history": history,
        "completed_at": datetime.now().isoformat()
    }
    
    with open(VAE_HISTORY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
        
    # Generate loss curve plot
    if history["train_loss"]:
        plot_loss_curves(
            history["train_loss"],
            history["val_loss"],
            VAE_LOSS_PLOT_PATH,
            VAE_CONFIG["name"]
        )
        
    print(f"[VAE Completed] Best Val Loss: {best_val_loss:.5f} in {total_duration:.1f}s.")
    return summary


# =====================================================================
# Main CLI Orchestrator
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Photo Restoration Deep Learning Models (CAE and VAE).")
    parser.add_argument("--model", type=str, choices=["cae", "vae", "all"], default="all",
                        help="Select neural network architecture to train.")
    parser.add_argument("--epochs", type=int, default=TRAIN_CONFIG["epochs"],
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"],
                        help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=TRAIN_CONFIG["learning_rate"],
                        help="Adam optimizer learning rate.")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Restrict dataset to N samples for rapid testing.")
    parser.add_argument("--kl-weight", type=float, default=VAE_CONFIG["kl_weight"],
                        help="VAE KL-divergence balancing weight beta.")
    parser.add_argument("--recon-loss", type=str, choices=["l1", "mse"], default=VAE_CONFIG["recon_loss_type"],
                        help="Reconstruction loss metric (L1 or MSE).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run quick verification on a few batches to test pipeline integrity.")
    parser.add_argument("--force-synthetic", action="store_true",
                        help="Force generation of synthetic degradation pairs without downloading from HF.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoint if available.")
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"],
                        help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.smoke_test:
        print("[Smoke-Test Mode Activated] Limiting to small subset and 1 epoch...")
        epochs = TRAIN_CONFIG["smoke_test_epochs"]
        subset_size = TRAIN_CONFIG["smoke_test_samples"]
        batch_size = 8
    else:
        epochs = args.epochs
        subset_size = args.subset_size
        batch_size = args.batch_size
        
    print(f"[Pipeline] Initializing DataLoaders on device: {DEVICE}...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        batch_size=batch_size,
        subset_size=subset_size,
        seed=args.seed,
        force_synthetic=args.force_synthetic
    )
    print(
        f"[Dataset Loaded] Source: {meta['source']} | "
        f"Train: {meta['train_pairs']} pairs | "
        f"Val: {meta['val_pairs']} pairs | "
        f"Test: {meta['test_pairs']} pairs"
    )
    
    if args.model in ["cae", "all"]:
        train_cae(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=args.lr,
            resume=args.resume,
            device=DEVICE
        )
        
    if args.model in ["vae", "all"]:
        train_vae(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=args.lr,
            kl_weight=args.kl_weight,
            recon_loss_type=args.recon_loss,
            resume=args.resume,
            device=DEVICE
        )
        
    print("\n" + "=" * 65)
    print("  ALL REQUESTED TRAINING PHASES COMPLETED SUCCESSFULLY")
    print("=" * 65)


if __name__ == "__main__":
    main()
