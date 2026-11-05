"""
Quantitative evaluation script for PhotoRestore AI.
Evaluates both trained models (CAE & VAE) on the exact same held-out test split,
computing genuine test loss, PSNR (dB), SSIM, and runtime latency.
"""

from typing import Dict, Any, List
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    CAE_CHECKPOINT_PATH,
    VAE_CHECKPOINT_PATH,
    EVALUATION_RESULTS_PATH,
    DEVICE,
)
from utils.inference import load_trained_model
from utils.metrics import calculate_psnr, calculate_ssim
from training.dataset import get_dataloaders


@torch.no_grad()
def evaluate_model_on_test_set(
    model: torch.nn.Module,
    test_loader: DataLoader,
    model_name: str,
    device: torch.device = DEVICE
) -> Dict[str, Any]:
    """Runs systematic quantitative evaluation on the held-out test split."""
    model.eval()
    psnr_scores: List[float] = []
    ssim_scores: List[float] = []
    latencies: List[float] = []
    
    print(f"\nEvaluating {model_name} on {len(test_loader.dataset)} test samples...")
    
    for batch in test_loader:
        damaged = batch["damaged"].to(device)
        pristine = batch["pristine"].to(device)
        
        batch_size = damaged.size(0)
        
        for i in range(batch_size):
            img_in = damaged[i:i+1]
            img_gt = pristine[i:i+1]
            
            # Synchronize for accurate timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            
            # Forward pass
            out = model(img_in)
            if isinstance(out, tuple):
                restored = out[0]  # VAE returns (restored, mu, logvar)
            else:
                restored = out
                
            if device.type == "cuda":
                torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)
            
            # Compute PSNR & SSIM against pristine ground truth
            psnr_val = calculate_psnr(img_gt, restored)
            ssim_val = calculate_ssim(img_gt, restored)
            
            psnr_scores.append(psnr_val)
            ssim_scores.append(ssim_val)
            
    mean_psnr = float(np.mean(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))
    mean_latency = float(np.mean(latencies))
    
    return {
        "model_name": model_name,
        "test_samples_evaluated": len(psnr_scores),
        "mean_psnr_db": round(mean_psnr, 2),
        "std_psnr_db": round(float(np.std(psnr_scores)), 2),
        "mean_ssim": round(mean_ssim, 4),
        "std_ssim": round(float(np.std(ssim_scores)), 4),
        "mean_latency_ms": round(mean_latency, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models on held-out test set.")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="Subset size matching training run.")
    parser.add_argument("--force-synthetic", action="store_true",
                        help="Evaluate on synthetic test set.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic test partition.")
    args = parser.parse_args()
    
    print(f"[Evaluation] Loading test split on {DEVICE}...")
    _, _, test_loader, meta = get_dataloaders(
        batch_size=16,
        subset_size=args.subset_size,
        seed=args.seed,
        force_synthetic=args.force_synthetic
    )
    
    results = {
        "dataset_metadata": meta,
        "evaluated_at": datetime.now().isoformat(),
        "device": str(DEVICE),
        "models": {}
    }
    
    # 1. Evaluate CAE
    cae_model, cae_msg = load_trained_model("cae", device=DEVICE)
    if cae_model is not None:
        cae_eval = evaluate_model_on_test_set(cae_model, test_loader, "Convolutional Autoencoder (CAE)", device=DEVICE)
        results["models"]["cae"] = cae_eval
    else:
        print(f"[Warning] CAE model could not be evaluated: {cae_msg}")
        
    # 2. Evaluate VAE
    vae_model, vae_msg = load_trained_model("vae", device=DEVICE)
    if vae_model is not None:
        vae_eval = evaluate_model_on_test_set(vae_model, test_loader, "Variational Autoencoder (VAE)", device=DEVICE)
        results["models"]["vae"] = vae_eval
    else:
        print(f"[Warning] VAE model could not be evaluated: {vae_msg}")
        
    # Save results to artifacts
    EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 65)
    print("  QUANTITATIVE EVALUATION RESULTS SUMMARY")
    print("=" * 65)
    for m_key, m_val in results["models"].items():
        print(f"Architecture: {m_val['model_name']}")
        print(f"  * PSNR: {m_val['mean_psnr_db']} dB (+/- {m_val['std_psnr_db']})")
        print(f"  * SSIM: {m_val['mean_ssim']} (+/- {m_val['std_ssim']})")
        print(f"  * Avg Latency: {m_val['mean_latency_ms']} ms/image")
        print("-" * 65)
    print(f"Results saved to: {EVALUATION_RESULTS_PATH}\n")


if __name__ == "__main__":
    main()
