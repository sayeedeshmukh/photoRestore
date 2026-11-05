# PhotoRestore AI

**Deep Learning-Based Restoration and Colorization of Damaged Photographs**

PhotoRestore AI is a streamlined, single-purpose deep learning tool designed to restore vintage, torn, scratched, and faded photographs using a unified deep learning pipeline:
- **Pretrained LaMa Inpainting:** Fast Fourier Convolution network for complete physical scratch, crack, stain, and paper abrasion removal.
- **GFPGAN Generative Facial Prior:** High-fidelity identity-preserving facial reconstruction (eyes, pupils, nose, mouth, teeth, hair) with zero blank faces.
- **Deep Convolutional Colorization (Zhang et al. ECCV 2016):** Converts black-and-white photographs into naturally colorized photographs with authentic skin tones and clothing colors.
- **In-House Trained Models:** Convolutional Autoencoder (CAE) and Variational Autoencoder (VAE) for neural substrate and noise regularization.

---

## Features

- **Single-Page Simplicity:** Focuses exclusively on uploading a damaged photograph, reconstructing it with deep neural networks, and downloading the restored image.
- **Face-Safe Neural Inpainting:** Automatically detects facial regions upfront using RetinaFace and protects them so inpainting heals background cracks and scratches without erasing or blurring facial details.
- **Generative Facial Geometry:** Restores every face with complete facial anatomy—no face remains empty or smoothed into a blank silhouette.
- **Deep Learning Colorization:** Converts black-and-white photos into natural color photographs while preserving 100% of the original uncompressed luminance sharpness.
- **Arbitrary Image Dimensions:** Accepts JPG, JPEG, and PNG images of any aspect ratio, dynamically preserving original proportions and resolution.
- **Clean & Professional UI:** Minimal, modern single-page Streamlit design with zero emojis or decorative clutter.

---

## Project Structure

```
DL-2 Project/
├── app.py                      # Single-page Streamlit application
├── config.py                   # Model configurations, image resolution, and paths
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation and guide
├── .gitignore                  # Git ignore rules for virtualenvs, caches, and heavy weights
├── .streamlit/
│   └── config.toml             # Streamlit theme and UI settings
├── models/
│   ├── __init__.py             # Exports models and parameter count utility
│   ├── autoencoder.py          # Convolutional Autoencoder (CAE) architecture
│   ├── vae.py                  # Variational Autoencoder (VAE) architecture & loss
│   ├── cae_best.pth            # Trained in-house CAE checkpoint
│   ├── colorization_deploy_v2.prototxt # Caffe colorization network definition
│   └── pts_in_hull.npy         # Quantized color cluster centers (Lab ab space)
├── scripts/
│   └── download_weights.py     # Automated downloader for pretrained model weights
├── training/
│   ├── __init__.py             # Package init
│   ├── dataset.py              # Authentic photographic dataset loader and degradation engine
│   ├── train.py                # Model training pipeline
│   └── evaluate.py             # Quantitative evaluation script (PSNR, SSIM, latency)
├── utils/
│   ├── __init__.py             # Package init
│   ├── colorizer.py            # Deep learning colorization engine (OpenCV DNN)
│   ├── damage_detection.py     # Multi-scale defect detection with face protection
│   ├── face_enhancer.py        # GFPGAN face restoration and RetinaFace detection
│   ├── image_processing.py     # Image validation, tensor conversions, aspect preservation
│   ├── inference.py            # Neural inference pipeline with latency timing
│   ├── lama_inpainting.py      # LaMa inpainting wrapper
│   ├── metrics.py              # Quantitative fidelity metrics (PSNR, SSIM)
│   └── ui_components.py        # Custom styling and cached model loaders
├── artifacts/
│   ├── cae_history.json        # CAE training logs
│   ├── vae_history.json        # VAE training logs
│   ├── evaluation_results.json # Held-out test evaluation benchmarks
│   ├── cae_loss_curves.png     # CAE loss curves
│   └── vae_loss_curves.png     # VAE loss curves
└── tests/
    ├── __init__.py
    ├── test_models.py          # Model architecture, tensor shape, and loss tests
    ├── test_image_processing.py# Preprocessing and quantitative metrics tests
    └── test_inference_pipeline.py # End-to-end pipeline and dimension tests
```

---

## Execution Guide

### 1. Installation
Install all dependencies in your Python environment:

```powershell
pip install -r requirements.txt
```

### 2. Download Pretrained Weights (Optional / If Needed)
To verify and download foundation model weights (GFPGAN, LaMa, and Colorization model):

```powershell
python scripts/download_weights.py
```

### 3. Launch the Application
Run the Streamlit application from PowerShell:

```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### 4. Run Unit Tests
To verify all neural network dimensions, tensor shapes, and pipeline integrations:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

---

## Empirical Benchmark Performance

Evaluated on the held-out test split of authentic degraded photographs:

| Model Architecture | Test PSNR (dB) | Test SSIM | Inference Latency |
| :--- | :---: | :---: | :---: |
| **Convolutional Autoencoder (CAE)** | **22.34 dB** (&plusmn; 1.99) | **0.5934** (&plusmn; 0.0753) | ~16 ms |
| **Variational Autoencoder (VAE)** | **21.82 dB** (&plusmn; 2.04) | **0.5910** (&plusmn; 0.0708) | ~10 ms |
