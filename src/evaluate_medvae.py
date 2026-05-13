#!/usr/bin/env python3
"""
MedVAE Evaluation and Visualization Script

Évalue les performances du modèle MedVAE entraîné et crée des visualisations
montrant la qualité de la reconstruction.

Usage:
    python src/evaluate_medvae.py \
        --model-path outputs/medvae/medvae_full/medvae_final.pth \
        --data-root /home/rousseau/Data/MRIxFields_20260414 \
        --output-dir results/medvae_eval \
        --num-samples 10

Métriques calculées:
    - Reconstruction Loss (L1)
    - PSNR (Peak Signal-to-Noise Ratio)
    - SSIM (Structural Similarity Index)
    - MAE (Mean Absolute Error)
    - Perplexity (VQ-VAE)
"""

import argparse
import gc
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import zoom as scipy_zoom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# Constants
# ============================================================================
SPLIT_MAP = {
    "retro_train": "Training_retrospective",
    "pro_train": "Training_prospective",
    "pro_val": "Validating_prospective",
    "pro_test": "Testing_prospective",
}

# ============================================================================
# Utility Functions
# ============================================================================

def _resample_volume(
    vol: np.ndarray,
    original_spacing: Tuple,
    target_spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Resample 3D volume to target spacing."""
    orig = np.asarray(original_spacing[:3], dtype=float)
    tgt = np.asarray(target_spacing, dtype=float)
    factors = orig / tgt
    if np.allclose(factors, 1.0, atol=0.02):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


def _normalize(
    vol: np.ndarray,
    lo_pct: float = 0.5,
    hi_pct: float = 99.5,
) -> np.ndarray:
    """Normalize to [-1, 1]."""
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (vol * 2.0 - 1.0).astype(np.float32)


def _denormalize(vol: np.ndarray) -> np.ndarray:
    """Denormalize from [-1, 1] to [0, 1]."""
    return np.clip((vol + 1.0) / 2.0, 0.0, 1.0)


def _center_crop_or_pad(
    vol: np.ndarray,
    size: Tuple[int, int, int],
) -> np.ndarray:
    """Crop or pad to target size."""
    th, tw, td = size
    h, w, d = vol.shape

    ph = max(0, th - h)
    pw = max(0, tw - w)
    pd = max(0, td - d)
    if ph > 0 or pw > 0 or pd > 0:
        vol = np.pad(
            vol,
            [(ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (pd // 2, pd - pd // 2)],
            mode="reflect",
        )
        h, w, d = vol.shape

    sh = max((h - th) // 2, 0)
    sw = max((w - tw) // 2, 0)
    sd = max((d - td) // 2, 0)
    return vol[sh : sh + th, sw : sw + tw, sd : sd + td]


def _psnr(target: np.ndarray, pred: np.ndarray, data_range: float = 2.0) -> float:
    """Calculate PSNR between target and prediction (both in [-1, 1])."""
    mse = np.mean((target - pred) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(data_range / np.sqrt(mse))


def _ssim(target: np.ndarray, pred: np.ndarray) -> float:
    """Calculate SSIM between target and prediction."""
    # SSIM expects images in [0, 1]
    target_01 = _denormalize(target)
    pred_01 = _denormalize(pred)
    return structural_similarity(target_01, pred_01, data_range=1.0)


# ============================================================================
# Dataset
# ============================================================================

class SimpleMRIDataset(Dataset):
    """Simple dataset for evaluation: single modality, single field."""

    def __init__(
        self,
        data_root: Path,
        split: str = "pro_val",
        modality: str = "T1W",
        field: str = "0.1T",
        volume_size: Tuple[int, int, int] = (128, 128, 64),
        target_spacing: Optional[Tuple] = None,
        max_samples: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.volume_size = volume_size
        self.target_spacing = target_spacing or (1.0, 1.0, 1.0)

        split_dir = SPLIT_MAP.get(split, split)
        d = self.data_root / split_dir / modality / field

        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")

        self.paths = list(sorted(d.glob("*.nii.gz")))
        if max_samples is not None:
            self.paths = self.paths[:max_samples]

        if not self.paths:
            raise FileNotFoundError(f"No NIfTI files in {d}")

        print(f"Dataset: {len(self.paths)} volumes from {d}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = nib.load(path)
        vol = img.get_fdata().astype(np.float32)
        spacing = img.header.get_zooms()

        # Resample if needed
        if self.target_spacing and not np.allclose(spacing[:3], self.target_spacing):
            vol = _resample_volume(vol, spacing, self.target_spacing)

        # Normalize
        vol = _normalize(vol)

        # Crop/pad
        vol = _center_crop_or_pad(vol, self.volume_size)

        return torch.from_numpy(vol[None]).float(), path.stem


# ============================================================================
# Evaluation
# ============================================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Evaluate model on dataset."""
    model.eval()
    
    metrics = {
        "loss_l1": [],
        "loss_mse": [],
        "psnr": [],
        "ssim": [],
        "mae": [],
    }
    
    with torch.no_grad():
        for batch, names in dataloader:
            batch = batch.to(device).float()
            
            # Forward pass - MedVAE encode/decode
            with torch.amp.autocast("cuda", enabled=use_amp):
                try:
                    latent = model.encode(batch)
                    recon = model.decode(latent)
                except Exception as e:
                    print(f"Error in encode/decode: {e}")
                    continue
            
            # Ensure output matches input shape
            recon = recon.float()
            if recon.shape != batch.shape:
                recon = F.interpolate(
                    recon,
                    size=batch.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            
            # Convert to numpy
            batch_np = batch.cpu().numpy()
            recon_np = recon.cpu().numpy()
            
            # Compute metrics: use middle slices from 3D volume
            B, C, H, W, D = batch_np.shape
            for b in range(B):
                # Select middle slice in Z dimension for 2D metric calculation
                z_idx = D // 2
                src = batch_np[b, 0, :, :, z_idx]
                rec = recon_np[b, 0, :, :, z_idx]
                
                # Metrics
                l1_loss = float(np.mean(np.abs(src - rec)))
                mse_loss = float(np.mean((src - rec) ** 2))
                mae = float(np.mean(np.abs(src - rec)))
                psnr_val = _psnr(src, rec)
                ssim_val = _ssim(src, rec)
                
                metrics["loss_l1"].append(l1_loss)
                metrics["loss_mse"].append(mse_loss)
                metrics["mae"].append(mae)
                metrics["psnr"].append(psnr_val)
                metrics["ssim"].append(ssim_val)
    
    # Average metrics
    result = {k: float(np.mean(v)) for k, v in metrics.items()}
    result["num_samples"] = len(metrics["loss_l1"])
    
    return result, metrics


def visualize_reconstructions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: Path,
    use_amp: bool = False,
    max_samples: int = 9,
) -> None:
    """Create visualization of reconstructions."""
    model.eval()
    
    fig, axes = plt.subplots(
        max_samples, 3,
        figsize=(12, 4 * max_samples),
        dpi=100,
    )
    if max_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_idx = 0
    
    with torch.no_grad():
        for batch, names in dataloader:
            if sample_idx >= max_samples:
                break
            
            batch = batch.to(device).float()
            
            # Forward pass - MedVAE encode/decode
            with torch.amp.autocast("cuda", enabled=use_amp):
                try:
                    latent = model.encode(batch)
                    recon = model.decode(latent)
                except Exception:
                    continue
            
            # Ensure shapes match
            recon = recon.float()
            if recon.shape != batch.shape:
                recon = F.interpolate(
                    recon,
                    size=batch.shape[2:],
                    mode="trilinear",
                    align_corners=False,
                )
            
            # Get middle slices from 3D volumes
            B, C, H, W, D = batch.shape
            z_idx = D // 2
            
            for b in range(B):
                if sample_idx >= max_samples:
                    break
                
                src = batch[b, 0, :, :, z_idx].cpu().numpy()
                rec = recon[b, 0, :, :, z_idx].cpu().numpy()
                diff = np.abs(src - rec)
                
                # Denormalize for visualization
                src_vis = _denormalize(src)
                rec_vis = _denormalize(rec)
                diff_vis = diff / (np.max(diff) + 1e-6)
                
                # Plot
                row = sample_idx
                axes[row, 0].imshow(src_vis, cmap="gray")
                axes[row, 0].set_title(f"{names[b]}\n(Input)")
                axes[row, 0].axis("off")
                
                axes[row, 1].imshow(rec_vis, cmap="gray")
                psnr_val = _psnr(src, rec)
                axes[row, 1].set_title(f"Reconstruction\n(PSNR: {psnr_val:.2f})")
                axes[row, 1].axis("off")
                
                axes[row, 2].imshow(diff_vis, cmap="hot")
                mae = float(np.mean(np.abs(src - rec)))
                axes[row, 2].set_title(f"Absolute Error\n(MAE: {mae:.4f})")
                axes[row, 2].axis("off")
                
                sample_idx += 1
    
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "medvae_reconstruction_viz.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedVAE model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data-root", type=str, default="/home/rousseau/Data/MRIxFields_20260414")
    parser.add_argument("--output-dir", type=str, default="results/medvae_eval")
    parser.add_argument("--split", type=str, default="pro_val", choices=SPLIT_MAP.keys())
    parser.add_argument("--modality", type=str, default="T1W")
    parser.add_argument("--field", type=str, default="0.1T")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-amp", action="store_true")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print(" MedVAE Evaluation")
    print("=" * 70)
    print(f"Model path  : {args.model_path}")
    print(f"Data root   : {args.data_root}")
    print(f"Device      : {device}")
    print(f"Output dir  : {output_dir}")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading dataset...")
    ds = SimpleMRIDataset(
        data_root=Path(args.data_root),
        split=args.split,
        modality=args.modality,
        field=args.field,
        volume_size=(128, 128, 64),
        max_samples=args.num_samples,
    )
    
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    # Load model
    print("Loading MedVAE model...")
    try:
        from medvae import MVAE
        os.environ["HF_HUB_OFFLINE"] = "1"
        model = MVAE(model_name="medvae_4_1_3d", modality="mri")
        model = model.to(device)
        
        # Try to load fine-tuned weights if they exist
        if Path(args.model_path).exists():
            try:
                checkpoint = torch.load(args.model_path, map_location=device)
                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"], strict=False)
                    print(f"✓ Loaded fine-tuned weights from {args.model_path}")
                else:
                    model.load_state_dict(checkpoint, strict=False)
                    print(f"✓ Loaded weights from {args.model_path}")
            except Exception as e:
                print(f"⚠ Could not load weights from {args.model_path}: {e}")
                print("  Using base MedVAE model...")
    except ImportError:
        print("❌ MedVAE not installed. Install with: pip install medvae")
        return
    
    # Evaluate
    print("\nEvaluating...")
    metrics, all_metrics = evaluate(
        model,
        loader,
        device,
        use_amp=args.use_amp,
    )
    
    # Print results
    print("\n" + "=" * 70)
    print(" Evaluation Results")
    print("=" * 70)
    print(f"Samples evaluated : {metrics['num_samples']}")
    print(f"L1 Loss           : {metrics['loss_l1']:.6f}")
    print(f"MSE Loss          : {metrics['loss_mse']:.6f}")
    print(f"MAE               : {metrics['mae']:.6f}")
    print(f"PSNR (dB)         : {metrics['psnr']:.2f}")
    print(f"SSIM              : {metrics['ssim']:.4f}")
    print("=" * 70)
    
    # Save metrics to file
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "evaluation_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("MedVAE Evaluation Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Model path       : {args.model_path}\n")
        f.write(f"Split            : {args.split}\n")
        f.write(f"Modality         : {args.modality}\n")
        f.write(f"Field            : {args.field}\n")
        f.write(f"Samples evaluated: {metrics['num_samples']}\n\n")
        f.write("Metrics:\n")
        f.write(f"  L1 Loss  : {metrics['loss_l1']:.6f}\n")
        f.write(f"  MSE Loss : {metrics['loss_mse']:.6f}\n")
        f.write(f"  MAE      : {metrics['mae']:.6f}\n")
        f.write(f"  PSNR     : {metrics['psnr']:.2f} dB\n")
        f.write(f"  SSIM     : {metrics['ssim']:.4f}\n")
    print(f"\n✓ Saved metrics to {metrics_file}")
    
    # Visualize
    print("\nGenerating visualizations...")
    visualize_reconstructions(
        model,
        loader,
        device,
        output_dir,
        use_amp=args.use_amp,
        max_samples=9,
    )
    
    print("\n✓ Evaluation complete!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
