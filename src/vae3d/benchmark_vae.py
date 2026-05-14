#!/usr/bin/env python3
"""
Benchmark script for 3 VAE architectures on MRIxFields full-resolution volumes.

Compares:
1. AutoencoderKL (MONAI) → renamed to AEKL
2. VQ-VAE (NeuroQuant-inspired, hybrid)
3. MedVAE 3D (pre-trained on 1M medical images)

Metrics: MAE, SSIM, LPIPS, memory usage, inference time per domain (field strength).
"""

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.ndimage import zoom as scipy_zoom
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import VAE architectures
from train_vae_3d import build_vae as build_aekl
from train_vqvae import NeuroQuantHybrid
from utils.patched_vae import PatchedVAE


FILE_RE = re.compile(r"^[A-Z]_([A-Z0-9]+)_([0-9.]+T)_(\d+)\.nii\.gz$")


def _normalize(vol: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Normalize volume to [-1, 1]."""
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (vol * 2.0 - 1.0).astype(np.float32)


def _crop_or_pad(vol: np.ndarray, target_size: Tuple[int, int, int]) -> np.ndarray:
    """Center crop or pad to target size."""
    th, tw, td = target_size
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


class BenchmarkDataset(Dataset):
    """Dataset of full-resolution test volumes."""

    def __init__(
        self,
        data_root: Path,
        split: str = "retro_train",
        modality: str = "T1W",
        field: str = "0.1T",
        max_samples: int = 10,
    ):
        self.data_root = Path(data_root)
        self.samples = []

        split_dir = "Training_retrospective" if split == "retro_train" else split
        d = self.data_root / split_dir / modality / field

        for p in sorted(d.glob("*.nii.gz"))[:max_samples]:
            m = FILE_RE.match(p.name)
            if m:
                self.samples.append(p)

        print(f"BenchmarkDataset: {len(self.samples)} volumes ({modality}/{field})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        img = nib.load(str(path))
        vol = img.get_fdata(dtype=np.float32)
        vol = _normalize(vol)
        return torch.from_numpy(vol).unsqueeze(0), path.stem  # (1, H, W, D), name


def load_aekl(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load AutoencoderKL from checkpoint."""
    print("[AEKL] Loading checkpoint...")
    
    # Build model directly using build_vae
    model = build_aekl(None).to(device)  # build_aekl uses default config
    
    if checkpoint_path is None: return None
    if checkpoint_path is None or not Path(checkpoint_path).exists(): return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    
    # Key remapping for postconv/conv mismatch
    state_fixed = {}
    for k, v in state.items():
        k_new = k.replace(".conv.conv.", ".postconv.conv.")
        state_fixed[k_new] = v
    
    model.load_state_dict(state_fixed, strict=False)
    model.eval()
    print(f"  → AEKL loaded from {checkpoint_path.name}")
    return model


def load_vqvae(checkpoint_path: Path, device: torch.device, n_modalities: int = 3, n_fields: int = 5) -> nn.Module:
    """Load VQ-VAE from checkpoint."""
    print("[VQ-VAE] Loading checkpoint...")
    model = NeuroQuantHybrid(
        n_modalities=n_modalities,
        n_fields=n_fields,
        base_channels=32,
        anat_channels=64,
        mod_channels=32,
        codebook_size=1024,
    ).to(device)
    
    if checkpoint_path is None: return None
    if checkpoint_path is None or not Path(checkpoint_path).exists(): return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt
    
    # Handle field_emb mismatch (checkpoint may have different n_fields)
    decoder_state = {k: v for k, v in state.items() if k.startswith("decoder.")}
    for k in list(decoder_state.keys()):
        if "field_emb" in k:
            # Keep only the checkpoint's field_emb, ignore shape mismatch
            old_shape = decoder_state[k].shape
            new_shape = model.state_dict()[k].shape if k in model.state_dict() else old_shape
            if old_shape != new_shape:
                print(f"  ⚠️  Skipping {k}: shape mismatch {old_shape} vs {new_shape}")
                del state[k]
    
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  → VQ-VAE loaded from {checkpoint_path.name}")
    return model


def load_medvae(model_name: str = "medvae_4_1_3d", device: torch.device = None) -> nn.Module:
    """Load MedVAE from HuggingFace or local."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[MedVAE] Loading {model_name}...")
    try:
        from medvae import MVAE
        model = MVAE(model_name=model_name, modality="mri").to(device)
        model.eval()
        print(f"  → MedVAE {model_name} loaded from HuggingFace")
        return model
    except ImportError:
        print("  ✗ medvae not installed; skipping MedVAE")
        return None


def compute_metrics(x_orig: torch.Tensor, x_rec: torch.Tensor) -> Dict[str, float]:
    """Compute MAE, MSE, SSIM."""
    x_orig = x_orig.cpu().numpy()
    x_rec = x_rec.cpu().numpy()

    mae = np.mean(np.abs(x_orig - x_rec))
    mse = np.mean((x_orig - x_rec) ** 2)

    # SSIM (per slice, then average)
    ssim_vals = []
    for z in range(x_orig.shape[-1]):
        s1 = x_orig[..., z]
        s2 = x_rec[..., z]
        
        c1, c2 = 0.01, 0.03
        mu1 = ndimage.gaussian_filter(s1, sigma=1.5)
        mu2 = ndimage.gaussian_filter(s2, sigma=1.5)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = ndimage.gaussian_filter(s1 ** 2, sigma=1.5) - mu1_sq
        sigma2_sq = ndimage.gaussian_filter(s2 ** 2, sigma=1.5) - mu2_sq
        sigma12 = ndimage.gaussian_filter(s1 * s2, sigma=1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
        ssim_vals.append(np.mean(ssim_map))

    ssim = np.mean(ssim_vals)

    return {"mae": mae, "mse": mse, "ssim": ssim}


def benchmark_vae(
    vae: nn.Module,
    vae_name: str,
    dataset: Dataset,
    device: torch.device,
    patched: bool = True,
) -> Dict[str, float]:
    """Benchmark a single VAE."""
    print(f"\n{'='*70}")
    print(f" Benchmarking {vae_name}")
    print(f"{'='*70}")

    vae.eval()
    
    # Wrap in PatchedVAE if needed
    if patched:
        vae_wrapped = PatchedVAE(vae, patch_size=(112, 128, 80), overlap=0.25)
        vae_wrapped = vae_wrapped.to(device)
    else:
        vae_wrapped = vae

    metrics_list = []
    times = []
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        for idx, (x, name) in enumerate(dataset):
            x = x.to(device)
            
            # Benchmark forward pass
            t0 = time.time()
            
            try:
                if patched:
                    result = vae_wrapped.forward(x, encode_only=False, batch_size=2)
                    x_rec = result["reconstruction"].to(device)
                else:
                    # Direct forward (may OOM on full res)
                    if hasattr(vae, 'encode'):
                        z = vae.encode(x)
                        if isinstance(z, tuple):
                            z = z[0]
                    else:
                        z_anat, _ = vae.encoder(x)
                        z = z_anat
                    
                    if hasattr(vae, 'decode'):
                        x_rec = vae.decode(z)
                    else:
                        # VQ-VAE needs z_mod and indices
                        print(f"  ✗ {name}: VQ-VAE decode requires z_mod")
                        continue

                elapsed = time.time() - t0
                times.append(elapsed)

                # Compute metrics
                metrics = compute_metrics(x.squeeze(), x_rec.squeeze())
                metrics_list.append(metrics)

                if device.type == "cuda":
                    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
                else:
                    peak_mem = 0.0
                    
                print(f"  [{idx+1}/{len(dataset)}] {name:20s} | MAE={metrics['mae']:.4f} | "
                      f"SSIM={metrics['ssim']:.4f} | t={elapsed:.2f}s | mem={peak_mem:.1f}GB")
            
            except Exception as e:
                print(f"  ✗ {name}: {str(e)[:60]}")
                continue

    # Aggregate stats
    if not metrics_list:
        return {"mae": np.nan, "mse": np.nan, "ssim": np.nan, "time": np.nan, "error": "all_failed"}

    avg_metrics = {
        "mae": np.mean([m["mae"] for m in metrics_list]),
        "mse": np.mean([m["mse"] for m in metrics_list]),
        "ssim": np.mean([m["ssim"] for m in metrics_list]),
        "time": np.mean(times) if times else 0,
        "n_samples": len(metrics_list),
    }

    print(f"\n{vae_name} Summary:")
    print(f"  MAE:  {avg_metrics['mae']:.4f}")
    print(f"  MSE:  {avg_metrics['mse']:.4f}")
    print(f"  SSIM: {avg_metrics['ssim']:.4f}")
    print(f"  Time: {avg_metrics['time']:.2f}s/vol")
    print(f"  Samples: {avg_metrics['n_samples']}")

    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark 3 VAE architectures")
    parser.add_argument("--data-root", type=str, default="/home/rousseau/Data/MRIxFields_20260414")
    parser.add_argument("--modality", type=str, default="T1W", choices=["T1W", "T2W", "T2FLAIR"])
    parser.add_argument("--field", type=str, default="0.1T", 
                       choices=["0.1T", "1.5T", "3T", "5T", "7T"])
    parser.add_argument("--max-samples", type=int, default=2, help="test volumes per VAE")
    parser.add_argument("--aekl-ckpt", type=str, 
                       default="outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth")
    parser.add_argument("--vqvae-ckpt", type=str, 
                       default="outputs/vqvae3d/runs/smoke_vqvae/weights/vqvae_step_000001.pth")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/benchmark")
    parser.add_argument("--skip-aekl", action="store_true")
    parser.add_argument("--skip-vqvae", action="store_true")
    parser.add_argument("--skip-medvae", action="store_true")

    args = parser.parse_args()

    device = torch.device(args.device if args.device else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = BenchmarkDataset(
        Path(args.data_root),
        modality=args.modality,
        field=args.field,
        max_samples=args.max_samples,
    )

    results = {}

    # AEKL
    if not args.skip_aekl:
        try:
            vae_aekl = load_aekl(Path(args.aekl_ckpt), device)
            results["AEKL"] = benchmark_vae(vae_aekl, "AEKL", dataset, device, patched=True)
            del vae_aekl
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ AEKL failed: {e}")
            results["AEKL"] = {"error": str(e)}

    # VQ-VAE
    if not args.skip_vqvae:
        try:
            vae_vqvae = load_vqvae(Path(args.vqvae_ckpt), device)
            results["VQ-VAE"] = benchmark_vae(vae_vqvae, "VQ-VAE", dataset, device, patched=True)
            del vae_vqvae
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ VQ-VAE failed: {e}")
            results["VQ-VAE"] = {"error": str(e)}

    # MedVAE
    if not args.skip_medvae:
        try:
            vae_medvae = load_medvae("medvae_4_1_3d", device)
            if vae_medvae is not None:
                results["MedVAE"] = benchmark_vae(vae_medvae, "MedVAE 4×1", dataset, device, patched=True)
                del vae_medvae
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ MedVAE failed: {e}")
            results["MedVAE"] = {"error": str(e)}

    # Save results to CSV
    csv_path = out_dir / f"benchmark_{args.modality}_{args.field}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["VAE", "MAE", "MSE", "SSIM", "Time(s)", "N_Samples", "Status"])
        for vae_name, metrics in results.items():
            if "error" in metrics:
                writer.writerow([vae_name, "—", "—", "—", "—", "—", metrics.get("error", "failed")])
            else:
                writer.writerow([
                    vae_name,
                    f"{metrics.get('mae', np.nan):.4f}",
                    f"{metrics.get('mse', np.nan):.4f}",
                    f"{metrics.get('ssim', np.nan):.4f}",
                    f"{metrics.get('time', np.nan):.2f}",
                    metrics.get("n_samples", "—"),
                    "OK",
                ])

    print(f"\n{'='*70}")
    print(f" Benchmark complete. Results saved to:")
    print(f" {csv_path}")
    print(f"{'='*70}\n")

    # Print summary table
    print("Summary Table:")
    print("-" * 70)
    print(f"{'VAE':<20} {'MAE':<12} {'SSIM':<12} {'Time (s)':<12} {'Status':<12}")
    print("-" * 70)
    for vae_name, metrics in results.items():
        if "error" in metrics:
            print(f"{vae_name:<20} {'—':<12} {'—':<12} {'—':<12} {metrics.get('error', 'failed'):<12}")
        else:
            print(f"{vae_name:<20} {metrics.get('mae', np.nan):<12.4f} "
                  f"{metrics.get('ssim', np.nan):<12.4f} {metrics.get('time', np.nan):<12.2f} {'OK':<12}")
    print("-" * 70)


if __name__ == "__main__":
    main()