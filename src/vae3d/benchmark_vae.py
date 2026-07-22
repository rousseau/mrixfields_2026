#!/usr/bin/env python3
"""Benchmark script for VAE architectures — reconstruction quality on prospective data.

Evaluates encode-decode cycle for all VAE architectures on Training_prospective
subjects (0006, 0007, 0009) across all modalities and field strengths.

Metrics : MAE, MSE, SSIM (slice-wise axial), nRMSE, LPIPS (slice-wise axial)
Output  : results/benchmark_vae/metrics/benchmark_results.csv
          one row per (vae, modality, field, subject)

Usage:
    PYTHONPATH=src python src/vae3d/benchmark_vae.py [--device cuda] [--skip-lpips]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Ensure project-local imports work when running:
#   python src/vae3d/benchmark_vae.py
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from common.io import DOMAINS, MODALITIES, normalize_volume
from common.metrics import compute_nrmse, compute_ssim, compute_lpips
from models.vae_loader import load_vae
from utils.patched_vae import PatchedVAE
from models.registry import VAE_REGISTRY, PROSPECTIVE_SUBJECTS, RHVAE_VOLUME_SIZE, PATCH_SIZE, PATCH_OVERLAP

# VAE Registry moved to src/models/registry.py

# RHVAE has a vectorial latent — PatchedVAE is incompatible.
# For RHVAE we crop/pad the volume to a fixed size and do a single forward pass.
# These constants are now managed in src/models/registry.py

DATA_ROOT_DEFAULT = "/home/rousseau/Data/MRIxFields_20260414"
PROJECT_ROOT = Path(__file__).resolve().parents[2]



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_prospective_volume(
    data_root: Path,
    modality: str,
    field: str,
    subject_id: str,
) -> Optional[np.ndarray]:
    """Load, normalize and return a prospective volume as float32 numpy in [0,1]."""
    data_root = Path(data_root)
    vol_dir = data_root / "Training_prospective" / modality / field
    if not vol_dir.exists():
        return None
    candidates = sorted(vol_dir.glob(f"*_{subject_id}.nii.gz"))
    if not candidates:
        return None

    import nibabel as nib
    nii = nib.load(str(candidates[0]))
    vol = nii.get_fdata(dtype=np.float32)

    # Percentile normalization → [0, 1]
    p_lo, p_hi = np.percentile(vol, 0.5), np.percentile(vol, 99.5)
    vol = np.clip(vol, p_lo, p_hi)
    vol = (vol - p_lo) / (p_hi - p_lo + 1e-8)
    return vol.astype(np.float32)


def crop_or_pad(vol: np.ndarray, target: tuple) -> np.ndarray:
    """Center-crop or zero-pad a volume to target (H, W, D)."""
    out = vol
    for axis, t in enumerate(target):
        s = out.shape[axis]
        if s > t:
            start = (s - t) // 2
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(start, start + t)
            out = out[tuple(sl)]
        elif s < t:
            pad = [(0, 0)] * out.ndim
            before = (t - s) // 2
            pad[axis] = (before, t - s - before)
            out = np.pad(out, pad)
    return out


def save_reconstruction_figure(
    orig: np.ndarray,
    recon: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save an incremental qualitative figure (orig vs recon, 3 orthogonal views)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w, d = orig.shape
    mh, mw, md = h // 2, w // 2, d // 2

    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    fig.suptitle(title, fontsize=11)

    views = [
        (orig[mh, :, :], recon[mh, :, :], "Axial"),
        (orig[:, mw, :], recon[:, mw, :], "Coronal"),
        (orig[:, :, md], recon[:, :, md], "Sagittal"),
    ]

    for i, (o_slice, r_slice, name) in enumerate(views):
        axes[i, 0].imshow(o_slice, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 0].set_title(f"{name} - original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(r_slice, cmap="gray", vmin=0.0, vmax=1.0)
        axes[i, 1].set_title(f"{name} - recon")
        axes[i, 1].axis("off")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-volume inference
# ---------------------------------------------------------------------------

def encode_decode(
    vae: nn.Module,
    vol: np.ndarray,
    device: torch.device,
    use_patched: bool,
) -> np.ndarray:
    """Run encode-decode on a volume; return reconstruction as numpy float32 in [0,1]."""
    if use_patched:
        vae_wrapped = PatchedVAE(vae, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP)
        vae_wrapped = vae_wrapped.to(device)
        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            result = vae_wrapped.forward(x, encode_only=False, batch_size=1)
        xhat = result["reconstruction"].squeeze().cpu().numpy()
    else:
        # Single forward pass for all non-patched inference (spatial + vector latents)
        vol_crop = crop_or_pad(vol, RHVAE_VOLUME_SIZE)
        x = torch.from_numpy(vol_crop).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            z = vae.encode(x)
            xhat = vae.decode(z).squeeze().cpu().numpy()
        # Reverse crop: pad back to original vol shape for fair metric computation
        vol = vol_crop  # compare against the cropped version for consistency

    # Clip to [0, 1]
    xhat = np.clip(xhat, 0.0, 1.0)
    return vol, xhat


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    orig: np.ndarray,
    recon: np.ndarray,
    device: torch.device,
    skip_lpips: bool = False,
) -> Dict[str, float]:
    mae   = float(np.mean(np.abs(orig - recon)))
    mse   = float(np.mean((orig - recon) ** 2))
    ssim  = compute_ssim(recon, orig, slice_axis=2)
    nrmse = compute_nrmse(recon, orig)
    lpips = (
        compute_lpips(recon, orig, device=device.type, slice_axis=2)
        if not skip_lpips
        else float("nan")
    )
    return {"mae": mae, "mse": mse, "ssim": ssim, "nrmse": nrmse, "lpips": lpips}


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    modalities: List[str],
    fields: List[str],
    subjects: List[str],
    data_root: Path,
    output_dir: Path,
    device: torch.device,
    skip_lpips: bool,
    vae_names: Optional[List[str]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"

    fieldnames = [
        "vae", "modality", "field", "subject",
        "mae", "mse", "ssim", "nrmse", "lpips",
        "time_s", "partial", "epoch_info",
    ]

    # Open CSV (append mode so re-runs add rows without wiping previous data)
    existing_keys = set()
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                existing_keys.add((row["vae"], row["modality"], row["field"], row["subject"]))

    f_out = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if not existing_keys:
        writer.writeheader()

    for vae_name, vae_cfg, partial, epoch_info in VAE_REGISTRY:
        if vae_names and vae_name not in vae_names:
            continue

        print(f"\n{'='*70}")
        print(f"  VAE: {vae_name}  (partial={partial}, {epoch_info})")
        print(f"{'='*70}")

        # Load VAE once for all volumes
        try:
            vae = load_vae(vae_cfg, device)
        except Exception as e:
            print(f"  ✗ Failed to load {vae_name}: {e}")
            traceback.print_exc()
            continue

        is_rhvae = (vae.latent_format == "vector")
        use_patched = not is_rhvae

        vae.eval()

        for modality in modalities:
            for field in fields:
                for subject in subjects:
                    # --- Visual check: skip if image already exists ---
                    visual_dir = output_dir.parent / "benchmark_visuals" / vae_name / modality / field
                    visual_path = visual_dir / f"{subject}.png"
                    
                    key = (vae_name, modality, field, subject)
                    if key in existing_keys and visual_path.exists():
                        print(f"  skip {modality}/{field}/sub{subject} (already processed)")
                        continue

                    vol = load_prospective_volume(data_root, modality, field, subject)
                    if vol is None:
                        print(f"  ✗ No volume: {modality}/{field}/sub{subject}")
                        continue

                    t0 = time.time()
                    try:
                        orig, recon = encode_decode(vae, vol, device, use_patched)
                        elapsed = time.time() - t0
                        
                        # --- Generate and save figure immediately ---
                        title = f"{vae_name} | {modality} | {field} | Sub {subject}"
                        save_reconstruction_figure(orig, recon, visual_path, title)
                        
                        m = compute_all_metrics(orig, recon, device, skip_lpips)
                        print(
                            f"  {modality}/{field}/sub{subject}: "
                            f"MAE={m['mae']:.4f} SSIM={m['ssim']:.4f} "
                            f"nRMSE={m['nrmse']:.4f} LPIPS={m['lpips']:.4f} "
                            f"t={elapsed:.1f}s | Image saved"
                        )
                        writer.writerow({
                            "vae": vae_name,
                            "modality": modality,
                            "field": field,
                            "subject": subject,
                            **{k: f"{v:.5f}" if not np.isnan(v) else "nan" for k, v in m.items()},
                            "time_s": f"{elapsed:.2f}",
                            "partial": str(partial),
                            "epoch_info": epoch_info,
                        })
                        f_out.flush()
                        existing_keys.add(key)
                    except Exception as e:
                        print(f"  ✗ {modality}/{field}/sub{subject}: {e}")
                        traceback.print_exc()

        del vae
        torch.cuda.empty_cache()

    f_out.close()
    print(f"\n{'='*70}")
    print(f"  Résultats sauvegardés : {csv_path}")
    print(f"{'='*70}\n")

    # Print summary table
    _print_summary(csv_path)


def _print_summary(csv_path: Path) -> None:
    """Print mean metrics per VAE averaged over all (modality, field, subject)."""
    from collections import defaultdict
    rows = defaultdict(list)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            vae = row["vae"]
            for metric in ["mae", "ssim", "nrmse", "lpips"]:
                try:
                    v = float(row[metric])
                    if not np.isnan(v):
                        rows[(vae, metric)].append(v)
                except (ValueError, KeyError):
                    pass

    all_vaes = sorted({k[0] for k in rows})
    header = f"{'VAE':<25} {'MAE':>8} {'SSIM':>8} {'nRMSE':>8} {'LPIPS':>8}"
    print("\n" + "=" * 65)
    print("  BENCHMARK SUMMARY (mean over all modalities/fields/subjects)")
    print("=" * 65)
    print(header)
    print("-" * 65)
    for vae in all_vaes:
        vals = {m: np.mean(rows[(vae, m)]) if rows[(vae, m)] else float("nan")
                for m in ["mae", "ssim", "nrmse", "lpips"]}
        print(f"  {vae:<23} {vals['mae']:>8.4f} {vals['ssim']:>8.4f} "
              f"{vals['nrmse']:>8.4f} {vals['lpips']:>8.4f}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VAE reconstruction on prospective data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT,
                        help="Root data directory (contains Training_prospective/)")
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    parser.add_argument("--fields", nargs="+", default=list(DOMAINS))
    parser.add_argument("--subjects", nargs="+", default=PROSPECTIVE_SUBJECTS)
    parser.add_argument("--output-dir", default="results/benchmark_vae/metrics")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-lpips", action="store_true",
                        help="Skip LPIPS computation (fast debug mode)")
    parser.add_argument("--vae", nargs="+", default=None,
                        help="Run only these VAEs (e.g. --vae NV_Generate MedVAE_frozen)")
    args = parser.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")
    print(f"VAEs à évaluer: {[v[0] for v in VAE_REGISTRY] if not args.vae else args.vae}")
    print(f"Modalités: {args.modalities}")
    print(f"Champs: {args.fields}")
    print(f"Sujets: {args.subjects}")

    run_benchmark(
        modalities=args.modalities,
        fields=args.fields,
        subjects=args.subjects,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        device=device,
        skip_lpips=args.skip_lpips,
        vae_names=args.vae,
    )


if __name__ == "__main__":
    main()
