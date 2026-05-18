#!/usr/bin/env python3
"""
Benchmark script for 4 VAE architectures on MRIxFields full-resolution volumes.

Compares:
1. AutoencoderKL (MONAI) — AEKL
2. VQ-VAE (NeuroQuant-inspired, hybrid)
3. MedVAE 3D frozen (pre-trained on 1M medical images, poids HuggingFace originaux)
4. MedVAE 3D fine-tuné (poids adaptés sur MRIxFields via train_vae.py)

Metrics: MAE, SSIM, LPIPS, memory usage, inference time per domain (field strength).

Usage:
  # Benchmark complet (4 VAE)
  python src/vae3d/benchmark_vae.py --modality T1W --field 3T

  # Avec chemins explicites
  python src/vae3d/benchmark_vae.py \\
      --aekl-ckpt outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth \\
      --vqvae-ckpt outputs/vqvae3d/runs/vqvae_full/weights/vqvae_final.pth \\
      --medvae-finetuned-ckpt outputs/medvae/runs/medvae_T1W/weights/model_final.pth

  # Sauter un VAE
  python src/vae3d/benchmark_vae.py --skip-vqvae --skip-medvae-finetuned
"""

import argparse
import csv
import re
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from scipy.ndimage import zoom as scipy_zoom
from torch.utils.data import DataLoader, Dataset

# Import VAE architectures
from train_vae_3d import build_vae as build_aekl
from train_vqvae import NeuroQuantHybrid

from utils.patched_vae import PatchedVAE

FILE_RE = re.compile(r"^[A-Z]_([A-Z0-9]+)_([0-9.]+T)_(\d+)\.nii\.gz$")


def _normalize(
    vol: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5
) -> np.ndarray:
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

    if checkpoint_path is None:
        return None
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return None
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


def load_vqvae(
    checkpoint_path: Path,
    device: torch.device,
    n_modalities: int = 3,
    n_fields: int = 5,
) -> nn.Module:
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

    if checkpoint_path is None:
        return None
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        return None
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt["model"] if (isinstance(ckpt, dict) and "model" in ckpt) else ckpt

    # Handle field_emb mismatch (checkpoint may have different n_fields)
    decoder_state = {k: v for k, v in state.items() if k.startswith("decoder.")}
    for k in list(decoder_state.keys()):
        if "field_emb" in k:
            # Keep only the checkpoint's field_emb, ignore shape mismatch
            old_shape = decoder_state[k].shape
            new_shape = (
                model.state_dict()[k].shape if k in model.state_dict() else old_shape
            )
            if old_shape != new_shape:
                print(f"  ⚠️  Skipping {k}: shape mismatch {old_shape} vs {new_shape}")
                del state[k]

    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"  → VQ-VAE loaded from {checkpoint_path.name}")
    return model


def load_medvae(
    model_name: str = "medvae_4_1_3d",
    device: torch.device = None,
    checkpoint_path: Optional[Path] = None,
) -> Optional[nn.Module]:
    """
    Load MedVAE (frozen depuis HuggingFace, ou fine-tuné depuis un checkpoint local).

    Args:
        model_name:       Nom du modèle HuggingFace (ex : "medvae_4_1_3d").
        device:           Device cible.
        checkpoint_path:  Si fourni, charge les poids depuis ce .pth local
                          (fine-tuning) en lieu et place des poids HuggingFace.
                          Le fichier doit contenir le state_dict complet de
                          l'objet MVAE (clés "model.encoder…", "model.decoder…").
                          Si None → poids HuggingFace originaux (mode frozen).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label = "(fine-tuné)" if checkpoint_path is not None else "(frozen)"
    print(f"[MedVAE {label}] Loading {model_name}...")

    try:
        from medvae import MVAE
    except ImportError:
        print("  ✗ medvae non installé — pip install medvae")
        return None

    # Instanciation de l'architecture avec les poids HuggingFace
    try:
        model = MVAE(model_name=model_name, modality="mri").to(device)
    except Exception as e:
        print(f"  ✗ Impossible d'instancier MVAE : {e}")
        return None

    if checkpoint_path is not None:
        # --- Mode fine-tuné : écrase les poids HF par le checkpoint local ---
        cp = Path(checkpoint_path)
        if not cp.exists():
            print(f"  ✗ Checkpoint introuvable : {cp}")
            return None

        raw = torch.load(cp, map_location=device, weights_only=False)

        # Accepte plusieurs formats de sauvegarde :
        #   1. state_dict direct de MVAE  (clés "model.encoder…") — train_vae.py
        #   2. dict avec clé "model_state" ou "model"
        if isinstance(raw, dict):
            if "model_state" in raw:
                state = raw["model_state"]
            elif "model" in raw and isinstance(raw["model"], dict):
                state = raw["model"]
            else:
                # Suppose que c'est directement le state_dict
                state = raw
        else:
            print(f"  ✗ Format de checkpoint inattendu : {type(raw)}")
            return None

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  ⚠  {len(missing)} clé(s) manquante(s) dans le checkpoint")
        if unexpected:
            print(f"  ⚠  {len(unexpected)} clé(s) inattendues dans le checkpoint")
        if not missing and not unexpected:
            print(f"  ✓ Chargement parfait (0 missing, 0 unexpected)")
        print(f"  → MedVAE fine-tuné chargé depuis {cp.name}")
    else:
        # --- Mode frozen : poids HuggingFace déjà chargés par MVAE() ---
        print(f"  → MedVAE frozen chargé depuis HuggingFace ({model_name})")

    model.eval()
    return model


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
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = ndimage.gaussian_filter(s1**2, sigma=1.5) - mu1_sq
        sigma2_sq = ndimage.gaussian_filter(s2**2, sigma=1.5) - mu2_sq
        sigma12 = ndimage.gaussian_filter(s1 * s2, sigma=1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
        )
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
    print(f"\n{'=' * 70}")
    print(f" Benchmarking {vae_name}")
    print(f"{'=' * 70}")

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
                    if hasattr(vae, "encode"):
                        z = vae.encode(x)
                        if isinstance(z, tuple):
                            z = z[0]
                    else:
                        z_anat, _ = vae.encoder(x)
                        z = z_anat

                    if hasattr(vae, "decode"):
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

                print(
                    f"  [{idx + 1}/{len(dataset)}] {name:20s} | MAE={metrics['mae']:.4f} | "
                    f"SSIM={metrics['ssim']:.4f} | t={elapsed:.2f}s | mem={peak_mem:.1f}GB"
                )

            except Exception as e:
                print(f"  ✗ {name}: {str(e)[:60]}")
                continue

    # Aggregate stats
    if not metrics_list:
        return {
            "mae": np.nan,
            "mse": np.nan,
            "ssim": np.nan,
            "time": np.nan,
            "error": "all_failed",
        }

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
    parser = argparse.ArgumentParser(
        description="Benchmark 4 VAE architectures (AEKL, VQ-VAE, MedVAE frozen, MedVAE fine-tuné)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Données ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-root", type=str, default="/home/rousseau/Data/MRIxFields_20260414"
    )
    parser.add_argument(
        "--modality", type=str, default="T1W", choices=["T1W", "T2W", "T2FLAIR"]
    )
    parser.add_argument(
        "--field", type=str, default="0.1T", choices=["0.1T", "1.5T", "3T", "5T", "7T"]
    )
    parser.add_argument(
        "--max-samples", type=int, default=2, help="Nombre de volumes de test par VAE"
    )

    # ── Checkpoints ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--aekl-ckpt",
        type=str,
        default="outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth",
        help="Checkpoint AEKL (.pth)",
    )
    parser.add_argument(
        "--vqvae-ckpt",
        type=str,
        default="outputs/vqvae3d/runs/vqvae_full/weights/vqvae_final.pth",
        help="Checkpoint VQ-VAE (.pth)",
    )
    parser.add_argument(
        "--medvae-model-name",
        type=str,
        default="medvae_4_1_3d",
        help="Nom du modèle MedVAE HuggingFace (frozen et fine-tuné)",
    )
    parser.add_argument(
        "--medvae-finetuned-ckpt",
        type=str,
        default="outputs/medvae/runs/medvae_T1W/weights/model_final.pth",
        help="Checkpoint MedVAE fine-tuné (.pth). "
        "Doit contenir le state_dict complet de l'objet MVAE "
        '(clés "model.encoder…", "model.decoder…").',
    )

    # ── Flags skip ───────────────────────────────────────────────────────────
    parser.add_argument("--skip-aekl", action="store_true", help="Ignorer AEKL")
    parser.add_argument("--skip-vqvae", action="store_true", help="Ignorer VQ-VAE")
    parser.add_argument(
        "--skip-medvae", action="store_true", help="Ignorer MedVAE frozen"
    )
    parser.add_argument(
        "--skip-medvae-finetuned", action="store_true", help="Ignorer MedVAE fine-tuné"
    )

    # ── Misc ─────────────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/benchmark")

    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = BenchmarkDataset(
        Path(args.data_root),
        modality=args.modality,
        field=args.field,
        max_samples=args.max_samples,
    )

    results = {}

    # ── 1. AEKL ──────────────────────────────────────────────────────────────
    if not args.skip_aekl:
        try:
            vae = load_aekl(Path(args.aekl_ckpt), device)
            results["AEKL"] = benchmark_vae(vae, "AEKL", dataset, device, patched=True)
            del vae
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ AEKL failed: {e}")
            results["AEKL"] = {"error": str(e)}

    # ── 2. VQ-VAE ────────────────────────────────────────────────────────────
    if not args.skip_vqvae:
        try:
            vae = load_vqvae(Path(args.vqvae_ckpt), device)
            results["VQ-VAE"] = benchmark_vae(
                vae, "VQ-VAE", dataset, device, patched=True
            )
            del vae
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ VQ-VAE failed: {e}")
            results["VQ-VAE"] = {"error": str(e)}

    # ── 3. MedVAE frozen ─────────────────────────────────────────────────────
    if not args.skip_medvae:
        try:
            vae = load_medvae(
                model_name=args.medvae_model_name,
                device=device,
                checkpoint_path=None,  # poids HuggingFace originaux
            )
            if vae is not None:
                results["MedVAE (frozen)"] = benchmark_vae(
                    vae, "MedVAE (frozen)", dataset, device, patched=True
                )
                del vae
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ MedVAE frozen failed: {e}")
            results["MedVAE (frozen)"] = {"error": str(e)}

    # ── 4. MedVAE fine-tuné ──────────────────────────────────────────────────
    if not args.skip_medvae_finetuned:
        ckpt = Path(args.medvae_finetuned_ckpt)
        if not ckpt.exists():
            print(f"⚠  MedVAE fine-tuné : checkpoint introuvable ({ckpt})")
            print(
                "   Entraîner d'abord : sbatch src/slurm/train_vae_jeanzay.slurm medvae"
            )
            results["MedVAE (fine-tuné)"] = {"error": "checkpoint_not_found"}
        else:
            try:
                vae = load_medvae(
                    model_name=args.medvae_model_name,
                    device=device,
                    checkpoint_path=ckpt,  # poids fine-tunés locaux
                )
                if vae is not None:
                    results["MedVAE (fine-tuné)"] = benchmark_vae(
                        vae, "MedVAE (fine-tuné)", dataset, device, patched=True
                    )
                    del vae
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"✗ MedVAE fine-tuné failed: {e}")
                results["MedVAE (fine-tuné)"] = {"error": str(e)}

    # ── Sauvegarde CSV ───────────────────────────────────────────────────────
    csv_path = out_dir / f"benchmark_{args.modality}_{args.field}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["VAE", "MAE", "MSE", "SSIM", "Time(s)", "N_Samples", "Status"])
        for vae_name, metrics in results.items():
            if "error" in metrics:
                writer.writerow(
                    [vae_name, "—", "—", "—", "—", "—", metrics.get("error", "failed")]
                )
            else:
                writer.writerow(
                    [
                        vae_name,
                        f"{metrics.get('mae', np.nan):.4f}",
                        f"{metrics.get('mse', np.nan):.4f}",
                        f"{metrics.get('ssim', np.nan):.4f}",
                        f"{metrics.get('time', np.nan):.2f}",
                        metrics.get("n_samples", "—"),
                        "OK",
                    ]
                )

    print(f"\n{'=' * 70}")
    print(f" Benchmark complet — résultats sauvegardés :")
    print(f" {csv_path}")
    print(f"{'=' * 70}\n")

    # ── Tableau récapitulatif ────────────────────────────────────────────────
    col_w = 24
    print(f"{'VAE':<{col_w}} {'MAE':<12} {'SSIM':<12} {'Time (s)':<12} {'Status'}")
    print("-" * 72)
    for vae_name, metrics in results.items():
        if "error" in metrics:
            print(
                f"{vae_name:<{col_w}} {'—':<12} {'—':<12} {'—':<12} "
                f"{metrics.get('error', 'failed')}"
            )
        else:
            print(
                f"{vae_name:<{col_w}} "
                f"{metrics.get('mae', np.nan):<12.4f} "
                f"{metrics.get('ssim', np.nan):<12.4f} "
                f"{metrics.get('time', np.nan):<12.2f} OK"
            )
    print("-" * 72)


if __name__ == "__main__":
    main()
