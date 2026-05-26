#!/usr/bin/env python3
"""Benchmark script for VAE architectures on MRIxFields full-resolution volumes.

Refactored to use common infrastructure (common.io, common.metrics, models.vae_loader).
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common.io import (
    DOMAINS,
    MODALITIES,
    center_crop_or_pad_np,
    load_nifti_volume,
    normalize_volume,
)
from common.metrics import compute_nrmse, compute_ssim
from models.vae_loader import load_vae
from utils.patched_vae import PatchedVAE

# Index mappings pour le VQ-VAE (FiLM conditioning)
MODALITIES_IDX = {"T1W": 0, "T2W": 1, "T2FLAIR": 2}
FIELDS_IDX = {"0.1T": 0, "1.5T": 1, "3T": 2, "5T": 3, "7T": 4}


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

        from common.io import SPLIT_MAP
        split_dir = SPLIT_MAP.get(split, split)
        d = self.data_root / split_dir / modality / field

        for p in sorted(d.glob("*.nii.gz"))[:max_samples]:
            if p.name.startswith(("R_", "P_")):
                self.samples.append(p)

        print(f"BenchmarkDataset: {len(self.samples)} volumes ({modality}/{field})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        vol, _ = load_nifti_volume(path, normalize=True)
        return torch.from_numpy(vol).unsqueeze(0), path.stem


def compute_metrics(x_orig: torch.Tensor, x_rec: torch.Tensor) -> Dict[str, float]:
    """Compute MAE, MSE, SSIM."""
    x_orig_n = x_orig.cpu().numpy()
    x_rec_n = x_rec.cpu().numpy()

    mae = float(np.mean(np.abs(x_orig_n - x_rec_n)))
    mse = float(np.mean((x_orig_n - x_rec_n) ** 2))
    ssim = compute_ssim(x_rec_n, x_orig_n)

    return {"mae": mae, "mse": mse, "ssim": ssim}


def benchmark_vae(
    vae: nn.Module,
    vae_name: str,
    dataset: Dataset,
    device: torch.device,
    patched: bool = True,
    mod_idx: int = 0,
    field_idx: int = 0,
) -> Dict[str, float]:
    """Benchmark a single VAE."""
    print(f"\n{'=' * 70}")
    print(f" Benchmarking {vae_name}")
    print(f"{'=' * 70}")

    vae.eval()

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
            if x.dim() == 4:
                x = x.unsqueeze(0)

            t0 = time.time()

            try:
                if patched and hasattr(vae_wrapped, "forward"):
                    result = vae_wrapped.forward(x, encode_only=False, batch_size=2)
                    x_rec = result["reconstruction"].to(device)
                else:
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
                        print(f"  ✗ {name}: decode not available")
                        continue

                elapsed = time.time() - t0
                times.append(elapsed)

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


def _build_vae_from_args(vae_type: str, args) -> Optional[nn.Module]:
    """Build a VAE config dict and load via unified loader."""
    if vae_type == "aekl":
        cfg = {
            "vae": {
                "vae_type": "aekl",
                "source": "local" if args.aekl_ckpt else "random",
                "checkpoint": args.aekl_ckpt,
                "vae_config": args.vae_config,
            }
        }
    elif vae_type == "vqvae":
        cfg = {
            "vae": {
                "vae_type": "vqvae",
                "source": "local",
                "checkpoint": args.vqvae_ckpt,
                "vae_config": args.vqvae_config,
            }
        }
    elif vae_type == "medvae_frozen":
        cfg = {
            "vae": {
                "vae_type": "medvae",
                "source": "frozen",
                "model_name": args.medvae_model_name,
            }
        }
    elif vae_type == "medvae_finetuned":
        cfg = {
            "vae": {
                "vae_type": "medvae",
                "source": "local",
                "model_name": args.medvae_model_name,
                "checkpoint": args.medvae_finetuned_ckpt,
            }
        }
    elif vae_type == "medvae_disentangle_v1":
        cfg = {
            "vae": {
                "vae_type": "medvae_disentangle",
                "source": "local",
                "model_name": args.medvae_model_name,
                "checkpoint": args.medvae_disentangle_v1_ckpt,
            }
        }
    else:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_vae(cfg, device)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VAE architectures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--data-root", type=str, default="/home/rousseau/Data/MRIxFields_20260414")
    parser.add_argument("--modality", type=str, default="T1W", choices=list(MODALITIES))
    parser.add_argument("--field", type=str, default="0.1T", choices=list(DOMAINS))
    parser.add_argument("--max-samples", type=int, default=2)

    parser.add_argument("--aekl-ckpt", type=str, default="outputs/vae3d/runs/vae3d_T1W/weights/model_final.pth")
    parser.add_argument("--vae-config", type=str, default="configs/vae3d_T1W.yaml")
    parser.add_argument("--vqvae-ckpt", type=str, default="outputs/vqvae3d/runs/vqvae_final/weights/model_best.pth")
    parser.add_argument("--vqvae-config", type=str, default="configs/vqvae3d_T1W.yaml")
    parser.add_argument("--medvae-model-name", type=str, default="medvae_4_1_3d")
    parser.add_argument(
        "--medvae-finetuned-ckpt", type=str,
        default="outputs/medvae/runs/medvae_finetune_all/weights/model_final.pth",
    )
    parser.add_argument(
        "--medvae-disentangle-v1-ckpt", type=str,
        default="outputs/medvae_disentangle_v1/runs/medvae_disentangle_v1/weights/model_best.pth",
    )

    parser.add_argument("--skip-aekl", action="store_true")
    parser.add_argument("--skip-vqvae", action="store_true")
    parser.add_argument("--skip-medvae", action="store_true")
    parser.add_argument("--skip-medvae-finetuned", action="store_true")
    parser.add_argument("--skip-medvae-disentangle-v1", action="store_true")

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/benchmark")

    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = BenchmarkDataset(
        Path(args.data_root),
        modality=args.modality,
        field=args.field,
        max_samples=args.max_samples,
    )

    results = {}
    mod_idx = MODALITIES_IDX.get(args.modality, 0)
    field_idx = FIELDS_IDX.get(args.field, 0)
    print(f"VQ-VAE conditioning: modality={args.modality} (idx={mod_idx}), field={args.field} (idx={field_idx})")

    vae_configs = [
        ("AEKL", "aekl", args.skip_aekl),
        ("VQ-VAE", "vqvae", args.skip_vqvae),
        ("MedVAE (frozen)", "medvae_frozen", args.skip_medvae),
        ("MedVAE (fine-tuné)", "medvae_finetuned", args.skip_medvae_finetuned),
        ("MedVAE disentanglement v1", "medvae_disentangle_v1", args.skip_medvae_disentangle_v1),
    ]

    for name, vae_type, skip in vae_configs:
        if skip:
            continue
        try:
            vae = _build_vae_from_args(vae_type, args)
            if vae is None:
                continue
            results[name] = benchmark_vae(vae, name, dataset, device, patched=True, mod_idx=mod_idx, field_idx=field_idx)
            del vae
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"error": str(e)}

    # Sauvegarde CSV
    csv_path = out_dir / f"benchmark_{args.modality}_{args.field}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["VAE", "MAE", "MSE", "SSIM", "Time(s)", "N_Samples", "Status"])
        for vae_name, metrics in results.items():
            if "error" in metrics:
                writer.writerow([vae_name, "-", "-", "-", "-", "-", metrics.get("error", "failed")])
            else:
                writer.writerow([
                    vae_name,
                    f"{metrics.get('mae', np.nan):.4f}",
                    f"{metrics.get('mse', np.nan):.4f}",
                    f"{metrics.get('ssim', np.nan):.4f}",
                    f"{metrics.get('time', np.nan):.2f}",
                    metrics.get("n_samples", "-"),
                    "OK",
                ])

    print(f"\n{'=' * 70}")
    print(f" Benchmark complet — résultats sauvegardés : {csv_path}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
