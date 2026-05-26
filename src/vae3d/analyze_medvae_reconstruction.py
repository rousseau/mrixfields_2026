#!/usr/bin/env python3
"""
Analyse de reconstruction MedVAE sur MRIxFields.

Fonctionnalités:
- Prétraitement explicite: resampling isotrope 1mm + normalisation percentile.
- Reconstruction MedVAE en plein volume (prioritaire), fallback patch-based si OOM.
- Métriques globales (MAE, MSE, SSIM) + SSIM par orientation (sag/cor/axial).
- Analyse par type d'image (split, modalité, champ).
- Analyse par type d'artefact inféré (proxy IQM, en l'absence de labels explicites).
- Graphiques exportés (PNG) + CSV détaillé + résumé texte.

Usage:
  python src/vae3d/analyze_medvae_reconstruction.py \
      --env local \
      --splits Training_prospective Training_retrospective \
      --max-per-group 8
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import yaml
from nibabel.orientations import aff2axcodes
from scipy import ndimage
from scipy.ndimage import zoom as scipy_zoom

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))

from utils.patched_vae import PatchedVAE


FILE_SUFFIX = ".nii.gz"
DEFAULT_MODALITIES = ("T1W", "T2W", "T2FLAIR")
DEFAULT_FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")


@dataclass
class SampleRec:
    path: Path
    split: str
    modality: str
    field: str


def load_env_config(env_name: str) -> Dict[str, str]:
    env_path = Path("configs/env") / f"{env_name}.yaml"
    with open(env_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {k: str(v) for k, v in cfg.items()}


def normalize_percentile(vol: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    lo = np.percentile(vol, lo_pct)
    hi = np.percentile(vol, hi_pct)
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    v = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (v * 2.0 - 1.0).astype(np.float32)


def resample_to_1mm(vol: np.ndarray, affine: np.ndarray) -> np.ndarray:
    spacing = np.abs(np.diag(affine)[:3]).astype(np.float32)
    factors = spacing / np.array([1.0, 1.0, 1.0], dtype=np.float32)
    if np.allclose(factors, 1.0, atol=0.05):
        return vol.astype(np.float32)
    return scipy_zoom(vol, factors, order=1).astype(np.float32)


def ssim_2d(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = 0.01, 0.03
    mu1 = ndimage.gaussian_filter(a, sigma=1.5)
    mu2 = ndimage.gaussian_filter(b, sigma=1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = ndimage.gaussian_filter(a * a, sigma=1.5) - mu1_sq
    sigma2_sq = ndimage.gaussian_filter(b * b, sigma=1.5) - mu2_sq
    sigma12 = ndimage.gaussian_filter(a * b, sigma=1.5) - mu12

    num = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8
    return float(np.mean(num / den))


def _sample_indices(n: int, max_slices: int) -> np.ndarray:
    if n <= max_slices:
        return np.arange(n)
    return np.linspace(0, n - 1, max_slices).astype(int)


def ssim_3d(x: np.ndarray, y: np.ndarray, max_slices: int = 32) -> float:
    idxs = _sample_indices(x.shape[2], max_slices=max_slices)
    vals = [ssim_2d(x[:, :, z], y[:, :, z]) for z in idxs]
    return float(np.mean(vals)) if vals else float("nan")


def orientation_ssim(x: np.ndarray, y: np.ndarray, max_slices_per_axis: int = 24) -> Dict[str, float]:
    sag_idx = _sample_indices(x.shape[0], max_slices=max_slices_per_axis)
    cor_idx = _sample_indices(x.shape[1], max_slices=max_slices_per_axis)
    axi_idx = _sample_indices(x.shape[2], max_slices=max_slices_per_axis)
    sag = [ssim_2d(x[i, :, :], y[i, :, :]) for i in sag_idx]
    cor = [ssim_2d(x[:, j, :], y[:, j, :]) for j in cor_idx]
    axi = [ssim_2d(x[:, :, k], y[:, :, k]) for k in axi_idx]
    return {
        "ssim_sagittal": float(np.mean(sag)) if sag else float("nan"),
        "ssim_coronal": float(np.mean(cor)) if cor else float("nan"),
        "ssim_axial": float(np.mean(axi)) if axi else float("nan"),
    }


def infer_artifact_scores(vol: np.ndarray) -> Dict[str, float]:
    # Proxy simple sans labels explicites: bruit, flou, banding/ghosting, non-uniformite.
    hp = vol - ndimage.gaussian_filter(vol, sigma=1.2)
    noise_score = float(np.std(hp))

    gx = ndimage.sobel(vol, axis=0)
    gy = ndimage.sobel(vol, axis=1)
    gz = ndimage.sobel(vol, axis=2)
    sharpness = float(np.mean(np.sqrt(gx * gx + gy * gy + gz * gz) + 1e-8))
    blur_score = 1.0 / (sharpness + 1e-6)

    # Banding/ghosting proxy: variance des moyennes de coupes axiales.
    banding_score = float(np.std(np.mean(vol, axis=(0, 1))))

    # Bias/non-uniformite: energie tres basse frequence relative.
    low = ndimage.gaussian_filter(vol, sigma=18.0)
    bias_score = float(np.std(low) / (np.std(vol) + 1e-6))

    return {
        "noise_score": noise_score,
        "blur_score": blur_score,
        "banding_score": banding_score,
        "bias_score": bias_score,
    }


def classify_artifact_type(all_rows: List[Dict[str, float]]) -> None:
    keys = ["noise_score", "blur_score", "banding_score", "bias_score"]
    means = {k: float(np.mean([r[k] for r in all_rows])) for k in keys}
    stds = {k: float(np.std([r[k] for r in all_rows])) + 1e-8 for k in keys}

    label_map = {
        "noise_score": "noise",
        "blur_score": "blur",
        "banding_score": "banding_ghosting",
        "bias_score": "bias_nonuniformity",
    }

    for r in all_rows:
        z = {k: (r[k] - means[k]) / stds[k] for k in keys}
        max_k = max(z, key=z.get)
        max_v = z[max_k]
        r["artifact_type"] = "clean" if max_v < 0.6 else label_map[max_k]


def collect_samples(
    data_root: Path,
    splits: Sequence[str],
    modalities: Sequence[str],
    fields: Sequence[str],
    max_per_group: int,
) -> List[SampleRec]:
    samples: List[SampleRec] = []
    for split in splits:
        for mod in modalities:
            for field in fields:
                d = data_root / split / mod / field
                if not d.exists():
                    continue
                vols = sorted(d.glob(f"*{FILE_SUFFIX}"))[:max_per_group]
                for p in vols:
                    samples.append(SampleRec(path=p, split=split, modality=mod, field=field))
    return samples


def run_reconstruction(model: torch.nn.Module, x: torch.Tensor, try_full_image: bool, patched: PatchedVAE) -> Tuple[torch.Tensor, str]:
    if try_full_image:
        try:
            with torch.no_grad():
                z = model.encode(x)
                if isinstance(z, tuple):
                    z = z[0]
                rec = model.decode(z)
                if rec.shape[1] > 1:
                    rec = rec[:, :1]
            return rec, "full"
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" not in msg and "cuda" not in msg:
                raise

    with torch.no_grad():
        out = patched.forward(x, encode_only=False, batch_size=2)
        rec = out["reconstruction"].unsqueeze(0).unsqueeze(0)
    return rec, "patch"


def plot_box_by_category(rows: List[Dict[str, float]], key: str, value: str, title: str, out_png: Path) -> None:
    groups: Dict[str, List[float]] = {}
    for r in rows:
        k = str(r[key])
        groups.setdefault(k, []).append(float(r[value]))

    labels = sorted(groups.keys())
    data = [groups[k] for k in labels]
    plt.figure(figsize=(max(8, len(labels) * 0.8), 5))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(value)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_bar_mean(rows: List[Dict[str, float]], key: str, value: str, title: str, out_png: Path) -> None:
    groups: Dict[str, List[float]] = {}
    for r in rows:
        k = str(r[key])
        groups.setdefault(k, []).append(float(r[value]))

    labels = sorted(groups.keys())
    means = [float(np.mean(groups[k])) for k in labels]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, means)
    plt.title(title)
    plt.ylabel(f"mean {value}")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyse reconstruction MedVAE (MRIxFields)")
    ap.add_argument("--env", default="local", choices=["local", "jeanzay", "dgx"])
    ap.add_argument("--data-root", default=None, help="Override data_root")
    ap.add_argument("--output-dir", default="results/medvae_reconstruction_analysis")
    ap.add_argument("--model-name", default="medvae_4_1_3d")
    ap.add_argument("--device", default=None)
    ap.add_argument("--splits", nargs="+", default=["Training_prospective", "Training_retrospective"])
    ap.add_argument("--modalities", nargs="+", default=list(DEFAULT_MODALITIES))
    ap.add_argument("--fields", nargs="+", default=list(DEFAULT_FIELDS))
    ap.add_argument("--max-per-group", type=int, default=6)
    ap.add_argument("--max-ssim-slices", type=int, default=32)
    ap.add_argument("--max-orientation-slices", type=int, default=24)
    ap.add_argument("--try-full-image", action="store_true")
    ap.add_argument("--patch-size", nargs=3, type=int, default=[112, 128, 80])
    ap.add_argument("--patch-overlap", type=float, default=0.25)
    args = ap.parse_args()

    env_cfg = load_env_config(args.env)
    data_root = Path(args.data_root) if args.data_root else Path(env_cfg["data_root"])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    from medvae import MVAE

    model = MVAE(model_name=args.model_name, modality="mri").to(device)
    model.eval()
    patched = PatchedVAE(model, patch_size=tuple(args.patch_size), overlap=args.patch_overlap).to(device)

    samples = collect_samples(
        data_root=data_root,
        splits=args.splits,
        modalities=args.modalities,
        fields=args.fields,
        max_per_group=args.max_per_group,
    )
    if not samples:
        raise RuntimeError("Aucun sample trouve pour les splits demandes.")

    print(f"Samples collectes: {len(samples)}")

    rows: List[Dict[str, float]] = []
    full_count = 0
    patch_count = 0

    for i, s in enumerate(samples, start=1):
        img = nib.load(str(s.path))
        vol = img.get_fdata(dtype=np.float32)
        vol = resample_to_1mm(vol, img.affine)
        vol = normalize_percentile(vol)

        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
        rec_t, mode = run_reconstruction(model=model, x=x, try_full_image=args.try_full_image, patched=patched)
        rec = rec_t.squeeze().detach().cpu().numpy().astype(np.float32)

        if mode == "full":
            full_count += 1
        else:
            patch_count += 1

        if vol.shape != rec.shape:
            d0, d1, d2 = min(vol.shape[0], rec.shape[0]), min(vol.shape[1], rec.shape[1]), min(vol.shape[2], rec.shape[2])
            vol = vol[:d0, :d1, :d2]
            rec = rec[:d0, :d1, :d2]
        mae = float(np.mean(np.abs(vol - rec)))
        mse = float(np.mean((vol - rec) ** 2))
        ssim = ssim_3d(vol, rec, max_slices=args.max_ssim_slices)
        ori = orientation_ssim(vol, rec, max_slices_per_axis=args.max_orientation_slices)
        axcodes = "".join(aff2axcodes(img.affine))

        artifact_scores = infer_artifact_scores(vol)

        row: Dict[str, float] = {
            "file": s.path.name,
            "split": s.split,
            "modality": s.modality,
            "field": s.field,
            "axcodes": axcodes,
            "reconstruction_mode": mode,
            "mae": mae,
            "mse": mse,
            "ssim": ssim,
            **ori,
            **artifact_scores,
        }
        rows.append(row)

        print(f"[{i:03d}/{len(samples)}] {s.path.name} | mode={mode} | MAE={mae:.4f} | SSIM={ssim:.4f}")

    classify_artifact_type(rows)

    # CSV detail
    csv_path = out_dir / "medvae_reconstruction_detailed.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Graphiques
    image_type_key = "split"
    plot_box_by_category(
        rows, key=image_type_key, value="ssim",
        title="SSIM par type d'image (split)",
        out_png=out_dir / "ssim_by_image_type.png",
    )
    plot_box_by_category(
        rows, key="artifact_type", value="ssim",
        title="SSIM par type d'artefact (inference proxy)",
        out_png=out_dir / "ssim_by_artifact_type.png",
    )
    plot_bar_mean(
        rows, key="artifact_type", value="mae",
        title="MAE moyen par type d'artefact (inference proxy)",
        out_png=out_dir / "mae_by_artifact_type.png",
    )

    # Orientation
    orient_keys = ["ssim_sagittal", "ssim_coronal", "ssim_axial"]
    orient_means = {k: float(np.mean([r[k] for r in rows])) for k in orient_keys}

    # Resume texte
    mean_ssim = float(np.mean([r["ssim"] for r in rows]))
    mean_mae = float(np.mean([r["mae"] for r in rows]))
    clean_rows = [r for r in rows if r["artifact_type"] == "clean"]
    art_rows = [r for r in rows if r["artifact_type"] != "clean"]
    clean_ssim = float(np.mean([r["ssim"] for r in clean_rows])) if clean_rows else float("nan")
    art_ssim = float(np.mean([r["ssim"] for r in art_rows])) if art_rows else float("nan")

    txt_path = out_dir / "summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Analyse reconstruction MedVAE\n")
        f.write(f"n_samples: {len(rows)}\n")
        f.write(f"mode_full: {full_count}\n")
        f.write(f"mode_patch: {patch_count}\n")
        f.write(f"mean_mae: {mean_mae:.6f}\n")
        f.write(f"mean_ssim: {mean_ssim:.6f}\n")
        f.write(f"mean_ssim_clean: {clean_ssim:.6f}\n")
        f.write(f"mean_ssim_with_artifact: {art_ssim:.6f}\n")
        f.write(
            "orientation_ssim: "
            f"sagittal={orient_means['ssim_sagittal']:.6f}, "
            f"coronal={orient_means['ssim_coronal']:.6f}, "
            f"axial={orient_means['ssim_axial']:.6f}\n"
        )

    print("\nAnalyse terminee:")
    print(f"- CSV: {csv_path}")
    print(f"- Resume: {txt_path}")
    print(f"- Graphiques: {out_dir}")


if __name__ == "__main__":
    main()
