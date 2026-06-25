#!/usr/bin/env python3
"""
⚠️  DEPRECATED — La fonctionnalité de ce script est entièrement couverte par
    `src/evaluation/evaluate.py` qui fait appel à l'évaluateur OFFICIEL du challenge.

    Ce fichier est conservé pour compatibilité historique.
    Utilisez `evaluate.py` pour toute nouvelle évaluation.

    python src/evaluation/evaluate.py --method mmfm --task task1

    Ou avec un dossier de prédiction spécifique :
    python src/evaluation/evaluate.py --method mmfm_unet --task task2 \
        --pred-dir results/mmfm/visuals/mmfm_unet_5k
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import nibabel as nib
import nibabel.processing as nib_proc
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import compute_lpips, compute_nrmse, compute_ssim


# ── helpers ──────────────────────────────────────────────────────────────

def extract_subject_id(name: str) -> str:
    """Extract trailing 4-digit subject ID."""
    m = re.search(r"(\d{4})", name)
    return m.group(1) if m else name


def load_volume(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load volume and return data + header."""
    nii = nib.load(str(path))
    vol = nii.get_fdata(dtype=np.float32)
    return vol, nii


def preprocess_gt(gt_vol: np.ndarray, gt_nii: nib.Nifti1Image,
                  pred_vol: np.ndarray, pred_nii: nib.Nifti1Image) -> np.ndarray:
    """Resample GT to prediction space if necessary."""
    if gt_vol.shape == pred_vol.shape and np.allclose(gt_nii.affine, pred_nii.affine, atol=1e-3):
        return gt_vol
    # Resample GT to prediction space
    gt_img = nib.Nifti1Image(gt_vol, gt_nii.affine)
    pred_img = nib.Nifti1Image(pred_vol, pred_nii.affine)
    resampled = nib_proc.resample_from_to(
        gt_img, pred_img, order=3, mode="constant", cval=0.0
    )
    return resampled.get_fdata(dtype=np.float32)


def compute_metrics(pred: np.ndarray, target: np.ndarray,
                    device: torch.device) -> dict:
    """Compute all quantitative metrics."""
    # Clip to valid range and normalize to [0, 1]
    pred = np.clip(pred, pred.min(), pred.max())
    target = np.clip(target, target.min(), target.max())

    p_min, p_max = pred.min(), pred.max()
    t_min, t_max = target.min(), target.max()

    pred_n = (pred - p_min) / max(p_max - p_min, 1e-8)
    target_n = (target - t_min) / max(t_max - t_min, 1e-8)

    mae = float(np.mean(np.abs(pred_n - target_n)))
    mse = float(np.mean((pred_n - target_n) ** 2))
    ssim = compute_ssim(pred_n, target_n)
    nrmse = compute_nrmse(pred_n, target_n)
    lpips = compute_lpips(pred_n, target_n, device=device)

    return {"mae": mae, "mse": mse, "ssim": ssim,
            "nrmse": nrmse, "lpips": lpips}


# ── matching ───────────────────────────────────────────────────────────────

def match_cfm(pred_dir: Path, gt_dir: Path, src_field: str, tgt_field: str):
    """Matching for CFM (unidirectional, prediction name contains pred_)."""
    pairs = []
    for pred_path in sorted(pred_dir.glob("*.nii*")):
        sid = extract_subject_id(pred_path.name)
        gt_path = gt_dir / f"P_T1W_{tgt_field}_{sid}.nii.gz"
        if not gt_path.exists():
            continue
        pairs.append((pred_path, gt_path, sid))
    return pairs


def match_mmfm(pred_dir: Path, gt_dir: Path, modality: str):
    """Matching for MMFM (multi-target, name pattern P_{mod}_{src}_{sid}_{mod}_{tgt}_mmfm...)."""
    pairs = []
    pattern = re.compile(
        rf"P_{modality}_([\d\.]+T)_(\d{{4}})_{modality}_([\d\.]+T)_.*\.nii\.gz"
    )
    for pred_path in sorted(pred_dir.glob("*.nii.gz")):
        m = pattern.match(pred_path.name)
        if not m:
            continue
        src_field, sid, tgt_field = m.group(1), m.group(2), m.group(3)
        gt_path = gt_dir / tgt_field / f"P_{modality}_{tgt_field}_{sid}.nii.gz"
        if not gt_path.exists():
            continue
        pairs.append((pred_path, gt_path, sid, src_field, tgt_field))
    return pairs


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Évaluation quantitative CFM / MMFM"
    )
    parser.add_argument("--pred-dir", required=True, type=Path)
    parser.add_argument("--gt-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--method", required=True)
    parser.add_argument("--modality", default="T1W")
    parser.add_argument("--src-field", default="0.1T")
    parser.add_argument("--tgt-field", default="7T")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Determine matching strategy based on directory content
    sample_file = next((f for f in args.pred_dir.glob("*.nii*")), None)
    if sample_file is None:
        print("❌ Aucune prédiction trouvée.")
        return

    if "mmfm" in sample_file.name.lower() or "_mmfm" in sample_file.name:
        pairs = match_mmfm(args.pred_dir, args.gt_dir, args.modality)
    elif "pred" in sample_file.name:
        pairs = match_cfm(args.pred_dir, args.gt_dir, args.src_field, args.tgt_field)
    else:
        # Generic matching
        pairs = []
        for pred_path in sorted(args.pred_dir.glob("*.nii*")):
            sid = extract_subject_id(pred_path.name)
            gt_path = args.gt_dir / f"P_{args.modality}_{args.tgt_field}_{sid}.nii.gz"
            if not gt_path.exists():
                continue
            pairs.append((pred_path, gt_path, sid))

    if not pairs:
        print("❌ Aucune paire trouvée.")
        return

    print(f"▶ {len(pairs)} paires trouvées — méthode: {args.method}")

    # Collect results
    fieldnames = [
        "method", "modality", "subject", "src_field", "tgt_field",
        "mae", "mse", "ssim", "nrmse", "lpips"
    ]

    results = []
    for items in pairs:
        if len(items) == 3:
            pred_path, gt_path, sid = items
            src_field, tgt_field = args.src_field, args.tgt_field
        else:
            pred_path, gt_path, sid, src_field, tgt_field = items

        print(f"  [{sid}] {pred_path.name} → {gt_path.name}")

        pred_vol, pred_nii = load_volume(pred_path)
        gt_vol, gt_nii = load_volume(gt_path)
        gt_proc = preprocess_gt(gt_vol, gt_nii, pred_vol, pred_nii)

        metrics = compute_metrics(pred_vol, gt_proc, device)
        metrics.update({
            "method": args.method,
            "modality": args.modality,
            "subject": sid,
            "src_field": src_field,
            "tgt_field": tgt_field,
        })
        results.append(metrics)
        print(f"      SSIM={metrics['ssim']:.4f}  nRMSE={metrics['nrmse']:.4f}  "
              f"MAE={metrics['mae']:.4f}  LPIPS={metrics['lpips']:.4f}")

    # Write / append CSV
    exist = args.output_csv.exists()
    with open(args.output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exist:
            writer.writeheader()
        writer.writerows(results)

    # Summary
    print(f"\n{'='*60}")
    print(f"Méthode : {args.method}")
    for k in ["mae", "mse", "ssim", "nrmse", "lpips"]:
        vals = [r[k] for r in results]
        print(f"  {k:>8s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print(f"  CSV    : {args.output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
