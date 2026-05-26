#!/usr/bin/env python3
"""Script d'évaluation UNIFIÉ — MRIxFields 2026

Évalue toutes les méthodes avec le même protocole, garantissant la cohérence
des comparaisons entre StarGAN (Étape 1), VAE (Étape 2), et CFM (Étape 3).

Refactored to use common infrastructure.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import (
    DOMAINS,
    MODALITIES,
    extract_subject_id,
    load_nifti_volume,
)
from common.metrics import (
    compute_dice,
    compute_lpips,
    compute_nrmse,
    compute_ssim,
    compute_volume_consistency,
    DGM_LABELS,
)


# ---------------------------------------------------------------------- #
#  Helpers matching
# ---------------------------------------------------------------------- #


def match_by_subject_prefix(
    pred_dir: Path, target_dir: Path
) -> list:
    """Match prediction files with ground truth by subject ID."""
    target_lookup = {}
    for f in target_dir.rglob("*.nii.gz"):
        sid = extract_subject_id(f.name)
        target_lookup[sid] = f

    pairs = []
    for pred_path in sorted(pred_dir.rglob("*.nii.gz")):
        sid = extract_subject_id(pred_path.name)
        if sid in target_lookup:
            pairs.append((pred_path, target_lookup[sid]))

    return pairs


# ---------------------------------------------------------------------- #
#  Pipeline d'évaluation par méthode
# ---------------------------------------------------------------------- #


def evaluate_stargan2d(
    checkpoint: Path,
    subjects: str,
    data_root: Path,
    modalities: List[str] = None,
    device: str = "cuda",
) -> Dict:
    """Évaluer StarGAN v2 2D (Étape 1)."""
    raise NotImplementedError(
        "evaluate_stargan2d() à implémenter.\n"
        "Utiliser le code officiel du challenge:\n"
        "  python ~/Code/MRIxFields2026/Baseline/mrixfields/inference.py"
    )


def evaluate_vae_cfm(
    vae_checkpoint: Path,
    cfm_checkpoint: Path,
    subjects: str,
    data_root: Path,
    vae_type: str,
    modalities: List[str] = None,
    device: str = "cuda",
) -> Dict:
    """Évaluer VAE + CFM (Étapes 2+3)."""
    raise NotImplementedError(
        "evaluate_vae_cfm() à implémenter.\n"
        "Voir src/cfm/train_cfm_3d.py pour l'inférence CFM."
    )


def evaluate_pair(
    pred_path: Path,
    target_path: Path,
    metrics: List[str],
    pred_seg_path: Optional[Path] = None,
    target_seg_path: Optional[Path] = None,
    device: str = "cuda",
) -> Dict:
    """Compute all metrics for a single prediction-target pair."""
    pred, _ = load_nifti_volume(pred_path, normalize=False)
    target, _ = load_nifti_volume(target_path, normalize=False)

    # Normalize to [0, 1] for metrics
    pred_n = (pred - pred.min()) / max(pred.ptp(), 1e-8)
    target_n = (target - target.min()) / max(target.ptp(), 1e-8)

    results = {}

    if "nrmse" in metrics:
        results["nrmse"] = compute_nrmse(pred_n, target_n)
    if "ssim" in metrics:
        results["ssim"] = compute_ssim(pred_n, target_n)
    if "lpips" in metrics:
        results["lpips"] = compute_lpips(pred_n, target_n, device=device)

    if "dice" in metrics or "volume" in metrics:
        if pred_seg_path is None or target_seg_path is None:
            raise ValueError(
                "Dice/Volume metrics require --pred_seg_dir and --target_seg_dir."
            )

        seg_pred = nib.load(str(pred_seg_path)).get_fdata().astype(np.int32)
        seg_target = nib.load(str(target_seg_path)).get_fdata().astype(np.int32)

        if "dice" in metrics:
            scores = compute_dice(seg_pred, seg_target)
            results["dice"] = float(np.mean(list(scores.values())))

        if "volume" in metrics:
            voxel_size = tuple(nib.load(str(target_seg_path)).header.get_zooms()[:3])
            voxel_vol = float(np.prod(voxel_size))
            scores = compute_volume_consistency(seg_pred, seg_target, voxel_vol)
            results["volume"] = float(np.mean(list(scores.values())))

    return results


# ---------------------------------------------------------------------- #
#  Pipeline complet
# ---------------------------------------------------------------------- #


def run_evaluation(
    method: str,
    checkpoint: Optional[Path] = None,
    vae_checkpoint: Optional[Path] = None,
    cfm_checkpoint: Optional[Path] = None,
    subjects: str = "prospective_5fields",
    data_root: Optional[Path] = None,
    pred_dir: Optional[Path] = None,
    target_dir: Optional[Path] = None,
    pred_seg_dir: Optional[Path] = None,
    target_seg_dir: Optional[Path] = None,
    modalities: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    device: str = "cuda",
    output_dir: Optional[Path] = None,
    output_csv: Optional[Path] = None,
) -> Dict:
    if metrics is None:
        metrics = ["nrmse", "ssim", "lpips"]

    if data_root is None:
        data_root = Path(
            os.environ.get("MRIXFIELDS_DATA", "/home/rousseau/Data/MRIxFields_20260414")
        )

    if modalities is None:
        modalities = ["T1W"]

    all_results = []

    for modality in modalities:
        if pred_dir is None or target_dir is None:
            if subjects == "prospective_5fields":
                subject_ids = ["0006", "0007", "0009"]
            else:
                subject_ids = subjects.split(",")

            if method == "stargan2d":
                src_field = "0.1T"
                fields = list(DOMAINS)
            elif method in ["aekl_cfm3d", "vqvae_cfm3d", "medvae_frozen_cfm", "medvae_ft_cfm"]:
                src_field = "0.1T"
                fields = ["7T"]
            else:
                raise ValueError(f"Méthode inconnue: {method}")

            for tgt_field in fields:
                if src_field == tgt_field:
                    continue

                for sid in subject_ids:
                    pred_path = pred_dir / f"{sid}.nii.gz" if pred_dir else None
                    target_path = (
                        data_root
                        / "Training_prospective"
                        / modality
                        / tgt_field
                        / f"P_{modality}_{tgt_field}_{sid}.nii.gz"
                        if target_dir is None
                        else None
                    )

                    if pred_path is None or not pred_path.exists():
                        print(f"⚠️  Prediction non trouvée: {pred_path}")
                        continue

                    if target_path is None or not target_path.exists():
                        print(f"⚠️  Ground truth non trouvé: {target_path}")
                        continue

                    pred_seg_path = target_seg_path = None
                    if "dice" in metrics or "volume" in metrics:
                        if pred_seg_dir is None or target_seg_dir is None:
                            print(f"⚠️  Segmentation requise pour Dice/Volume")
                            continue
                        pred_seg_path = pred_seg_dir / f"{sid}_seg.nii.gz"
                        target_seg_path = target_seg_dir / f"P_{modality}_{tgt_field}_{sid}_seg.nii.gz"

                    result = evaluate_pair(
                        pred_path,
                        target_path,
                        metrics,
                        pred_seg_path=pred_seg_path,
                        target_seg_path=target_seg_path,
                        device=device,
                    )
                    result["subject"] = sid
                    result["modality"] = modality
                    result["src_field"] = src_field
                    result["tgt_field"] = tgt_field
                    result["method"] = method
                    all_results.append(result)

        else:
            pairs = match_by_subject_prefix(pred_dir, target_dir)
            if not pairs:
                print("Aucune paire trouvée entre pred_dir et target_dir")
                return {}

            for pred_path, target_path in pairs:
                sid = extract_subject_id(pred_path.name)

                pred_seg_path = target_seg_path = None
                if pred_seg_dir is not None and target_seg_dir is not None:
                    pred_seg_path = pred_seg_dir / f"{sid}_seg.nii.gz"
                    target_seg_path = target_seg_dir / f"{sid}_seg.nii.gz"

                result = evaluate_pair(
                    pred_path,
                    target_path,
                    metrics,
                    pred_seg_path=pred_seg_path,
                    target_seg_path=target_seg_path,
                    device=device,
                )
                result["subject"] = sid
                result["modality"] = modality
                result["method"] = method
                all_results.append(result)

    if not all_results:
        print("Aucun résultat calculé.")
        return {}

    summary = {}
    metric_keys = [
        k for k in all_results[0]
        if k not in ["subject", "modality", "method", "src_field", "tgt_field"]
    ]

    for k in metric_keys:
        vals = [r[k] for r in all_results]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals))
        summary[f"{k}_min"] = float(np.min(vals))
        summary[f"{k}_max"] = float(np.max(vals))

    print(f"\n{'=' * 60}")
    print(f"Résultats — {method} ({len(all_results)} sujets)")
    print(f"{'=' * 60}")

    for k in metric_keys:
        direction = "↓ (lower is better)" if k in ["nrmse", "lpips"] else "↑ (higher is better)"
        print(f"  {k:>10s}: {summary[f'{k}_mean']:.4f} ± {summary[f'{k}_std']:.4f}  {direction}")
    print(f"{'=' * 60}")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        existing_rows = []
        if output_csv.exists():
            with open(output_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

        fieldnames = [
            "method", "modality", "subject", "src_field", "tgt_field",
        ] + metric_keys
        for r in all_results:
            row = {k: r.get(k, "") for k in fieldnames}
            existing_rows.append(row)

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)

        print(f"\nRésultats sauvegardés: {output_csv}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"{method}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Résumé JSON sauvegardé: {summary_path}")

    return summary


# ---------------------------------------------------------------------- #
#  CLI
# ---------------------------------------------------------------------- #

SUPPORTED_METHODS = [
    "stargan2d",
    "aekl_cfm3d",
    "vqvae_cfm3d",
    "medvae_frozen_cfm",
    "medvae_ft_cfm",
]

ALL_METRICS = ["nrmse", "ssim", "lpips", "dice", "volume"]


def parse_args():
    parser = argparse.ArgumentParser(description="MRIxFields2026 Evaluation (unifié)")
    parser.add_argument("--method", required=True, choices=SUPPORTED_METHODS)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--vae-checkpoint", type=str, default=None)
    parser.add_argument("--cfm-checkpoint", type=str, default=None)
    parser.add_argument("--subjects", type=str, default="prospective_5fields")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--target-dir", type=str, default=None)
    parser.add_argument("--pred-seg-dir", type=str, default=None)
    parser.add_argument("--target-seg-dir", type=str, default=None)
    parser.add_argument("--modalities", type=str, default="T1W")
    parser.add_argument("--metrics", type=str, default="nrmse,ssim,lpips")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default="results/evaluation_table.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    vae_checkpoint = Path(args.vae_checkpoint) if args.vae_checkpoint else None
    cfm_checkpoint = Path(args.cfm_checkpoint) if args.cfm_checkpoint else None
    data_root = Path(args.data_root) if args.data_root else None
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    target_dir = Path(args.target_dir) if args.target_dir else None
    pred_seg_dir = Path(args.pred_seg_dir) if args.pred_seg_dir else None
    target_seg_dir = Path(args.target_seg_dir) if args.target_seg_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    output_csv = Path(args.output_csv)
    modalities = [m.strip() for m in args.modalities.split(",")]
    metrics = [m.strip() for m in args.metrics.split(",")]

    if args.method == "stargan2d" and checkpoint is None:
        print("ERREUR: --checkpoint requis pour stargan2d")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"MRIxFields2026 Evaluation")
    print(f"{'=' * 60}")
    print(f"Méthode       : {args.method}")
    print(f"Modalités     : {modalities}")
    print(f"Métriques     : {metrics}")
    print(f"Sujets        : {args.subjects}")
    print(f"Device        : {args.device}")
    print()

    summary = run_evaluation(
        method=args.method,
        checkpoint=checkpoint,
        vae_checkpoint=vae_checkpoint,
        cfm_checkpoint=cfm_checkpoint,
        subjects=args.subjects,
        data_root=data_root,
        pred_dir=pred_dir,
        target_dir=target_dir,
        pred_seg_dir=pred_seg_dir,
        target_seg_dir=target_seg_dir,
        modalities=modalities,
        metrics=metrics,
        device=args.device,
        output_dir=output_dir,
        output_csv=output_csv,
    )

    if summary:
        print(f"\n✅ Évaluation terminée.")
        sys.exit(0)
    else:
        print(f"\n❌ Évaluation incomplète.")
        sys.exit(1)


if __name__ == "__main__":
    main()
