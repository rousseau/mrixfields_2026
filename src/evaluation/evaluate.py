#!/usr/bin/env python3
"""MRIxFields 2026 — Orchestrateur d'évaluation multi-tâches

Wrapper qui appelle l'évaluateur officiel du challenge (~/Code/MRIxFields2026/Evaluation/evaluate.py)
une fois par paire source→cible, puis agrège les résultats.

Usage:
    # Task 2 (0.1T → 1.5T/3T/5T/7T) — auto-découverte
    python src/evaluation/evaluate.py --method mmfm_unet --task task2

    # Task 1 (Any → 7T) — avec dossier de prédiction spécifique
    python src/evaluation/evaluate.py --method mmfm_unet --task task1 --pred-dir results/task1/...
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import nibabel as nib
import nibabel.processing as nib_proc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --------------------------------------------------------------------------- #
#  Challenge constants
# --------------------------------------------------------------------------- #

DOMAINS = ["0.1T", "1.5T", "3T", "5T", "7T"]

# Paires par tâche (selon Submission/README.md du challenge)
TASK_PAIRS = {
    "task1": [("0.1T", "7T"), ("1.5T", "7T"), ("3T", "7T"), ("5T", "7T")],
    "task2": [("0.1T", "1.5T"), ("0.1T", "3T"), ("0.1T", "5T"), ("0.1T", "7T")],
    "task3": [(s, t) for s in DOMAINS for t in DOMAINS if s != t],  # 20 paires
}

PROSPECTIVE_SUBJECTS = ["0006", "0007", "0009"]

# --------------------------------------------------------------------------- #
#  Registry & Auto-discovery
# --------------------------------------------------------------------------- #

VAE_REG = {
    "aekl":        {"path": "outputs/vae3d/runs/vae3d_multimodal/weights/model_best.pth"},
    "vqvae":       {"path": "outputs/pythae_vqvae3d/runs/pythae_vqvae3d_multimodal/weights/model_best.pth"},
    "medvae_frozen": {"path": None},
    "medvae_ft":   {"path": "outputs/medvae/runs/medvae_finetune_all/weights/model_best.pth"},
}

CFM_REG = {
    "cfm":         {"path": "outputs/cfm3d/runs/cfm3d_T1W_medvae_0p1T_7T/weights/model_final.pth"},
    "mmfm":        {"path": "outputs/cfm3d/runs/mmfm3d_medvae_multimodal_vectorized_v1/weights/model_final.pth"},
    "mmfm_unet":   {"path": "outputs/cfm3d/runs/mmfm3d_unet_medvae_multimodal/weights/model_final.pth"},
}


def _parse_iter(name: str) -> int:
    m = re.search(r"[\W_](\d+)(k?)", name)
    if not m:
        return 0
    n, k = m.group(1), m.group(2)
    return int(n) * (1000 if k else 1)


def discover_pred_dir(method: str, modality: str = "T1W", task: str = "task3") -> Optional[Path]:
    # New structured full-resolution output: outputs/predictions/{method}/{task}/{modality}
    if method in ("mmfm_unet", "mmfm"):
        fullres_candidate = Path("outputs/predictions") / method / task / modality
        if fullres_candidate.exists() and any(fullres_candidate.rglob("*.nii*")):
            return fullres_candidate

        # Legacy low-resolution fallback (will be removed once full-res is default)
        root, pattern = Path("results/mmfm/visuals"), f"{method}_*"
        if not root.exists():
            return None
        dirs = [d for d in root.glob(pattern) if d.is_dir()]
        if not dirs:
            return None
        return sorted(dirs, key=lambda p: _parse_iter(p.name), reverse=True)[0]

    elif method == "cfm":
        root = Path("results/cfm/visuals")
        if root.exists():
            candidates = [d for d in root.rglob("*") if d.is_dir() and any(d.glob("*.nii*"))]
            if candidates:
                for c in candidates:
                    if c.name.startswith("predictions"):
                        return c
                return sorted(candidates, key=lambda p: len(list(p.glob("*.nii*"))), reverse=True)[0]
        return None

    elif method == "stargan2d":
        root = Path("outputs/stargan2d/runs")
        if root.exists():
            candidates = [d for d in root.rglob("predictions") if d.is_dir()]
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)
        return None

    return None


# --------------------------------------------------------------------------- #
#  Parsing prediction files
# --------------------------------------------------------------------------- #

def parse_mmfm_filename(name: str) -> Optional[Dict]:
    pattern = re.compile(r"^P_([A-Z0-9]+)_([\d\.]+T)_(\d{4})_([A-Z0-9]+)_([\d\.]+T)_.*\.nii.*$")
    m = pattern.match(name)
    if m:
        return {
            "modality": m.group(1), "src_field": m.group(2), "subject": m.group(3),
            "tgt_field": m.group(5),
        }
    return None


def parse_generic_filename(name: str) -> Optional[Dict]:
    pattern = re.compile(r"^P_([A-Z0-9]+)_([\d\.]+T)_(\d{4})\.nii.*$")
    m = pattern.match(name)
    if m:
        return {"modality": m.group(1), "tgt_field": m.group(2), "subject": m.group(3)}
    return None


def parse_pair_dir_name(name: str) -> Optional[Tuple[str, str]]:
    """Parse parent directory name like '0.1T_to_7T' -> (src, tgt)."""
    m = re.match(r"^([\d\.]+T)_to_([\d\.]+T)$", name)
    if m:
        return m.group(1), m.group(2)
    return None


def build_prediction_matrix(pred_dir: Path, modality: str = "T1W") -> Dict[str, Dict[str, Dict[str, Path]]]:
    """Build matrix: {src_field: {subject: {tgt_field: pred_path}}}."""
    matrix = {}
    for pred_path in pred_dir.rglob("*.nii*"):
        info = parse_mmfm_filename(pred_path.name) or parse_generic_filename(pred_path.name)
        if not info:
            continue

        # Modality filter
        if info.get("modality") != modality:
            continue

        # Source/target fields: prefer parent dir name (official structured output),
        # fallback to filename parsing.
        pair = parse_pair_dir_name(pred_path.parent.name)
        if pair:
            src, tgt = pair
        else:
            src = info.get("src_field", "0.1T")
            tgt = info.get("tgt_field")

        sid = info.get("subject")
        if sid and src and tgt:
            if src not in matrix:
                matrix[src] = {}
            if sid not in matrix[src]:
                matrix[src][sid] = {}
            matrix[src][sid][tgt] = pred_path
    return matrix


def print_matrix(matrix: Dict, task_pairs: List[Tuple[str, str]], method: str, task: str, modality: str = "T1W"):
    """Print ASCII matrix. matrix: {src_field: {subject: {tgt_field: path}}}"""
    # Collect all target fields actually needed for this task
    tgt_fields = sorted({t for _, t in task_pairs})
    src_fields = sorted({s for s, _ in task_pairs})
    print(f"\n{'=' * 70}")
    print(f"MATRICE DE COMPLÉTUDE — {method} | {task} | {modality}")
    print(f"{'=' * 70}")
    header = "Source | Sujet  | " + " | ".join(f"{t:6s}" for t in tgt_fields)
    print(header)
    print("-" * len(header))
    total = 0
    expected = len(PROSPECTIVE_SUBJECTS) * len(task_pairs)
    for src in src_fields:
        for sid in PROSPECTIVE_SUBJECTS:
            row = f"{src:6s} | {sid}  |"
            for tgt in tgt_fields:
                # Check if this specific src→tgt pair exists in task
                if (src, tgt) not in task_pairs:
                    row += "   -   |"
                elif tgt in matrix.get(src, {}).get(sid, {}):
                    row += "   ✅  |"
                    total += 1
                else:
                    row += "   ❌  |"
            print(row)
    print(f"\nTotal: {total}/{expected} prédictions trouvées")
    print(f"{'=' * 70}\n")
    return total, expected


# --------------------------------------------------------------------------- #
#  Official evaluator wrapper (per pair)
# --------------------------------------------------------------------------- #

def prepare_pair_dir(
    matrix: Dict,
    src_field: str,
    tgt_field: str,
    target_dir: Path,
    tmpbase: Path,
    modality: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Create official pred/target dirs for ONE pair, with resampling.

    Args:
        matrix: {subject: {tgt_field: pred_path}} for a fixed source field.
    """
    pred_out = tmpbase / f"{src_field}_to_{tgt_field}" / "pred"
    tgt_out = tmpbase / f"{src_field}_to_{tgt_field}" / "gt"
    pred_out.mkdir(parents=True, exist_ok=True)
    tgt_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for sid in PROSPECTIVE_SUBJECTS:
        if tgt_field not in matrix.get(sid, {}):
            continue

        pred_src = matrix[sid][tgt_field]
        official_name = f"P_{modality}_{tgt_field}_{sid}.nii.gz"
        gt_src = target_dir / tgt_field / official_name

        if not gt_src.exists():
            print(f"⚠️  GT manquant: {gt_src}")
            continue

        # Load & resample if needed
        pred_nii = nib.load(str(pred_src))
        gt_nii = nib.load(str(gt_src))
        pred_data = pred_nii.get_fdata(dtype=np.float32)
        gt_data = gt_nii.get_fdata(dtype=np.float32)

        if pred_data.shape != gt_data.shape or not np.allclose(pred_nii.affine, gt_nii.affine, atol=1e-3):
            pred_img = nib.Nifti1Image(pred_data, pred_nii.affine)
            gt_img = nib.Nifti1Image(gt_data, gt_nii.affine)
            pred_resampled = nib_proc.resample_from_to(pred_img, gt_img, order=3, mode="constant", cval=0.0)
            pred_data = pred_resampled.get_fdata(dtype=np.float32)

        nib.save(nib.Nifti1Image(pred_data, gt_nii.affine, gt_nii.header), str(pred_out / official_name))
        shutil.copy2(gt_src, tgt_out / official_name)
        count += 1

    if count == 0:
        return None, None
    return pred_out, tgt_out


def run_official_eval(
    pred_dir: Path,
    target_dir: Path,
    metrics: List[str],
    device: str,
    output_json: Path,
) -> Dict:
    official_script = Path.home() / "Code" / "MRIxFields2026" / "Evaluation" / "evaluate.py"
    if not official_script.exists():
        raise FileNotFoundError(f"Script officiel non trouvé: {official_script}")

    cmd = [
        sys.executable, str(official_script),
        "--pred_dir", str(pred_dir),
        "--target_dir", str(target_dir),
        "--metrics", *metrics,
        "--device", device,
        "--output_json", str(output_json),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"Évaluateur officiel a échoué (code {result.returncode})")

    with open(output_json) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  Task evaluation
# --------------------------------------------------------------------------- #

def evaluate_task(
    task: str,
    method: str,
    pred_dir: Path,
    target_dir: Path,
    modality: str,
    metrics: List[str],
    device: str,
    output_csv: Path,
) -> List[Dict]:
    """Evaluate all pairs for a given task."""
    pairs = TASK_PAIRS[task]
    matrix = build_prediction_matrix(pred_dir, modality=modality)
    total, expected = print_matrix(matrix, pairs, method, task, modality)

    if total < expected:
        print(f"❌ Évaluation impossible: {total}/{expected} prédictions manquantes")
        return []

    print(f"✅ Cohorte complète pour {task}: {total}/{expected} prédictions")

    all_results = []
    with tempfile.TemporaryDirectory(prefix=f"mrix_eval_{method}_{task}_") as tmpdir:
        tmpbase = Path(tmpdir)

        for src_field, tgt_field in pairs:
            print(f"\n--- Pair: {src_field} → {tgt_field} ---")
            # Pass only the sub-matrix for this specific source
            src_matrix = matrix.get(src_field, {})
            pair_pred, pair_tgt = prepare_pair_dir(src_matrix, src_field, tgt_field, target_dir, tmpbase, modality)

            if pair_pred is None:
                print(f"⚠️  Aucune prédiction pour {src_field}→{tgt_field}, SKIP")
                continue

            json_path = tmpbase / f"{src_field}_to_{tgt_field}.json"
            try:
                summary = run_official_eval(pair_pred, pair_tgt, metrics, device, json_path)
                # Expand per-subject results if available in future versions of official script,
                # otherwise use summary
                row = {
                    "method": method,
                    "task": task,
                    "pair": f"{src_field}_to_{tgt_field}",
                    "n_subjects": len(list(pair_pred.glob("*.nii*"))),
                }
                for k in metrics:
                    row[f"{k}_mean"] = summary.get(f"{k}_mean", float("nan"))
                    row[f"{k}_std"] = summary.get(f"{k}_std", float("nan"))
                all_results.append(row)

                print(f"  nRMSE: {row['nrmse_mean']:.4f} ± {row['nrmse_std']:.4f}")
                print(f"  SSIM : {row['ssim_mean']:.4f} ± {row['ssim_std']:.4f}")
                print(f"  LPIPS: {row['lpips_mean']:.4f} ± {row['lpips_std']:.4f}")

            except Exception as e:
                print(f"❌ Erreur pour {src_field}→{tgt_field}: {e}")
                continue

    # Save aggregated CSV
    if all_results:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["method", "task", "pair", "n_subjects"] + \
                     [f"{k}_mean" for k in metrics] + [f"{k}_std" for k in metrics]

        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n✅ Résultats sauvegardés: {output_csv}")

    return all_results


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

SUPPORTED_METHODS = ["stargan2d", "cfm", "mmfm", "mmfm_unet"]
SUPPORTED_TASKS = ["task1", "task2", "task3"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="MRIxFields2026 — Évaluation multi-tâches (wrapper officiel)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python src/evaluation/evaluate.py --method mmfm_unet --task task2
  python src/evaluation/evaluate.py --method mmfm_unet --task task1 --pred-dir results/task1/...
"""
    )
    parser.add_argument("--method", required=True, choices=SUPPORTED_METHODS)
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--modality", type=str, default="T1W",
                        choices=["T1W", "T2W", "T2FLAIR"])
    parser.add_argument("--vae-type", type=str, default="aekl",
                        choices=["aekl", "vqvae", "medvae_frozen", "medvae_ft"])
    parser.add_argument("--pred-dir", type=str, default=None)
    parser.add_argument("--target-dir", type=str, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--metrics", type=str, default="nrmse,ssim,lpips")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-csv", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    data_root = Path(args.data_root) if args.data_root else \
                Path(os.environ.get("MRIXFIELDS_DATA", "/home/rousseau/Data/MRIxFields_20260414"))

    pred_dir = Path(args.pred_dir) if args.pred_dir else discover_pred_dir(args.method, modality=args.modality, task=args.task)
    target_dir = Path(args.target_dir) if args.target_dir else data_root / "Training_prospective" / args.modality
    metrics = [m.strip() for m in args.metrics.split(",")]

    if pred_dir is None:
        print(f"❌ Aucune prédiction trouvée pour {args.method}")
        sys.exit(1)

    print(f"[Auto] pred_dir:  {pred_dir}")
    print(f"[Auto] target_dir: {target_dir}")
    print(f"[Auto] modality:   {args.modality}")

    # Output CSV
    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        output_csv = Path(f"results/{args.task}_{args.method}_{args.modality}.csv")

    # Run evaluation
    results = evaluate_task(
        task=args.task,
        method=args.method,
        pred_dir=pred_dir,
        target_dir=target_dir,
        modality=args.modality,
        metrics=metrics,
        device=args.device,
        output_csv=output_csv,
    )

    if results:
        # Print final summary table
        print(f"\n{'=' * 70}")
        print(f"RÉSULTATS AGRÉGÉS — {args.method} | {args.task} | {args.modality}")
        print(f"{'=' * 70}")
        for r in results:
            print(f"\n{r['pair']}:")
            for k in metrics:
                mean_v = r.get(f"{k}_mean", float("nan"))
                std_v = r.get(f"{k}_std", float("nan"))
                direction = "↓" if k in ["nrmse", "lpips"] else "↑"
                print(f"  {k:>8s}: {mean_v:.4f} ± {std_v:.4f}  {direction}")
        print(f"{'=' * 70}")
        print(f"\n✅ Évaluation terminée. CSV: {output_csv}")
        sys.exit(0)
    else:
        print("\n❌ Évaluation incomplète.")
        sys.exit(1)


if __name__ == "__main__":
    main()
