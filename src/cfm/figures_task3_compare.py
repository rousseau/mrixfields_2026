#!/usr/bin/env python3
"""Figures qualitatives Task 3 — comparaison des trois architectures MMFM.

Les prédictions de toutes les méthodes sont sauvegardées à la résolution NATIVE
du challenge (364x436x364 @ 0.5mm) quelle que soit la résolution de travail
interne du pipeline : elles sont donc directement comparables en image. La
résolution de travail de chaque méthode est indiquée dans le titre de sa
colonne, car c'est justement la variable qui change entre les runs de
2026-08-07 (1mm) et les précédents (2mm).

Les métriques affichées proviennent des CSV produits par l'ÉVALUATEUR OFFICIEL
(src/evaluation/evaluate.py) — on ne recalcule pas de métrique maison ici, pour
éviter toute divergence avec les chiffres rapportés dans le manifest.

Usage:
    PYTHONPATH=src python src/cfm/figures_task3_compare.py \
        --pairs 0.1T_to_7T 7T_to_0.1T --subjects 0006
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env

MODALITY = "T1W"

# (label affiché, dossier de prédictions, CSV de métriques officielles)
METHODS = [
    ("Vectorisé @1mm", "outputs/mmfm/vectorized/predictions/task3/T1W",
     "results/mmfm/comparison_20260807_1mm/task3_vectorized_1mm_T1W.csv"),
    ("UNet @1mm", "outputs/mmfm/unet/predictions/task3/T1W",
     "results/mmfm/comparison_20260807_1mm/task3_unet_1mm_T1W.csv"),
    ("INR @1mm (z=129024)", "outputs/mmfm/inr/predictions/task3/T1W",
     "results/mmfm/comparison_20260807_1mm/task3_inr_129k_T1W.csv"),
]


def _load_metrics(csv_path: str) -> dict:
    p = Path(csv_path)
    if not p.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        out[r["pair"]] = (float(r["nrmse_mean"]), float(r["ssim_mean"]), float(r["lpips_mean"]))
    return out


def _norm(v: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(v, 0.5), np.percentile(v, 99.5)
    return np.clip((v - lo) / max(hi - lo, 1e-8), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["0.1T_to_7T", "7T_to_0.1T"])
    ap.add_argument("--subjects", nargs="+", default=["0006"])
    ap.add_argument("--outdir", default="results/mmfm/comparison_20260807_1mm/figures")
    args = ap.parse_args()

    data_root = Path(load_env("local")["data_root"])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = {label: _load_metrics(csv) for label, _, csv in METHODS}

    for pair in args.pairs:
        src, tgt = pair.split("_to_")
        for sid in args.subjects:
            gt_path = data_root / "Training_prospective" / MODALITY / tgt / f"P_{MODALITY}_{tgt}_{sid}.nii.gz"
            if not gt_path.exists():
                print(f"  [skip] GT absent : {gt_path}")
                continue
            gt = _norm(nib.load(str(gt_path)).get_fdata(dtype=np.float32))

            cols = [("Vérité terrain", gt, None)]
            for label, pred_dir, _ in METHODS:
                f = Path(pred_dir) / pair / f"P_{MODALITY}_{tgt}_{sid}.nii.gz"
                if not f.exists():
                    print(f"  [skip] {label} : {f} absent")
                    continue
                cols.append((label, _norm(nib.load(str(f)).get_fdata(dtype=np.float32)),
                             metrics.get(label, {}).get(pair)))

            h, w, d = gt.shape
            planes = {"Axiale": (slice(None), slice(None), d // 2),
                      "Coronale": (slice(None), w // 2, slice(None)),
                      "Sagittale": (h // 2, slice(None), slice(None))}

            fig, axes = plt.subplots(3, len(cols), figsize=(3.5 * len(cols), 10.5))
            for r, (pname, sl) in enumerate(planes.items()):
                for c, (label, vol, mt) in enumerate(cols):
                    axes[r, c].imshow(np.rot90(vol[sl]), cmap="gray", vmin=0, vmax=1)
                    axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                    if r == 0:
                        title = label
                        if mt:
                            title += f"\nnRMSE={mt[0]:.3f} SSIM={mt[1]:.3f} LPIPS={mt[2]:.3f}"
                        axes[r, c].set_title(title, fontsize=10)
                axes[r, 0].set_ylabel(pname, fontsize=11)

            fig.suptitle(f"Task 3 — {MODALITY} {src} → {tgt}, sujet {sid} "
                         f"(prédictions à la résolution native 0.5mm)", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            out = outdir / f"task3_{MODALITY}_{sid}_{pair}.png"
            fig.savefig(out, dpi=125)
            plt.close(fig)
            print(f"  → {out}", flush=True)

    print(f"\nFigures : {outdir}")


if __name__ == "__main__":
    main()
