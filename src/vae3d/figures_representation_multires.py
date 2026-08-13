#!/usr/bin/env python3
"""Figures qualitatives de capacité de représentation, multi-résolution.

Compare visuellement, sur les mêmes volumes et les mêmes coupes :
    GT | MedVAE pré-entraîné | MedVAE fine-tuné L1 | INR (SIREN)
à une résolution donnée (2mm, 1mm, ...).

Complète les mesures de `src/vae3d/eval_representation_multires.py` (MedVAE) et
`src/cfm/eval_inr_multires.py` (INR). MedVAE est appliqué PAR PATCHES au-delà
de 2mm (contrainte mémoire : attention quadratique au goulot, OOM en plein
volume dès 192x224x192) ; l'INR est évalué sur la grille complète (aucune
contrainte de ce type).

Usage:
    PYTHONPATH=src python src/vae3d/figures_representation_multires.py --spacing 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import center_crop_or_pad_np, load_nifti_volume
from common.metrics import compute_nrmse, compute_ssim
from vae3d.eval_representation_multires import (
    RESOLUTION_TABLE, FINETUNED_CKPT, _build_model, patched_reconstruct,
)

MODALITY = "T1W"


def _to01(x):
    return np.clip((x + 1) / 2, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--subjects", nargs="+", default=["0006"])
    ap.add_argument("--fields", nargs="+", default=["0.1T", "3T", "7T"])
    ap.add_argument("--patch", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--with-inr", action="store_true", default=True)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spacing = args.spacing
    volume_size = RESOLUTION_TABLE[spacing]
    patch = tuple(args.patch)
    stride = tuple(max(1, int(p * (1 - args.overlap))) for p in patch)
    outdir = Path(args.outdir or
                  f"results/mmfm/comparison_20260801_final/figures_representation_{spacing:g}mm")
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = resolve_paths(load_yaml_with_include("configs/mmfm/inr.yaml"), load_env("local"))
    data_root = Path(cfg["data"]["data_root"])

    print("Chargement MedVAE (pré-entraîné + fine-tuné)...")
    models = {"MedVAE pré-entraîné": _build_model(device, False)}
    if Path(FINETUNED_CKPT).exists():
        models["MedVAE fine-tuné L1"] = _build_model(device, True)

    backbone = None
    if args.with_inr:
        from cfm.arch_inr import load_inr_backbone
        from cfm.inr_backbone import fit_new_volume, make_coord_grid
        from cfm.eval_inr_multires import _decode_chunked
        print("Chargement du backbone INR...")
        backbone = load_inr_backbone(cfg, device)
        coords = make_coord_grid(volume_size, device=device)

    h, w, d = volume_size
    planes = {"Axiale": (slice(None), slice(None), d // 2),
              "Coronale": (slice(None), w // 2, slice(None)),
              "Sagittale": (h // 2, slice(None), slice(None))}

    for sid in args.subjects:
        for field in args.fields:
            p = data_root / "Training_prospective" / MODALITY / field / f"P_{MODALITY}_{field}_{sid}.nii.gz"
            if not p.exists():
                print(f"  [skip] {p.name}")
                continue
            vol, _ = load_nifti_volume(p, target_spacing=(spacing,) * 3, volume_size=None,
                                       normalize=True, lo_pct=0.5, hi_pct=99.5)
            vol = center_crop_or_pad_np(vol, volume_size)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            gt = _to01(vol)
            fg = gt > 0.025

            recons = {}
            for name, inner in models.items():
                rec = patched_reconstruct(inner, x, patch, stride, batch_size=8)
                recons[name] = _to01(rec[0, 0].cpu().numpy())
                del rec
                torch.cuda.empty_cache()

            if backbone is not None:
                z = fit_new_volume(backbone, x, coords,
                                   num_steps=backbone.cfg.inner_steps_eval,
                                   lr=backbone.cfg.inner_lr,
                                   num_points=min(131072, int(np.prod(volume_size))))
                rec = _decode_chunked(backbone, coords, z).reshape(volume_size)
                recons["INR (SIREN)"] = _to01(rec.float().cpu().numpy())
                del rec, z
                torch.cuda.empty_cache()

            cols = ["GT"] + list(recons.keys())
            fig, axes = plt.subplots(3, len(cols), figsize=(3.4 * len(cols), 10.5))
            for r, (pname, sl) in enumerate(planes.items()):
                axes[r, 0].imshow(np.rot90(gt[sl]), cmap="gray", vmin=0, vmax=1)
                axes[r, 0].set_ylabel(pname, fontsize=11)
                if r == 0:
                    axes[r, 0].set_title("GT", fontsize=12)
                for c, name in enumerate(recons, start=1):
                    axes[r, c].imshow(np.rot90(recons[name][sl]), cmap="gray", vmin=0, vmax=1)
                    if r == 0:
                        nr = compute_nrmse(recons[name], gt, mask=fg)
                        ss = compute_ssim(recons[name], gt)
                        axes[r, c].set_title(f"{name}\nnRMSE={nr:.3f} SSIM={ss:.3f}", fontsize=10)
                for ax in axes[r]:
                    ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(
                f"Capacité de représentation @ {spacing:g}mm ({'x'.join(map(str, volume_size))}) — "
                f"sujet {sid}, {field}, {MODALITY} — auto-reconstruction",
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            out = outdir / f"repr_{spacing:g}mm_{MODALITY}_{sid}_{field}.png"
            fig.savefig(out, dpi=125)
            plt.close(fig)
            print(f"  → {out}", flush=True)

    print(f"\nFigures : {outdir}")


if __name__ == "__main__":
    main()
