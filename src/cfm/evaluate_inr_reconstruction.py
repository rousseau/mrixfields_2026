#!/usr/bin/env python3
"""Évalue la capacité de représentation du backbone INR — auto-reconstruction
(fit Algorithme 2 + decode, PAS de traduction de champ), comparée à MedVAE
(simple encode+decode), pour diagnostiquer si le flou observé sur les
prédictions MMFM-INR (Task 3) vient de la représentation (backbone
sous-dimensionné/sous-entraîné) ou du flow matching lui-même.

Teste deux populations séparément :
  - "prospective" : sujets tenus à l'écart de l'entraînement du backbone
    (Training_prospective, les mêmes sujets utilisés pour l'éval Task 3) —
    mesure la capacité de GÉNÉRALISATION, directement pertinente puisque
    c'est exactement ce que le flow voit en entrée à l'inférence.
  - "retro_train" : échantillon de volumes vus PENDANT l'entraînement du
    backbone — mesure la capacité brute de représentation (mémorisation),
    borne supérieure de ce que le backbone peut faire dans l'absolu.

Un net écart entre les deux indique un problème de généralisation
(sur-apprentissage/backbone trop dépendant du fitting) ; une mauvaise
fidélité sur LES DEUX indique une limite de capacité/architecture (SIREN
trop petit, pas assez de pas de fitting, etc.) plutôt qu'un problème
d'entraînement.

Usage:
    PYTHONPATH=src python src/cfm/evaluate_inr_reconstruction.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import SPLIT_MAP, center_crop_or_pad_np, load_nifti_volume
from common.metrics import compute_nrmse, compute_ssim
from cfm.arch_inr import load_inr_backbone
from cfm.inr_backbone import fit_new_volume, make_coord_grid
from models.vae_loader import load_vae

VOLUME_SIZE = (96, 112, 96)
TARGET_SPACING = (2.0, 2.0, 2.0)
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITY = "T1W"
PROSPECTIVE_SUBJECTS = ["0006", "0007", "0009"]
N_RETRO_PER_FIELD = 3
OUT_CSV = Path("results/mmfm/comparison_20260801_final/inr_reconstruction_capacity.csv")


def _load_volume(path: Path) -> np.ndarray:
    vol, _ = load_nifti_volume(
        path, target_spacing=TARGET_SPACING, volume_size=None,
        normalize=True, lo_pct=0.5, hi_pct=99.5,
    )
    return center_crop_or_pad_np(vol, VOLUME_SIZE)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inr_cfg = load_yaml_with_include("configs/mmfm/inr.yaml")
    inr_cfg = resolve_paths(inr_cfg, load_env("local"))
    print("Chargement du backbone INR (production)...")
    backbone = load_inr_backbone(inr_cfg, device)
    coords_full = make_coord_grid(VOLUME_SIZE, device=device)

    vec_cfg = load_yaml_with_include("configs/mmfm/vectorized.yaml")
    vec_cfg = resolve_paths(vec_cfg, load_env("local"))
    print("Chargement de MedVAE (référence)...")
    vae = load_vae(vec_cfg, device)
    vae.eval()

    data_root = Path(inr_cfg["data"]["data_root"])

    samples = []
    for sid in PROSPECTIVE_SUBJECTS:
        for field in FIELDS:
            p = data_root / "Training_prospective" / MODALITY / field / f"P_{MODALITY}_{field}_{sid}.nii.gz"
            if p.exists():
                samples.append(("prospective", field, sid, p))

    for field in FIELDS:
        src_dir = data_root / SPLIT_MAP["retro_train"] / MODALITY / field
        files = sorted(src_dir.glob("*.nii.gz"))[:N_RETRO_PER_FIELD]
        for f in files:
            sid = f.stem.split("_")[-1]
            samples.append(("retro_train", field, sid, f))

    print(f"{len(samples)} volumes à évaluer (auto-reconstruction, pas de traduction de champ)\n")

    results = []
    t_start = time.time()
    for split, field, sid, path in samples:
        vol = _load_volume(path)
        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

        t0 = time.time()
        z = fit_new_volume(
            backbone, x, coords_full,
            num_steps=backbone.cfg.inner_steps_eval, lr=backbone.cfg.inner_lr,
        )
        with torch.no_grad():
            inr_recon = backbone.decode(coords_full.unsqueeze(0), z).reshape(VOLUME_SIZE)
        inr_time = time.time() - t0

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            z_vae = vae.encode(x)
            vae_recon = vae.decode(z_vae)

        gt_np = ((x[0, 0].clamp(-1, 1) + 1) / 2).cpu().numpy()
        inr_np = ((inr_recon.clamp(-1, 1) + 1) / 2).cpu().numpy()
        vae_np = ((vae_recon[0, 0].clamp(-1, 1) + 1) / 2).float().cpu().numpy()
        fg_mask = gt_np > 0.05

        row = {
            "split": split, "field": field, "subject": sid,
            "inr_nrmse": compute_nrmse(inr_np, gt_np, mask=fg_mask),
            "inr_ssim": compute_ssim(inr_np, gt_np),
            "vae_nrmse": compute_nrmse(vae_np, gt_np, mask=fg_mask),
            "vae_ssim": compute_ssim(vae_np, gt_np),
            "inr_fit_time_s": round(inr_time, 2),
        }
        results.append(row)
        print(
            f"[{split:12s}] {field:5s} {sid} | INR nRMSE={row['inr_nrmse']:.4f} SSIM={row['inr_ssim']:.4f} "
            f"| MedVAE nRMSE={row['vae_nrmse']:.4f} SSIM={row['vae_ssim']:.4f}  ({inr_time:.1f}s)",
            flush=True,
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'=' * 70}\nRÉSUMÉ (temps total {(time.time() - t_start) / 60:.1f} min)\n{'=' * 70}")
    for split in ["prospective", "retro_train"]:
        sub = [r for r in results if r["split"] == split]
        if not sub:
            continue
        print(f"\n--- {split} (n={len(sub)}) ---")
        print(f"  INR    : nRMSE={np.mean([r['inr_nrmse'] for r in sub]):.4f}  "
              f"SSIM={np.mean([r['inr_ssim'] for r in sub]):.4f}")
        print(f"  MedVAE : nRMSE={np.mean([r['vae_nrmse'] for r in sub]):.4f}  "
              f"SSIM={np.mean([r['vae_ssim'] for r in sub]):.4f}")
        for field in FIELDS:
            fsub = [r for r in sub if r["field"] == field]
            if not fsub:
                continue
            print(f"    {field:5s} : INR SSIM={np.mean([r['inr_ssim'] for r in fsub]):.4f}  "
                  f"MedVAE SSIM={np.mean([r['vae_ssim'] for r in fsub]):.4f}")

    print(f"\nCSV : {OUT_CSV}")


if __name__ == "__main__":
    main()
