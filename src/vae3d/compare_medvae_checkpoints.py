#!/usr/bin/env python3
"""Compare la fidélité de reconstruction de MedVAE PRÉ-ENTRAÎNÉ (poids Stanford
bruts) vs notre checkpoint fine-tuné L1, à plusieurs résolutions.

Question posée : « les poids MedVAE disponibles peuvent-ils fonctionner à la
résolution des données du challenge ? » Le papier (Varma et al.,
arXiv:2502.14753) rapporte MS-SSIM 0.983 sur IRM cérébrale pour ce modèle ;
nos reconstructions sont à ~0.88-0.95 avec du flou visible. Deux hypothèses à
départager, et ce script les départage sans aucun ré-entraînement :

  (H1) notre fine-tuning L1 pur a DÉGRADÉ le décodeur pré-entraîné (qui, lui,
       a été entraîné avec LPIPS + PatchGAN) ;
  (H2) le modèle est intrinsèquement limité à cette résolution / sur ces données.

Si le pré-entraîné brut reconstruit nettement mieux que notre fine-tuné, c'est
H1 — et le fine-tuning doit être refait avec la vraie recette (voir
configs/medvae_finetune_lpips.yaml), voire être inutile.

On évalue à plusieurs `target_spacing` car le VAE est entièrement convolutif :
la grille native du challenge est 364x436x364 @ 0.5mm, notre pipeline MMFM
travaille à 2mm (96x112x96), et le fine-tuning avait été fait à 1mm.

Usage:
    PYTHONPATH=src python src/vae3d/compare_medvae_checkpoints.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from common.io import center_crop_or_pad_np, load_nifti_volume
from common.metrics import compute_nrmse, compute_ssim

MODALITY = "T1W"
SUBJECTS = ["0006", "0007", "0009"]
FIELDS = ["0.1T", "3T", "7T"]

# (spacing, volume_size) — volume_size choisi pour couvrir ~192mm de FOV et
# rester divisible par 4 (contrainte de downsampling de MedVAE 4x).
RESOLUTIONS = [
    ((2.0, 2.0, 2.0), (96, 112, 96)),    # résolution actuelle du pipeline MMFM
    ((1.0, 1.0, 1.0), (192, 224, 192)),  # résolution du fine-tuning existant
]

FINETUNED_CKPT = "outputs/medvae/runs/medvae_finetune_all/weights/model_best.pth"
OUT_CSV = Path("results/benchmark_vae/metrics/medvae_pretrained_vs_finetuned.csv")


def _load_volume(path: Path, spacing, volume_size) -> np.ndarray:
    vol, _ = load_nifti_volume(
        path, target_spacing=spacing, volume_size=None,
        normalize=True, lo_pct=0.5, hi_pct=99.5,
    )
    return center_crop_or_pad_np(vol, volume_size)


def _build_model(device, finetuned: bool):
    from medvae import MVAE

    model = MVAE(model_name="medvae_4_1_3d", modality="mri").to(device)
    if finetuned:
        ckpt = torch.load(FINETUNED_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _reconstruct(model, x):
    inner = getattr(model, "model", model)
    post = inner.encode(x)
    z = post.mode() if hasattr(post, "mode") else post.sample()
    return inner.decode(z)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(load_env("local")["data_root"])

    variants = {}
    print("Chargement MedVAE pré-entraîné (poids Stanford bruts)...")
    variants["pretrained"] = _build_model(device, finetuned=False)
    if Path(FINETUNED_CKPT).exists():
        print(f"Chargement MedVAE fine-tuné L1 ({FINETUNED_CKPT})...")
        variants["finetuned_l1"] = _build_model(device, finetuned=True)
    else:
        print(f"[WARN] {FINETUNED_CKPT} introuvable — comparaison sur le pré-entraîné seul")

    rows = []
    for spacing, volume_size in RESOLUTIONS:
        tag = f"{spacing[0]:g}mm_{'x'.join(map(str, volume_size))}"
        print(f"\n{'=' * 70}\nRésolution {tag}\n{'=' * 70}")
        for sid in SUBJECTS:
            for field in FIELDS:
                p = data_root / "Training_prospective" / MODALITY / field / f"P_{MODALITY}_{field}_{sid}.nii.gz"
                if not p.exists():
                    continue
                vol = _load_volume(p, spacing, volume_size)
                x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
                gt = np.clip((vol + 1) / 2, 0, 1)
                fg = gt > 0.025

                line = f"  {sid} {field:5s} |"
                for name, model in variants.items():
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                        rec = _reconstruct(model, x)
                    rec_np = np.clip((rec[0, 0].float().cpu().numpy() + 1) / 2, 0, 1)
                    nrmse = compute_nrmse(rec_np, gt, mask=fg)
                    ssim = compute_ssim(rec_np, gt)
                    rows.append({
                        "resolution": tag, "spacing_mm": spacing[0], "variant": name,
                        "subject": sid, "field": field,
                        "nrmse": round(float(nrmse), 5), "ssim": round(float(ssim), 5),
                    })
                    line += f"  {name}: nRMSE={nrmse:.4f} SSIM={ssim:.4f} |"
                print(line, flush=True)

                # Écriture incrémentale : une interruption ne doit pas faire
                # perdre les résolutions déjà mesurées (la 1mm est longue).
                OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
                with open(OUT_CSV, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'=' * 70}\nRÉSUMÉ (moyennes)\n{'=' * 70}")
    for tag in sorted({r["resolution"] for r in rows}):
        print(f"\n--- {tag} ---")
        for name in variants:
            sub = [r for r in rows if r["resolution"] == tag and r["variant"] == name]
            print(f"  {name:14s} : nRMSE={np.mean([r['nrmse'] for r in sub]):.4f}  "
                  f"SSIM={np.mean([r['ssim'] for r in sub]):.4f}")
        for field in FIELDS:
            parts = []
            for name in variants:
                sub = [r for r in rows if r["resolution"] == tag and r["variant"] == name and r["field"] == field]
                if sub:
                    parts.append(f"{name} SSIM={np.mean([r['ssim'] for r in sub]):.4f}")
            print(f"    {field:5s} : " + "  |  ".join(parts))

    print(f"\nCSV : {OUT_CSV}")


if __name__ == "__main__":
    main()
