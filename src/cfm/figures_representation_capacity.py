#!/usr/bin/env python3
"""Figures qualitatives : capacité de représentation INR (SIREN) vs MedVAE —
auto-reconstruction (fit Algorithme 2 + decode / encode + decode), PAS de
traduction de champ. Complète `evaluate_inr_reconstruction.py` (chiffré
uniquement) par une inspection visuelle sur un échantillon représentatif.

Usage:
    PYTHONPATH=src python src/cfm/figures_representation_capacity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import center_crop_or_pad_np, load_nifti_volume
from common.metrics import compute_nrmse, compute_ssim
from cfm.arch_inr import load_inr_backbone
from cfm.inr_backbone import fit_new_volume, make_coord_grid
from models.vae_loader import load_vae

VOLUME_SIZE = (96, 112, 96)
TARGET_SPACING = (2.0, 2.0, 2.0)
MODALITY = "T1W"
SUBJECTS = ["0006", "0007", "0009"]
FIELDS = ["0.1T", "3T", "7T"]
OUT_DIR = Path("results/mmfm/comparison_20260801_final/figures_representation_capacity")


def _load_volume(path: Path) -> np.ndarray:
    vol, _ = load_nifti_volume(
        path, target_spacing=TARGET_SPACING, volume_size=None,
        normalize=True, lo_pct=0.5, hi_pct=99.5,
    )
    return center_crop_or_pad_np(vol, VOLUME_SIZE)


def _to01(x: np.ndarray) -> np.ndarray:
    return np.clip((x + 1) / 2, 0, 1)


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    h, w, d = VOLUME_SIZE
    slices = {"Axiale": (slice(None), slice(None), d // 2),
              "Coronale": (slice(None), w // 2, slice(None)),
              "Sagittale": (h // 2, slice(None), slice(None))}

    for sid in SUBJECTS:
        for field in FIELDS:
            path = data_root / "Training_prospective" / MODALITY / field / f"P_{MODALITY}_{field}_{sid}.nii.gz"
            if not path.exists():
                print(f"  [skip] {path} introuvable")
                continue

            vol = _load_volume(path)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

            z = fit_new_volume(backbone, x, coords_full, num_steps=backbone.cfg.inner_steps_eval, lr=backbone.cfg.inner_lr)
            with torch.no_grad():
                inr_recon = backbone.decode(coords_full.unsqueeze(0), z).reshape(VOLUME_SIZE)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                z_vae = vae.encode(x)
                vae_recon = vae.decode(z_vae)

            gt_np = _to01(x[0, 0].cpu().numpy())
            inr_np = _to01(inr_recon.clamp(-1, 1).cpu().numpy())
            vae_np = _to01(vae_recon[0, 0].clamp(-1, 1).float().cpu().numpy())
            fg_mask = gt_np > 0.025

            inr_nrmse = compute_nrmse(inr_np, gt_np, mask=fg_mask)
            inr_ssim = compute_ssim(inr_np, gt_np)
            vae_nrmse = compute_nrmse(vae_np, gt_np, mask=fg_mask)
            vae_ssim = compute_ssim(vae_np, gt_np)

            fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
            for row, (plane_name, sl) in enumerate(slices.items()):
                gt_sl, inr_sl, vae_sl = gt_np[sl], inr_np[sl], vae_np[sl]
                diff_inr = np.abs(inr_sl - gt_sl)
                diff_vae = np.abs(vae_sl - gt_sl)

                axes[row, 0].imshow(np.rot90(gt_sl), cmap="gray", vmin=0, vmax=1)
                axes[row, 0].set_ylabel(plane_name, fontsize=11)
                axes[row, 1].imshow(np.rot90(inr_sl), cmap="gray", vmin=0, vmax=1)
                axes[row, 2].imshow(np.rot90(vae_sl), cmap="gray", vmin=0, vmax=1)
                im_d = axes[row, 3].imshow(np.rot90(np.maximum(diff_inr, diff_vae)), cmap="hot", vmin=0, vmax=0.5)
                if row == 0:
                    axes[row, 0].set_title("GT", fontsize=12)
                    axes[row, 1].set_title(f"INR (SIREN)\nnRMSE={inr_nrmse:.3f} SSIM={inr_ssim:.3f}", fontsize=11)
                    axes[row, 2].set_title(f"MedVAE\nnRMSE={vae_nrmse:.3f} SSIM={vae_ssim:.3f}", fontsize=11)
                    axes[row, 3].set_title("|diff| max(INR,MedVAE)", fontsize=11)
                for ax in axes[row]:
                    ax.set_xticks([]); ax.set_yticks([])

            fig.suptitle(f"Capacité de représentation — sujet {sid}, {field}, {MODALITY} (auto-reconstruction, pas de traduction de champ)",
                         fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            out_path = OUT_DIR / f"repr_{MODALITY}_{sid}_{field}.png"
            fig.savefig(out_path, dpi=130)
            plt.close(fig)
            print(f"  [{sid} {field}] INR nRMSE={inr_nrmse:.4f} SSIM={inr_ssim:.4f} | "
                  f"MedVAE nRMSE={vae_nrmse:.4f} SSIM={vae_ssim:.4f} -> {out_path}")

    print(f"\nFigures sauvées dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
