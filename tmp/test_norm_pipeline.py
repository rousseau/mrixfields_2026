#!/usr/bin/env python3
"""
Test de la chaîne de normalisation complète :
  MedVAE fine-tuné  |  train_cfm_3d [-1,1]  |  inférence [0,1] bug vs corrigé
"""

import os
import sys

sys.path.insert(0, "src")
os.environ["HF_HUB_OFFLINE"] = "1"

import nibabel as nib
import numpy as np
import torch

# ── MedVAE fine-tuné ─────────────────────────────────────────────────────────
from medvae import MVAE
from scipy.ndimage import zoom as scipy_zoom

# ── utilitaires de cfm ───────────────────────────────────────────────────────
from cfm.train_cfm_3d import _center_crop_or_pad_np, _resample_volume

model_ft = MVAE(model_name="medvae_4_1_3d", modality="mri")
ckpt_path = "outputs/medvae/runs/medvae_finetune_all/weights/model_best.pth"
raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model_ft.load_state_dict(raw["model"] if "model" in raw else raw, strict=False)
model_ft.eval()
print("=== MedVAE fine-tuné chargé ===")

# ── Charger un vrai volume 0.1T ──────────────────────────────────────────────
nii_path = "/home/rousseau/Data/MRIxFields_20260414/Training_prospective/T1W/0.1T/P_T1W_0.1T_0006.nii.gz"
img = nib.load(nii_path)
vol = img.get_fdata(dtype=np.float32)
orig_sp = np.abs(np.diag(img.affine)[:3])
vol = _resample_volume(vol, orig_sp, (1.0, 1.0, 1.0))
print(
    f"Volume brut   : shape={vol.shape}  min={vol.min():.1f}  max={vol.max():.1f}  mean={vol.mean():.1f}"
)

lo = np.percentile(vol, 0.5)
hi = np.percentile(vol, 99.5)
print(f"Percentiles   : lo(0.5%)={lo:.2f}  hi(99.5%)={hi:.2f}")

# ── Trois variantes de normalisation ─────────────────────────────────────────
# A) CORRECTE : percentile → [-1, 1]   (entraînement CFM + inférence après fix)
vol_A = np.clip((vol - lo) / max(hi - lo, 1e-8), 0.0, 1.0) * 2.0 - 1.0
vol_A = _center_crop_or_pad_np(vol_A, (128, 128, 80))

# B) BUGGUÉE  : percentile → [0, 1]    (inférence AVANT fix)
vol_B = np.clip((vol - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
vol_B = _center_crop_or_pad_np(vol_B, (128, 128, 80))

# C) MedVAE orig : min-max global → [0,1] → [-1,1]  (pipeline HuggingFace)
v_min, v_max = vol.min(), vol.max()
vol_C = (vol - v_min) / max(v_max - v_min, 1e-8) * 2.0 - 1.0
vol_C = _center_crop_or_pad_np(vol_C, (128, 128, 80))

print(
    f"\nVariante A (correct [-1,1]) : min={vol_A.min():.4f}  max={vol_A.max():.4f}  mean={vol_A.mean():.4f}"
)
print(
    f"Variante B (bug    [ 0,1]) : min={vol_B.min():.4f}  max={vol_B.max():.4f}  mean={vol_B.mean():.4f}"
)
print(
    f"Variante C (HF     [-1,1]) : min={vol_C.min():.4f}  max={vol_C.max():.4f}  mean={vol_C.mean():.4f}"
)

# ── Encodage avec le modèle fine-tuné ────────────────────────────────────────
tA = torch.from_numpy(vol_A).unsqueeze(0).unsqueeze(0).float()
tB = torch.from_numpy(vol_B).unsqueeze(0).unsqueeze(0).float()
tC = torch.from_numpy(vol_C).unsqueeze(0).unsqueeze(0).float()

with torch.no_grad():
    zA = model_ft.encode(tA)
    zB = model_ft.encode(tB)
    zC = model_ft.encode(tC)

print(f"\n=== LATENTS (modèle fine-tuné) ===")
print(
    f"  zA (input correct [-1,1]) : shape={tuple(zA.shape)}  min={zA.min():.4f}  max={zA.max():.4f}  mean={zA.mean():.4f}  std={zA.std():.4f}"
)
print(
    f"  zB (input bug     [ 0,1]) : shape={tuple(zB.shape)}  min={zB.min():.4f}  max={zB.max():.4f}  mean={zB.mean():.4f}  std={zB.std():.4f}"
)
print(
    f"  zC (input HF      [-1,1]) : shape={tuple(zC.shape)}  min={zC.min():.4f}  max={zC.max():.4f}  mean={zC.mean():.4f}  std={zC.std():.4f}"
)
print(
    f"  |zA - zB| MAE = {(zA - zB).abs().mean():.4f}  max={(zA - zB).abs().max():.4f}"
)
print(
    f"  |zA - zC| MAE = {(zA - zC).abs().mean():.4f}  max=(zA - zC).abs().max()={(zA - zC).abs().max():.4f}"
)

# ── Decode + round-trip ───────────────────────────────────────────────────────
with torch.no_grad():
    rA = model_ft.decode(zA).squeeze().numpy()
    rB = model_ft.decode(zB).squeeze().numpy()

print(f"\n=== DECODE (round-trip) ===")
print(f"  decode(zA): min={rA.min():.4f}  max={rA.max():.4f}  mean={rA.mean():.4f}")
print(f"  decode(zB): min={rB.min():.4f}  max={rB.max():.4f}  mean={rB.mean():.4f}")

# MAE round-trip dans l'espace [-1,1]
print(f"  MAE round-trip A ([-1,1] ↔ decode): {np.abs(rA - vol_A).mean():.4f}")
print(f"  MAE round-trip B ([ 0,1] ↔ decode): {np.abs(rB - vol_B).mean():.4f}")

# Après dénorm (dec+1)/2 → [0,1]
predA_01 = np.clip((rA + 1.0) / 2.0, 0.0, 1.0)
predB_01 = np.clip((rB + 1.0) / 2.0, 0.0, 1.0)
refA_01 = np.clip((vol_A + 1.0) / 2.0, 0.0, 1.0)  # reference en [0,1]

print(f"\n=== APRÈS DÉNORM (dec+1)/2 → [0,1] ===")
print(
    f"  predA: min={predA_01.min():.4f}  max={predA_01.max():.4f}  mean={predA_01.mean():.4f}  MAE/ref={np.abs(predA_01 - refA_01).mean():.4f}"
)
print(
    f"  predB: min={predB_01.min():.4f}  max={predB_01.max():.4f}  mean={predB_01.mean():.4f}  MAE/ref={np.abs(predB_01 - refA_01).mean():.4f}"
)

print("\nDone.")
