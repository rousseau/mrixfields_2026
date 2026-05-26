#!/usr/bin/env python3
"""Test VAE output range."""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, "src")
import torch
import yaml

from cfm.train_cfm_3d import load_vae

cfg_path = "configs/cfm3d_T1W_medvae_0p1T_7T.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
cfg["data"]["data_root"] = "/home/rousseau/Data/MRIxFields_20260414"

device = torch.device("cpu")  # CPU pour test
print("Device:", device)

print("Chargement VAE ...")
vae = load_vae(cfg, device)
vae.eval()
print(f"  VAE latent_channels={vae.latent_channels}")

# Load one GT 7T
gt_path = Path(
    "/home/rousseau/Data/MRIxFields_20260414/Validating_prospective/T1W/7T/P_T1W_7T_0016.nii.gz"
)
vol = nib.load(str(gt_path)).get_fdata().astype(np.float32)

# Preprocess (exactement comme dans le pipeline)
from cfm.run_inference_viz_cfm3d import _center_crop_or_pad_np, _resample_volume

orig_spacing = np.array([0.5, 0.5, 0.5])
target_spacing = np.array([1.0, 1.0, 1.0])
factors = orig_spacing / target_spacing
vol_res = _resample_volume(vol, orig_spacing, target_spacing)
lo, hi = np.percentile(vol_res, [0.5, 99.5])
vol_n = np.clip((vol_res - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
vol_c = _center_crop_or_pad_np(vol_n, (128, 128, 80))

# Encode (VAE attend [-1,1])
vol_t = torch.from_numpy(vol_c * 2 - 1).unsqueeze(0).unsqueeze(0).float()  # [-1,1]

print(
    f"Input (0-1): min={vol_c.min():.4f}, max={vol_c.max():.4f}, mean={vol_c.mean():.4f}"
)

with torch.no_grad():
    z = vae.encode(vol_t)
    recon = vae.decode(z)

recon_np = recon.squeeze().numpy()
print()
print("=== Encode-Decode direct VAE ===")
print(f"Z shape: {z.shape}")
print(
    f"Recon (raw from VAE): min={recon_np.min():.4f}, max={recon_np.max():.4f}, mean={recon_np.mean():.4f}"
)

# Vérifier si le VAE produit déjà en [0,1] ou [-1,1]
print()
print("=== Vérification de la sortie du VAE ===")
if recon_np.min() >= -1.0 and recon_np.max() <= 1.0:
    print("✓ Le VAE produit des valeurs dans [-1, 1] (attendu)")
    recon_01 = np.clip((recon_np + 1) / 2, 0, 1)
else:
    print("⚠ Le VAE produit des valeurs hors de [-1, 1]")
    print(f"  Valeurs réelles: min={recon_np.min():.4f}, max={recon_np.max():.4f}")

print(
    f"Recon en [0,1] (via (x+1)/2): min={recon_01.min():.4f}, max={recon_01.max():.4f}, mean={recon_01.mean():.4f}"
)

# Comparer avec GT preprocess
print()
print("=== Comparaison ===")
print(f"Input (0-1): mean={vol_c.mean():.4f}")
print(f"Recon (0-1): mean={recon_01.mean():.4f}")
print(f"MAE (recon vs input): {np.mean(np.abs(recon_01 - vol_c)):.4f}")
