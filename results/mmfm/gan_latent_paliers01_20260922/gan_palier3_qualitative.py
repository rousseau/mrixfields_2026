#!/usr/bin/env python3
"""Panel qualitatif Palier 3 (GAN latent) : sujet 0006, 3 paires, T1W.

Script AUTONOME plutot qu'un ajout au registre METHODS de
eval_qualitative_3arch.py -- diagnostic ponctuel de ce palier, pas une
architecture a comparer durablement dans l'outil partage des 3 architectures.
Reutilise load_mid_slice_aligned/normalize (visualization/common.py) tels
quels.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/home/rousseau/Exp/mrixfields_2026/src")
from visualization.common import load_mid_slice_aligned, normalize

DATA = Path.home() / "Data/MRIxFields_20260414/Training_prospective/T1W"
GAN_ROOT = Path("outputs/mmfm/vectorized_trajectory_gan/predictions_smoke_cc_20260922/task3/T1W")
BASE_ROOT = Path("outputs/mmfm/vectorized_trajectory/predictions_cc_baseline_20260916/task3/T1W")
SUBJECT = "0006"
PAIRS = ["3T_to_7T", "0.1T_to_7T", "0.1T_to_1.5T"]
AXIS = 2  # axial, coherent avec le classement officiel (voir feedback_zclip_metrique_officielle)

fig, axes = plt.subplots(len(PAIRS), 6, figsize=(18, 3.2 * len(PAIRS)))
col_titles = ["Source", "Vérité terrain", "Témoin (sans GAN)", "GAN (Palier 3)",
              "Erreur témoin", "Erreur GAN"]

for row, pair in enumerate(PAIRS):
    src_field, tgt_field = pair.split("_to_")
    src_path = DATA / src_field / f"P_T1W_{src_field}_{SUBJECT}.nii.gz"
    tgt_path = DATA / tgt_field / f"P_T1W_{tgt_field}_{SUBJECT}.nii.gz"
    base_path = BASE_ROOT / pair / f"P_T1W_{tgt_field}_{SUBJECT}.nii.gz"
    gan_path = GAN_ROOT / pair / f"P_T1W_{tgt_field}_{SUBJECT}.nii.gz"

    src = normalize(load_mid_slice_aligned(src_path, tgt_path, axis=AXIS))
    tgt = normalize(load_mid_slice_aligned(tgt_path, tgt_path, axis=AXIS))
    base = normalize(load_mid_slice_aligned(base_path, tgt_path, axis=AXIS))
    gan = normalize(load_mid_slice_aligned(gan_path, tgt_path, axis=AXIS))

    err_base = np.abs(base - tgt)
    err_gan = np.abs(gan - tgt)
    err_vmax = float(np.percentile(err_base, 99))

    imgs = [src, tgt, base, gan, err_base, err_gan]
    cmaps = ["gray"] * 4 + ["inferno", "inferno"]
    for col, (im, cmap) in enumerate(zip(imgs, cmaps)):
        ax = axes[row, col]
        vmax = err_vmax if col >= 4 else 1.0
        ax.imshow(np.rot90(im), cmap=cmap, vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0:
            ax.set_title(col_titles[col], fontsize=11)
    axes[row, 0].set_ylabel(pair, fontsize=11)

fig.suptitle(f"Palier 3 GAN latent — sujet {SUBJECT}, T1W — inspection qualitative ciblée", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])

out = Path("results/mmfm/gan_latent_paliers01_20260922/qualitative_palier3_0006_T1W.png")
fig.savefig(out, dpi=150)
print(f"Ecrit -> {out}")
