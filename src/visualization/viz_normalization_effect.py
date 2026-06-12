#!/usr/bin/env python3
"""
Visualisation de l'effet de la normalisation p0.5/p99.5 sur les sujets
prospectifs 0006, 0007, 0009 — Training_prospective.

3 figures (T1W / T2W / T2FLAIR) :
  - 2 lignes par sujet : brut (affiché clampé p0.5–p99.5) et normalisé [0,1]
  - 5 colonnes : 0.1T → 1.5T → 3T → 5T → 7T
  - Annotation dans chaque case : valeurs p0.5 et p99.5 brutes,
    et dynamique (p99.5 − p0.5)

Sortie : results/benchmark_vae/analysis/normalization_effect/norm_{T1W,T2W,T2FLAIR}.png

Usage :
  python src/visualization/viz_normalization_effect.py
"""

from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT = Path("/home/rousseau/Data/MRIxFields_20260414/Training_prospective")
OUT = Path("results/benchmark_vae/analysis/normalization_effect")
OUT.mkdir(parents=True, exist_ok=True)

# ── Paramètres ───────────────────────────────────────────────────────────────
SUBJECTS = ["0006", "0007", "0009"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]

# ── Boucle par modalité ──────────────────────────────────────────────────────
for mod in MODALITIES:
    n_subj = len(SUBJECTS)
    n_field = len(FIELDS)

    # 2 lignes par sujet (brut / normalisé), 5 colonnes (champs)
    fig, axes = plt.subplots(
        n_subj * 2,
        n_field,
        figsize=(3.2 * n_field, 2.8 * n_subj * 2),
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        f"Normalisation p0.5 / p99.5  —  {mod}  —  sujets 0006 / 0007 / 0009",
        fontsize=13,
        fontweight="bold",
    )

    for s, subj in enumerate(SUBJECTS):
        for c, field in enumerate(FIELDS):
            ax_raw = axes[s * 2][c]
            ax_norm = axes[s * 2 + 1][c]

            path = ROOT / mod / field / f"P_{mod}_{field}_{subj}.nii.gz"

            if not path.exists():
                for ax in (ax_raw, ax_norm):
                    ax.text(
                        0.5,
                        0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=10,
                    )
                    ax.axis("off")
                continue

            vol = nib.load(str(path)).get_fdata(dtype=np.float32)
            lo = float(np.percentile(vol, 0.5))
            hi = float(np.percentile(vol, 99.5))
            sl = np.rot90(vol[:, :, vol.shape[2] // 2])

            # ── Brut : affiché avec clamp [lo, hi] ──────────────────────────
            ax_raw.imshow(
                np.clip(sl, lo, hi),
                cmap="gray",
                vmin=lo,
                vmax=hi,
                aspect="equal",
                interpolation="nearest",
            )
            ax_raw.axis("off")
            ax_raw.text(
                0.02,
                0.98,
                f"p0.5  = {lo:.3f}\np99.5 = {hi:.3f}",
                transform=ax_raw.transAxes,
                fontsize=7,
                va="top",
                color="white",
                bbox=dict(fc="black", alpha=0.6, pad=1.5, boxstyle="round"),
            )

            # ── Normalisé [0, 1] ─────────────────────────────────────────────
            norm_sl = np.clip((sl - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
            ax_norm.imshow(
                norm_sl,
                cmap="gray",
                vmin=0,
                vmax=1,
                aspect="equal",
                interpolation="nearest",
            )
            ax_norm.axis("off")
            ax_norm.text(
                0.02,
                0.98,
                f"range = {hi - lo:.3f}",
                transform=ax_norm.transAxes,
                fontsize=7,
                va="top",
                color="white",
                bbox=dict(fc="black", alpha=0.6, pad=1.5, boxstyle="round"),
            )

            # ── Titre colonne (1ère paire de lignes seulement) ───────────────
            if s == 0:
                ax_raw.set_title(field, fontsize=10, fontweight="bold")

            # ── Label ligne (1ère colonne seulement) ─────────────────────────
            if c == 0:
                ax_raw.text(
                    -0.03,
                    0.5,
                    f"{subj}\nbrut",
                    transform=ax_raw.transAxes,
                    fontsize=8,
                    va="center",
                    ha="right",
                    rotation=90,
                    color="#222",
                )
                ax_norm.text(
                    -0.03,
                    0.5,
                    f"{subj}\nnorm.",
                    transform=ax_norm.transAxes,
                    fontsize=8,
                    va="center",
                    ha="right",
                    rotation=90,
                    color="#222",
                )

    plt.tight_layout(rect=[0.04, 0, 1, 0.97])
    out_path = OUT / f"norm_{mod}.png"
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {out_path}")

print(f"\nFigures dans : {OUT}")
