#!/usr/bin/env python3
"""Rendre VISIBLE le résultat central : le flow translate.

CE QUE LA FIGURE MONTRE. On retire de chaque déplacement sa moyenne sur les
sujets ; ce qui reste — `D_i − D̄` — est par construction la part PROPRE au sujet :

  · rangée du haut    les trois volumes source ;
  · rangée du milieu  part propre au sujet du déplacement PRÉDIT ;
  · rangée du bas     part propre au sujet du VRAI déplacement.

Les deux rangées de déplacement partagent la MÊME échelle de couleur, sans quoi la
comparaison ne voudrait rien dire.

ATTENTION À L'INTERPRÉTATION. Deux légendes successives de cette figure ont été
retirées pour cause d'affirmation non soutenue par ce qu'elle montre :
« la rangée du milieu est presque vide » (extrapolée de la mesure en espace LATENT,
3.5 % contre 71.7 % le 2026-09-02) puis « contours contre niveau régional ». Les
deux sont FAUSSES sur au moins une cellule. Le décodeur étant non linéaire, une
translation en latent produit un déplacement d'image qui dépend du sujet : **la
mesure en espace latent ne se transpose pas simplement en espace image.**

La figure porte donc deux quantités mesurées, et aucune thèse :

  `NIVEAU` — moyenne spatiale de la part propre au sujet sur le cerveau. Un
      décalage de niveau global donne une valeur élevée ; un motif centré sur zéro,
      presque rien. Le rapport vrai/prédit dit si le modèle sous-produit ce
      décalage.
  `corr` — corrélation spatiale entre part propre PRÉDITE et VRAIE.

Ce qui est mesuré, et qui est hétérogène (2026-09-03) :

  T1W    0.1T→7T   corr +0.842   niveau vrai/prédit 0.5
  T2W    0.1T→7T   corr +0.482   niveau vrai/prédit 1.9
  T2FLAIR 0.1T→7T  corr +0.875   niveau vrai/prédit 4.4
  T1W    0.1T→3T   corr +0.042   niveau vrai/prédit 8.7

Sur les sauts extrêmes le modèle produit une individualité corrélée à la vraie ;
sur la paire intermédiaire T1W 0.1T→3T la corrélation s'effondre et le décalage de
niveau manquant est d'un facteur 9. **Consigné sans hypothèse** — ce projet a
réfuté cinq explications formulées trop vite, dont deux à propos de cette figure.

Usage :
    PYTHONPATH=src python src/cfm/figures_translation.py
    PYTHONPATH=src python src/cfm/figures_translation.py --modality T2W --pair 0.1T_to_3T
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from cfm.eval_qualitative_3arch import brain_bbox, gt_path, load_vol, pred_path

SUBJECTS = ["0006", "0007", "0009"]
PRED_ROOTS = {
    "Vectorisé R-best": "outputs/mmfm/vec_rbest/predictions/task3",
    "Vectorisé (production)": "outputs/mmfm/vectorized/predictions/task3",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="T1W")
    ap.add_argument("--pair", default="0.1T_to_7T")
    ap.add_argument("--method", default="Vectorisé R-best", choices=list(PRED_ROOTS))
    ap.add_argument("--slice", type=int, default=None)
    ap.add_argument("--outdir", default="results/mmfm/qualitative_20260903")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args()

    src_f, tgt_f = a.pair.split("_to_")
    root = Path(load_env("local")["data_root"])
    pred_root = PRED_ROOTS[a.method]

    src, pred, true = [], [], []
    for sid in SUBJECTS:
        s = load_vol(gt_path(root, a.modality, src_f, sid))
        p = load_vol(pred_path(pred_root, a.modality, a.pair, tgt_f, sid))
        g = load_vol(gt_path(root, a.modality, tgt_f, sid))
        src.append(s); pred.append(p); true.append(g)
    src, pred, true = np.stack(src), np.stack(pred), np.stack(true)

    # Déplacements, puis retrait de la moyenne SUR LES SUJETS : ce qui reste est
    # par construction la part propre au sujet.
    Dp = pred - src
    Dt = true - src
    Rp = Dp - Dp.mean(0, keepdims=True)
    Rt = Dt - Dt.mean(0, keepdims=True)

    # chiffre affiché : la même quantité que la mesure du 2026-09-02
    own_p = float(np.mean([np.linalg.norm(r) for r in Rp]) /
                  max(np.linalg.norm(Dp.mean(0)), 1e-12))
    own_t = float(np.mean([np.linalg.norm(r) for r in Rt]) /
                  max(np.linalg.norm(Dt.mean(0)), 1e-12))

    # Ce qui distingue les deux : le VRAI ecart individuel est un decalage de
    # NIVEAU (moyenne spatiale elevee), le predit un motif de CONTOURS (centre
    # sur zero). Et se correspondent-ils seulement ?
    mask = true[0] > np.percentile(true[0], 60)
    lvl_p = float(np.mean([abs(r[mask].mean()) for r in Rp]))
    lvl_t = float(np.mean([abs(r[mask].mean()) for r in Rt]))
    corrs = [float(np.corrcoef(Rp[i][mask], Rt[i][mask])[0, 1]) for i in range(len(SUBJECTS))]
    corr = float(np.mean(corrs))

    bb = brain_bbox(true[0])
    z = a.slice if a.slice is not None else (bb[2].start + bb[2].stop) // 2

    def cut(v):
        return np.rot90(v[bb[0], bb[1], z])

    # ÉCHELLE COMMUNE aux deux rangées de déplacement — sans elle la comparaison
    # serait une illusion d'optique.
    vmax = float(np.percentile(np.abs(np.concatenate([Rp.ravel(), Rt.ravel()])), 99.5))

    fig, axes = plt.subplots(3, 3, figsize=(9.4, 9.9))
    for j, sid in enumerate(SUBJECTS):
        axes[0, j].imshow(cut(src[j]), cmap="gray")
        axes[0, j].set_title(f"sujet {sid}", fontsize=10)
        im1 = axes[1, j].imshow(cut(Rp[j]), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        im2 = axes[2, j].imshow(cut(Rt[j]), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(3):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    axes[0, 0].set_ylabel(f"source {src_f}", fontsize=10)
    axes[1, 0].set_ylabel("déplacement PRÉDIT\npart propre au sujet", fontsize=10)
    axes[2, 0].set_ylabel("déplacement VRAI\npart propre au sujet", fontsize=10)

    fig.suptitle(
        f"{a.modality} — {src_f} → {tgt_f}   ·   {a.method}\n"
        f"Part propre au sujet : {own_p:.1%} (prédit) contre {own_t:.1%} (vrai)   ·   "
        f"corrélation {corr:+.3f}   ·   niveau manquant ×{lvl_t/max(lvl_p,1e-12):.1f}",
        fontsize=11.5, y=0.985)
    fig.text(0.5, 0.016,
             "Rangées 2 et 3 à échelle de couleur IDENTIQUE : le déplacement moins sa "
             "moyenne sur les 3 sujets, donc ce qui distingue un sujet d'un autre.\n"
             f"Décalage de NIVEAU (moyenne spatiale, cerveau) — prédit {lvl_p:.4f}, "
             f"vrai {lvl_t:.4f}, soit un facteur {lvl_t/max(lvl_p,1e-12):.1f}.   "
             f"Corrélation spatiale prédit/vrai : {corr:+.3f}.\n"
             "Aucune thèse n'est plaquée sur ces images : la mesure en espace latent "
             "(translation à 3.5 % près) ne se transpose pas\nsimplement ici, le "
             "décodeur étant non linéaire. Voir le manifeste pour les quatre cellules "
             "mesurées.",
             ha="center", fontsize=8.4, color="0.35")

    cb = fig.colorbar(im2, ax=axes[1:, :], fraction=0.028, pad=0.015)
    cb.set_label("écart au déplacement moyen", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    name = out / f"translation_{a.modality}_{a.pair}.png"
    fig.savefig(name, dpi=a.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  part propre au sujet — prédit {own_p:.3f}  |  vrai {own_t:.3f}")
    print(f"  decalage de NIVEAU   — prédit {lvl_p:.4f}  |  vrai {lvl_t:.4f}  "
          f"(facteur {lvl_t/max(lvl_p,1e-12):.1f})")
    print(f"  correlation spatiale predit/vrai : {corr:+.3f}  "
          f"(par sujet : {', '.join(f'{c:+.3f}' for c in corrs)})")
    print(f"  ecrit : {name}")


if __name__ == "__main__":
    main()
