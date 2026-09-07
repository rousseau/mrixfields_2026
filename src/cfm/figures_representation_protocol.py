#!/usr/bin/env python3
"""Figure : ce que coûte le PROTOCOLE, et ce que coûte le VAE — côte à côte.

POURQUOI CETTE FIGURE EXISTE
La campagne du 2026-09-07 a établi que **60 % du « plafond de représentation »
(0.1140 sur la tranche notée) ne vient pas du VAE mais du protocole** : la chaîne
complète avec l'IDENTITÉ à la place du VAE coûte déjà 0.0685. Aucune figure du
dépôt ne le montre. Or c'est exactement le genre de résultat qu'un tableau seul ne
fait pas admettre : tant qu'on n'a pas vu la carte d'erreur du témoin sans VAE, on
attribue spontanément tout le défaut au modèle.

CE QUE LA FIGURE MONTRE, par cellule (contraste × champ × sujet) :

  colonne 1  vérité terrain, sur la grille native 0.5 mm
  colonne 2  PROTOCOLE SEUL — 1 mm, écrêtage, crop, retour 0.5 mm, SANS AUCUN VAE
  colonne 3  production — le même protocole AVEC MedVAE pré-entraîné
  colonne 4  protocole seul AVEC `hi` ×1.25 — le plancher de la recette corrigée
  colonne 5  LPIPS + `hi` ×1.25 — le correctif mesuré à −21 %

  Les colonnes 4 et 5 vont par PAIRE, et c'est la raison d'être de la colonne 4 :
  comparer directement 5 à 2 mélangerait deux changements (le VAE ET l'écrêtage), et
  laisserait croire qu'ajouter un VAE fait BAISSER l'erreur sous le témoin sans VAE.
  Chaque VAE doit être lu contre le plancher de SA propre recette.

  rangée 1   les images
  rangée 2   |x − vérité|, TOUTES SUR LA MÊME ÉCHELLE, nRMSE en titre

Trois choix qui ne sont pas cosmétiques :

  1. **La coupe est tirée de la tranche NOTÉE** `[150, 180)` (common/io.py
     ::Z_CLIP_RANGE), pas du milieu du volume. Le classement ne note que ces 30
     coupes sur 364 ; une figure prise ailleurs illustre une région qui ne compte pas.
  2. **L'échelle d'erreur est commune aux trois colonnes**, témoin protocole inclus.
     Une carte d'erreur seule paraît toujours mauvaise — c'est la comparaison au
     témoin qui informe, et c'est le principe déjà retenu par
     `eval_qualitative_3arch.py --mode panel` (qui, lui, place le témoin IDENTITÉ).
  3. **Le cadrage suit la boîte du cerveau.** 82 % du volume est du fond plat :
     afficher le FOV entier laisse l'œil sur ce qui est trivial.

Les volumes viennent de `bench_representation.run_cell(..., return_volumes=True)` —
donc du MÊME code que les chiffres publiés, pas d'un chemin parallèle qui finirait
par illustrer autre chose que ce qu'on a mesuré.

Usage :
    PYTHONPATH=src python src/cfm/figures_representation_protocol.py \
        --cells T1W@7T,T2FLAIR@3T --outdir results/mmfm/representation_20260906
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from common.io import Z_CLIP_RANGE
from cfm.bench_representation import ARMS, FIELD_NORM_STATS, SPLIT_DIR, nrmse, run_cell

# Les trois colonnes, dans l'ordre de lecture : le temoin d'abord, puis ce qu'on lui
# ajoute. `protocol` n'utilise AUCUN VAE (Arm.use_vae=False).
DEFAULT_ARMS = ["protocol", "prod", "protocol_hi125", "lpips_hi125"]
LABELS = {
    "protocol":    "protocole SEUL\n(sans aucun VAE)",
    "prod":        "production\n(MedVAE pre-entraine)",
    "lpips_hi125": "LPIPS + hi x1.25\n(le correctif)",
    "protocol_hi125": "protocole seul\navec hi x1.25",
    "lpips":       "MedVAE LPIPS",
    "prod_hi125":  "production + hi x1.25",
}


def brain_bbox(vol: np.ndarray, margin: int = 6):
    """Boite englobante du cerveau, pour ne pas cadrer sur du fond."""
    m = vol > (vol.max() * 0.02)
    if not m.any():
        return tuple(slice(0, s) for s in vol.shape[:2])
    out = []
    for ax in range(2):
        idx = np.where(m.any(axis=tuple(a for a in range(3) if a != ax)))[0]
        out.append(slice(max(0, int(idx[0]) - margin),
                         min(vol.shape[ax], int(idx[-1]) + 1 + margin)))
    return tuple(out)


def make_figure(mod: str, fld: str, sid: str, arm_names, outdir: Path, device) -> Path:
    root = Path(load_env("local")["data_root"]) / SPLIT_DIR
    nii = root / mod / fld / f"P_{mod}_{fld}_{sid}.nii.gz"
    if not nii.exists():
        raise FileNotFoundError(nii)
    stats = json.load(open(FIELD_NORM_STATS))["stats"][mod][fld]

    gt = None
    cols = []
    for name in arm_names:
        m, gt_v, pred = run_cell(ARMS[name], nii, stats["lo"], stats["hi"], device,
                                 return_volumes=True)
        gt = gt_v if gt is None else gt
        cols.append((name, pred, m["nrmse_slab"]))
        print(f"  {name:14s} nRMSE tranche notee {m['nrmse_slab']:.4f}", flush=True)

    # Coupe tiree de la tranche REELLEMENT notee, pas du milieu du volume.
    z0, z1 = Z_CLIP_RANGE
    z = (z0 + z1) // 2
    by, bx = brain_bbox(gt)
    sl = lambda v: v[by, bx, z]

    gt_s = sl(gt)
    errs = [np.abs(sl(p) - gt_s) for _, p, _ in cols]
    # ECHELLE D'ERREUR COMMUNE — c'est tout l'objet de la figure.
    # Bornee au p99.5 de l'union des cartes, et non au MAXIMUM : quelques points
    # chauds isoles (bords de tissu) valent 4x l'erreur typique et ecraseraient toute
    # la structure vers le noir. Le seuil est affiche, et les valeurs au-dela sont
    # saturees — pas masquees.
    emax = float(np.percentile(np.concatenate([e.ravel() for e in errs]), 99.5)) if errs else 1.0
    vmax = float(np.percentile(gt_s, 99.5))

    n = len(cols) + 1
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.6))
    fig.suptitle(
        f"{mod} @ {fld}, sujet {sid} — coupe z={z} de la tranche notee {Z_CLIP_RANGE}\n"
        f"Ce que coute le protocole seul, avant tout VAE",
        fontsize=11)

    axes[0, 0].imshow(gt_s.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    axes[0, 0].set_title("verite terrain", fontsize=10)
    axes[1, 0].axis("off")
    axes[1, 0].text(0.5, 0.5,
                    f"echelle d'erreur\ncommune aux {len(cols)} colonnes\n"
                    f"0 a {emax:.3g}  (p99.5,\nau-dela : sature)",
                    ha="center", va="center", fontsize=9, transform=axes[1, 0].transAxes)

    for k, ((name, _, nr), e) in enumerate(zip(cols, errs), start=1):
        axes[0, k].imshow(sl(cols[k - 1][1]).T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
        axes[0, k].set_title(LABELS.get(name, name), fontsize=10)
        im = axes[1, k].imshow(e.T, cmap="inferno", vmin=0, vmax=emax, origin="lower")
        axes[1, k].set_title(f"|erreur|   nRMSE {nr:.4f}", fontsize=10)
        if k == len(cols):
            fig.colorbar(im, ax=axes[1, k], fraction=0.046)

    for a in axes.ravel():
        if a.images:
            a.set_xticks([]); a.set_yticks([])

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"protocole_{mod}_{fld}_{sid}.png"
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default="T1W@7T,T2FLAIR@3T",
                    help="liste MOD@CHAMP separee par des virgules. Defaut : les deux "
                         "cellules ou le diagnostic differe le plus (T1W@7T = 34 %% de "
                         "l'erreur imputable au VAE, T2FLAIR@3T = 11 %%).")
    ap.add_argument("--subject", default="0006")
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--outdir", default="results/mmfm/representation_20260906")
    a = ap.parse_args()

    names = [s.strip() for s in a.arms.split(",")]
    bad = [n for n in names if n not in ARMS]
    if bad:
        raise SystemExit(f"variantes inconnues : {bad}\ndisponibles : {list(ARMS)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cell in [c.strip() for c in a.cells.split(",")]:
        mod, fld = cell.split("@")
        print(f"=== {mod} @ {fld}, sujet {a.subject}")
        out = make_figure(mod, fld, a.subject, names, Path(a.outdir), device)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
