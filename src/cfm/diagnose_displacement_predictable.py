#!/usr/bin/env python3
"""La composante INDIVIDUELLE du déplacement est-elle prédictible depuis la source ?

LA QUESTION, et pourquoi c'est la dernière. Le 2026-09-02 a mesuré que le vrai
transport champ→champ est propre au sujet à **71.7 %** quand le flow appris ne
l'est qu'à 3.5 %. Le 2026-09-03 a fermé la piste du couplage : il n'y a aucune
correspondance à trouver dans les données non appariées, en aucune dimension.
Reste une seule issue — si cette composante est une **fonction du volume source**,
un modèle peut l'apprendre sans couplage, en conditionnant sur `z_src`.

LE PIÈGE À ÉVITER. Il n'existe que 3 sujets appariés dans tout le jeu. Ajuster une
matrice 129 024 × 129 024, ou même un vecteur, interpolerait trivialement : trois
points dans un espace de cette taille sont toujours parfaitement explicables. Tout
résultat obtenu ainsi ne voudrait rien dire.

LA FORME TESTÉE, À UN SEUL PARAMÈTRE. Si la carte était approximativement affine
en intensité — `z_cible ≈ a·z_source + b` — alors

    D_i − D̄ = (a − 1)·(z_i − z̄)

et la composante individuelle serait **entièrement prédictible par un scalaire**.
On ajuste donc UN `α` par (contraste, paire de champs), par moindres carrés sur
les écarts centrés. Un paramètre contre 3 × 129 024 dimensions de résidu : le R²
est honnête.

ET UN VRAI TEST HORS ÉCHANTILLON. `α` ajusté sur 2 sujets, prédiction du
troisième, trois plis. Le chiffre qui compte est le rapport

    ‖D_k − D̂_k‖ / ‖D_k − D̄_ajust‖

c'est-à-dire : **le terme dépendant de la source fait-il mieux que la translation
seule ?** Inférieur à 1 = oui.

CONTRÔLE INDÉPENDANT. Si la forme affine est la bonne, `α + 1` doit retrouver le
facteur d'échelle d'intensité mesuré indépendamment le 2026-08-25 (oracle par
volume, 0.63 à 1.45 selon la paire). Deux mesures sans rapport qui concordent
valent mieux qu'un R² seul.

Usage :
    PYTHONPATH=src python src/cfm/diagnose_displacement_predictable.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE = Path("outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/pro_train")
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]


def fit_alpha(dz: torch.Tensor, dD: torch.Tensor) -> float:
    """α minimisant Σ‖dD_i − α·dz_i‖² — forme fermée, UN paramètre."""
    num = float((dD * dz).sum())
    den = float((dz * dz).sum())
    return num / den if den > 1e-12 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    cache = Path(a.cache)
    base = cache.parent.parent
    idx = json.load(open(cache / "index.json"))["samples"]

    def load(mod, field):
        out = {}
        for r in idx:
            if r["modality"] == mod and r["field"] == field:
                sid = Path(r["path"]).stem.split("_")[-1]
                o = torch.load(base / r["path"], map_location="cpu", weights_only=False)
                z = o["latent"] if isinstance(o, dict) and "latent" in o else o
                out[sid] = torch.as_tensor(z).double().reshape(-1)
        return out

    zs = {(m, f): load(m, f) for m in MODALITIES for f in FIELDS}
    print(f"Cache : {cache}")
    print(f"{len(zs[(MODALITIES[0], FIELDS[0])])} sujets appariés\n")

    rows = []
    print(f"  {'contraste':9s}{'paire':15s}{'alpha':>8s}{'alpha+1':>9s}"
          f"{'R2 (1 param)':>14s}{'LOO / translation':>19s}")
    for mod in MODALITIES:
        for i in range(len(FIELDS)):
            for j in range(len(FIELDS)):
                if i == j:
                    continue
                A, B = zs[(mod, FIELDS[i])], zs[(mod, FIELDS[j])]
                sids = sorted(set(A) & set(B))
                if len(sids) < 3:
                    continue
                Z = torch.stack([A[s] for s in sids])
                D = torch.stack([B[s] - A[s] for s in sids])

                # --- dans l'echantillon, 1 parametre -------------------------
                dz = Z - Z.mean(0, keepdim=True)
                dD = D - D.mean(0, keepdim=True)
                al = fit_alpha(dz, dD)
                res = dD - al * dz
                r2 = 1.0 - float((res ** 2).sum() / (dD ** 2).sum())

                # --- hors echantillon : ajuste sur 2, predit le 3e ----------
                ratios = []
                for k in range(len(sids)):
                    keep = [x for x in range(len(sids)) if x != k]
                    Zf, Df = Z[keep], D[keep]
                    zbar, dbar = Zf.mean(0, keepdim=True), Df.mean(0, keepdim=True)
                    alk = fit_alpha(Zf - zbar, Df - dbar)
                    pred = dbar.squeeze(0) + alk * (Z[k] - zbar.squeeze(0))
                    err_model = float((D[k] - pred).norm())
                    err_trans = float((D[k] - dbar.squeeze(0)).norm())
                    ratios.append(err_model / max(err_trans, 1e-12))
                loo = float(np.mean(ratios))
                rows.append({"modality": mod, "pair": f"{FIELDS[i]}_to_{FIELDS[j]}",
                             "alpha": al, "r2_insample": r2, "loo_ratio": loo})
                if FIELDS[i] == "0.1T":
                    print(f"  {mod:9s}{FIELDS[i]+'->'+FIELDS[j]:15s}{al:8.3f}{al+1:9.3f}"
                          f"{r2:14.3f}{loo:19.3f}")

    r2s = np.array([r["r2_insample"] for r in rows])
    loos = np.array([r["loo_ratio"] for r in rows])
    als = np.array([r["alpha"] for r in rows])
    print(f"\n  sur les {len(rows)} cellules (20 paires x 3 contrastes) :")
    print(f"    R2 dans l'echantillon, 1 parametre : moyenne {r2s.mean():.3f} "
          f"(min {r2s.min():.3f}, max {r2s.max():.3f})")
    print(f"    LOO, erreur modele / translation   : moyenne {loos.mean():.3f} "
          f"(min {loos.min():.3f}, max {loos.max():.3f})")
    print(f"    cellules ou le modele BAT la translation hors echantillon : "
          f"{int((loos < 1).sum())}/{len(loos)}")
    print(f"    alpha + 1 : moyenne {(als+1).mean():.3f} "
          f"(min {(als+1).min():.3f}, max {(als+1).max():.3f})")
    print(f"    [controle : les facteurs d'echelle d'intensite oracles mesures")
    print(f"     independamment le 2026-08-25 vont de 0.63 a 1.45]")

    from scipy.stats import binomtest, wilcoxon
    nb = int((loos < 1).sum())
    print(f"\n    signes (LOO < 1) : p = {binomtest(nb, len(loos), 0.5).pvalue:.4f}")
    print(f"    Wilcoxon (LOO contre 1) : p = {wilcoxon(loos - 1).pvalue:.4f}")

    print()
    if r2s.mean() > 0.5 and loos.mean() < 0.9:
        print("  => La composante individuelle EST largement prédictible depuis la")
        print("     source, et par un SEUL scalaire. Un modele conditionne sur z_src")
        print("     peut donc l'apprendre sans aucun couplage : c'est un levier reel.")
    elif r2s.mean() > 0.5:
        print("  => Bon R2 dans l'echantillon mais rien hors echantillon : avec 3")
        print("     sujets, c'est le signe d'un ajustement qui ne generalise pas.")
        print("     Ne pas conclure.")
    else:
        print("  => La composante individuelle n'est PAS prédictible sous cette forme.")
        print("     La carte champ->champ n'est pas affine en intensite, et la piste")
        print("     du conditionnement sur z_src ne tient pas ainsi.")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"cells": rows,
                   "mean": {"r2_insample": float(r2s.mean()),
                            "loo_ratio": float(loos.mean()),
                            "alpha_plus_1": float((als + 1).mean()),
                            "loo_wins": int((loos < 1).sum()), "n": len(loos)}},
                  open(a.json_out, "w"), indent=2)
        print(f"\n  ecrit -> {a.json_out}")


if __name__ == "__main__":
    main()
