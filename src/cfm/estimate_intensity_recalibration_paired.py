#!/usr/bin/env python3
"""Estime le recalage d'intensite par la formule ORACLE, sur un split APPARIE
tenu a l'ecart de l'evaluation.

Complement de `estimate_intensity_recalibration.py`, qui apparie des
DISTRIBUTIONS (aucun appariement requis, mais estimateur approche). Ici on
dispose des couples (prediction, verite) et on peut donc calculer le facteur
exact qui minimise ||a*p - g||^2 :

    a = <p, g> / <p, p> = (||g|| / ||p||) * cos(p, g)

Le rapport de normes seul, faute du cosinus, ne capte qu'une partie du signal
(correlation 0.441 mesuree le 2026-08-28) : c'est precisement ce terme que cette
voie recupere.

LE SPLIT COMPTE. `Validating_prospective` contient 3 sujets (0016/0017/0018)
distincts des 3 sujets d'evaluation (0006/0007/0009), avec verite terrain aux 5
champs et 3 contrastes, et n'avait jamais servi dans ce projet. Estimer dessus
puis appliquer a l'evaluation est donc un protocole propre — contrairement au
leave-one-subject-out du 2026-08-25, qui consommait les sujets d'evaluation
eux-memes.

Le script REFUSE d'estimer sur le split d'evaluation.

Usage :
    PYTHONPATH=src python src/cfm/estimate_intensity_recalibration_paired.py \\
        --pred-root outputs/mmfm/valid_rbest/predictions/task3 \\
        --split Validating_prospective \\
        --out configs/mmfm/intensity_recalibration_paired.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env

FORBIDDEN = "Training_prospective"   # le split d'EVALUATION


def _load(p: Path) -> np.ndarray:
    return np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32)).ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--split", default="Validating_prospective")
    ap.add_argument("--out", default="configs/mmfm/intensity_recalibration_paired.json")
    a = ap.parse_args()

    if a.split == FORBIDDEN:
        raise SystemExit(
            f"--split {FORBIDDEN} est le split d'EVALUATION. Estimer le recalage "
            f"dessus puis y mesurer le gain serait circulaire. Utiliser "
            f"Validating_prospective.")

    root = Path(load_env("local")["data_root"])
    pred_root = Path(a.pred_root)
    by_t: dict = defaultdict(lambda: defaultdict(list))
    by_p: dict = defaultdict(lambda: defaultdict(list))
    cosines: dict = defaultdict(lambda: defaultdict(list))
    n_pairs = 0

    for mdir in sorted(p for p in pred_root.iterdir() if p.is_dir()):
        mod = mdir.name
        for pdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            if "_to_" not in pdir.name:
                continue
            tgt = pdir.name.split("_to_")[1]
            for f in sorted(pdir.glob("*.nii.gz")):
                sid = f.name.replace(".nii.gz", "").split("_")[-1]
                g = root / a.split / mod / tgt / f"P_{mod}_{tgt}_{sid}.nii.gz"
                if not g.exists():
                    print(f"  [verite manquante] {g}")
                    continue
                p, gg = _load(f), _load(g)
                pp = float(p @ p)
                if pp < 1e-12:
                    continue
                fac = float(p @ gg) / pp
                cos = float(p @ gg) / max(np.linalg.norm(p) * np.linalg.norm(gg), 1e-12)
                by_t[mod][tgt].append(fac)
                by_p[mod][pdir.name].append(fac)
                cosines[mod][tgt].append(cos)
                n_pairs += 1

    if not n_pairs:
        raise SystemExit("aucun couple (prediction, verite) trouve.")

    def med(d):
        return {m: {k: float(np.median(v)) for k, v in per.items()} for m, per in d.items()}

    fac_t, fac_p = med(by_t), med(by_p)

    print(f"Facteurs ORACLE ({a.split}, {n_pairs} couples), par champ cible")
    print(f"  {'':10s}{'champ':7s}{'facteur':>9s}{'ecart-type':>11s}{'cos(p,g)':>10s}{'n':>4s}")
    for mod in sorted(fac_t):
        for tgt in ["0.1T", "1.5T", "3T", "5T", "7T"]:
            if tgt not in fac_t[mod]:
                continue
            v = np.array(by_t[mod][tgt])
            print(f"  {mod:10s}{tgt:7s}{fac_t[mod][tgt]:9.3f}{v.std():11.3f}"
                  f"{np.median(cosines[mod][tgt]):10.3f}{len(v):4d}")

    out = {
        "statistic": "oracle_lstsq",
        "estimated_from": {"split": a.split, "pred_root": str(a.pred_root),
                           "n_pairs": n_pairs},
        "note": ("Facteurs a = <p,g>/<p,p>, mediane par champ cible et par paire, "
                 "estimes sur un split APPARIE distinct de celui de l'evaluation. "
                 "Appliques a la prediction APRES denormalisation."),
        "by_target_field": fac_t,
        "by_pair": fac_p,
        "median_cosine": med(cosines),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\necrit -> {a.out}")


if __name__ == "__main__":
    main()
