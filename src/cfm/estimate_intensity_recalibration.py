#!/usr/bin/env python3
"""Estime le recalage d'intensite SANS appariement, sur les sujets d'entrainement.

LE PROBLEME (mesure le 2026-08-25, `results/mmfm/qualitative_20260824/manifest.md`
section 4). L'inference denormalise par un `hi` FIXE par (modalite, champ cible),
calcule sur les 1939 volumes d'entrainement : chaque sujet recoit l'echelle
MOYENNE de sa classe. Le facteur multiplicatif global qu'un oracle choisirait
enleve alors **80 % de l'energie de l'erreur** en T1W et en T2FLAIR. Le biais est
systematique par champ cible : les predictions sont trop sombres vers les champs
bas-moyens et trop claires vers 5T et 7T.

CE QUI BLOQUAIT. Estimer ce facteur demande des couples (prediction, verite) au
meme champ, et il n'existe que 3 sujets apparies dans tout le jeu. Le protocole
leave-one-subject-out du 2026-08-25 donnait 0.3794 -> 0.3231 (-15 %), mais sur
n = 2 par estimation, et il consomme les sujets d'evaluation.

LA VOIE UTILISEE ICI, laissee ouverte par ce manifeste et jamais instruite : le
biais est une difference entre la DISTRIBUTION d'intensite des volumes predits a
un champ donne et celle des volumes REELS au meme champ. Comparer deux
distributions ne demande AUCUN appariement -- les 143 sujets retrospectifs par
classe suffisent, et aucun sujet d'evaluation n'entre dans l'estimation.

POURQUOI UN RAPPORT DE NORMES. Le facteur oracle est `a = <p,g> / <p,p>`, qui
demande l'appariement. Si la prediction a la bonne forme et la mauvaise echelle
(`g ~ a*p`), alors `a = ||g|| / ||p||` : un rapport de statistiques de
DISTRIBUTION, estimable de chaque cote separement. C'est aussi la norme par
laquelle le nRMSE divise, donc la quantite directement pertinente. Le script
calcule aussi le rapport des moyennes sur le cerveau, comme variante.

Usage :
    PYTHONPATH=src python src/cfm/estimate_intensity_recalibration.py \\
        --pred-root outputs/mmfm/calib_rbest/predictions/task3 \\
        --split Training_retrospective \\
        --out configs/mmfm/intensity_recalibration.json
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

FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]


def _load(p: Path) -> np.ndarray:
    return np.asarray(nib.load(str(p)).get_fdata(dtype=np.float32))


def _stats(v: np.ndarray) -> dict:
    """Deux statistiques de niveau, calculees sur UN volume.

    `l2` : la norme, celle par laquelle le nRMSE divise — c'est la quantite dont
    le rapport approche le facteur oracle quand seule l'echelle est fausse.
    `brain` : moyenne sur le cerveau approche, moins sensible a la taille du fond
    mais aussi moins directement liee a la metrique.
    """
    m = v > np.percentile(v, 60)
    return {"l2": float(np.linalg.norm(v)),
            "brain": float(v[m].mean()) if m.any() else 0.0}


def real_levels(root: Path, split: str, modalities: list[str], limit: int) -> dict:
    out: dict = {}
    for mod in modalities:
        out[mod] = {}
        for f in FIELDS:
            d = root / split / mod / f
            if not d.exists():
                continue
            files = sorted(d.glob("*.nii.gz"))[:limit]
            s = [_stats(_load(p)) for p in files]
            if not s:
                continue
            out[mod][f] = {k: float(np.median([x[k] for x in s])) for k in ("l2", "brain")}
            out[mod][f]["n"] = len(s)
            print(f"  reel  {mod:8s} {f:6s} n={len(s):3d}  "
                  f"l2={out[mod][f]['l2']:9.1f}  cerveau={out[mod][f]['brain']:.4f}")
    return out


def pred_levels(pred_root: Path, modalities: list[str]) -> tuple[dict, dict]:
    """Niveaux des PREDICTIONS, agreges par champ cible et par paire."""
    by_tgt: dict = defaultdict(lambda: defaultdict(list))
    by_pair: dict = defaultdict(lambda: defaultdict(list))
    for mod in modalities:
        mdir = pred_root / mod
        if not mdir.exists():
            continue
        for pdir in sorted(mdir.iterdir()):
            if not pdir.is_dir() or "_to_" not in pdir.name:
                continue
            src, tgt = pdir.name.split("_to_")
            for p in sorted(pdir.glob("*.nii.gz")):
                st = _stats(_load(p))
                by_tgt[mod][tgt].append(st)
                by_pair[mod][pdir.name].append(st)

    def med(d):
        out = {}
        for mod, per in d.items():
            out[mod] = {}
            for k, lst in per.items():
                out[mod][k] = {m: float(np.median([x[m] for x in lst]))
                               for m in ("l2", "brain")}
                out[mod][k]["n"] = len(lst)
        return out

    return med(by_tgt), med(by_pair)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True,
                    help="racine des predictions de calibration (.../predictions/task3)")
    ap.add_argument("--split", default="Training_retrospective",
                    help="split ou lire les volumes REELS — jamais celui de l'evaluation")
    ap.add_argument("--modalities", nargs="+", default=["T1W", "T2W", "T2FLAIR"])
    ap.add_argument("--limit", type=int, default=40,
                    help="volumes reels par classe pour la mediane")
    ap.add_argument("--stat", default="l2", choices=["l2", "brain"])
    ap.add_argument("--out", default="configs/mmfm/intensity_recalibration.json")
    a = ap.parse_args()

    if "prospective" in a.split:
        raise SystemExit(
            f"--split {a.split} : les volumes reels doivent venir des sujets "
            f"d'ENTRAINEMENT. Utiliser Training_retrospective, sans quoi le "
            f"recalage est ajuste sur les donnees d'evaluation.")

    root = Path(load_env("local")["data_root"])
    print(f"Niveaux REELS ({a.split}, mediane sur <= {a.limit} sujets par classe)")
    real = real_levels(root, a.split, a.modalities, a.limit)
    print(f"\nNiveaux PREDITS ({a.pred_root})")
    ptgt, ppair = pred_levels(Path(a.pred_root), a.modalities)

    fac_tgt: dict = {}
    fac_pair: dict = {}
    print(f"\nFacteurs de recalage (statistique '{a.stat}') — reel / predit")
    for mod in a.modalities:
        if mod not in real or mod not in ptgt:
            continue
        fac_tgt[mod] = {}
        print(f"  {mod}")
        for f in FIELDS:
            if f not in real[mod] or f not in ptgt[mod]:
                continue
            r, p = real[mod][f][a.stat], ptgt[mod][f][a.stat]
            fac_tgt[mod][f] = float(r / p) if p > 1e-12 else 1.0
            print(f"    cible {f:6s} : {fac_tgt[mod][f]:6.3f}   "
                  f"(n predit {ptgt[mod][f]['n']}, n reel {real[mod][f]['n']})")
        fac_pair[mod] = {}
        for pair, st in sorted(ppair.get(mod, {}).items()):
            tgt = pair.split("_to_")[1]
            if tgt not in real[mod]:
                continue
            r, p = real[mod][tgt][a.stat], st[a.stat]
            fac_pair[mod][pair] = float(r / p) if p > 1e-12 else 1.0

    out = {
        "statistic": a.stat,
        "estimated_from": {"split": a.split, "pred_root": str(a.pred_root),
                           "limit": a.limit},
        "note": ("Facteurs multiplicatifs appliques a la prediction APRES "
                 "denormalisation. Estimes par appariement de DISTRIBUTIONS entre "
                 "volumes predits et volumes reels au meme champ, sur les sujets "
                 "d'entrainement uniquement — aucun sujet d'evaluation n'y entre."),
        "by_target_field": fac_tgt,
        "by_pair": fac_pair,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\necrit -> {a.out}")


if __name__ == "__main__":
    main()
