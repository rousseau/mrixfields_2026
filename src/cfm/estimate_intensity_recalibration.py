#!/usr/bin/env python3
"""Estime la recalibration d'intensite SANS appariement, sur les sujets d'entrainement.

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

# GARDE-FOU — seuil sur le rapport de contraste (predit / reel).
#
# L'estimateur vaut ||g||/||p||, alors que le facteur oracle vaut
# (||g||/||p||) * cos(p, g). L'approximation ne tient que si la prediction a
# approximativement la bonne FORME. Quand elle est plate, cos est petit, le
# rapport de normes explose, et multiplier par un facteur gonfle ne restaure
# rien -- il amplifie une bouillie.
#
# Mesure du 2026-08-30, sur les deux cas connus :
#   R-best (la recalibration GAGNE, nRMSE 0.3737 -> 0.3525) : rapport 0.701-1.282
#   INR    (elle DEGRADE,           nRMSE 0.3749 -> 0.4690) : rapport 0.277-0.777
# et les cellules T2W de l'INR, ou le degat etait le pire (+0.2357, 2 victoires
# sur 20), sont a 0.277-0.593.
#
# 0.65 accepte les 15 cellules de R-best (minimum 0.701) et rejette 13 des 15
# cellules de l'INR, dont TOUTES celles de T2W.
MIN_CONTRAST_RATIO = 0.65


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
    mean = float(v[m].mean()) if m.any() else 0.0
    return {"l2": float(np.linalg.norm(v)),
            "brain": mean,
            # Contraste RELATIF, sans dimension : invariant a l'echelle, donc il
            # mesure la FORME et non le niveau. C'est ce qui permet de detecter
            # qu'un ecart n'est PAS une simple erreur d'echelle.
            "contrast": (float(v[m].std()) / mean) if (m.any() and mean > 1e-9) else 0.0}


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
            out[mod][f] = {k: float(np.median([x[k] for x in s])) for k in ("l2", "brain", "contrast")}
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
                               for m in ("l2", "brain", "contrast")}
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
    ap.add_argument("--min-contrast-ratio", type=float, default=MIN_CONTRAST_RATIO,
                    help="en dessous, le facteur est force a 1.0 : la prediction est "
                         "trop plate pour qu'un scalaire corrige quoi que ce soit "
                         "(0 pour desactiver le garde-fou)")
    ap.add_argument("--out", default="configs/mmfm/intensity_recalibration.json")
    a = ap.parse_args()

    if "prospective" in a.split:
        raise SystemExit(
            f"--split {a.split} : les volumes reels doivent venir des sujets "
            f"d'ENTRAINEMENT. Utiliser Training_retrospective, sans quoi le "
            f"la recalibration serait ajustee sur les donnees d'evaluation.")

    root = Path(load_env("local")["data_root"])
    print(f"Niveaux REELS ({a.split}, mediane sur <= {a.limit} sujets par classe)")
    real = real_levels(root, a.split, a.modalities, a.limit)
    print(f"\nNiveaux PREDITS ({a.pred_root})")
    ptgt, ppair = pred_levels(Path(a.pred_root), a.modalities)

    fac_tgt: dict = {}
    fac_pair: dict = {}
    blocked: dict = defaultdict(list)
    print(f"\nFacteurs de recalibration (statistique '{a.stat}') — reel / predit")
    for mod in a.modalities:
        if mod not in real or mod not in ptgt:
            continue
        fac_tgt[mod] = {}
        print(f"  {mod}")
        for f in FIELDS:
            if f not in real[mod] or f not in ptgt[mod]:
                continue
            r, p = real[mod][f][a.stat], ptgt[mod][f][a.stat]
            fac = float(r / p) if p > 1e-12 else 1.0
            cr = (ptgt[mod][f]["contrast"] / real[mod][f]["contrast"]
                  if real[mod][f]["contrast"] > 1e-9 else 0.0)
            if cr < a.min_contrast_ratio:
                blocked[mod].append(f)
                fac_tgt[mod][f] = 1.0
                print(f"    cible {f:6s} : {fac:6.3f} -> 1.000  BLOQUE "
                      f"(contraste {cr:.3f} < {a.min_contrast_ratio:.2f} : la prediction "
                      f"est trop plate, l'ecart n'est pas une erreur d'echelle)")
            else:
                fac_tgt[mod][f] = fac
                print(f"    cible {f:6s} : {fac:6.3f}   contraste {cr:.3f}   "
                      f"(n predit {ptgt[mod][f]['n']}, n reel {real[mod][f]['n']})")
        fac_pair[mod] = {}
        for pair, st in sorted(ppair.get(mod, {}).items()):
            tgt = pair.split("_to_")[1]
            if tgt not in real[mod]:
                continue
            if tgt in blocked[mod]:
                fac_pair[mod][pair] = 1.0   # le blocage du champ cible s'herite
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
        **out_extra,
    }
    n_block = sum(len(v) for v in blocked.values())
    if n_block:
        print(f"\n  {n_block} champ(s) cible(s) BLOQUE(S) — facteur force a 1.000 :")
        for mod, fs in blocked.items():
            if fs:
                print(f"    {mod} : {', '.join(fs)}")
        print("  Motif : le contraste relatif des predictions y est trop faible ; "
              "l'ecart\n  a la verite n'est pas une erreur d'echelle et un facteur "
              "multiplicatif\n  amplifierait une image plate. Mesure sur l'INR le "
              "2026-08-30 : sans ce\n  garde-fou, nRMSE 0.3749 -> 0.4690 (19 victoires "
              "sur 60, p = 0.006).")

    out_extra = {"blocked_target_fields": {k: v for k, v in blocked.items() if v},
                 "min_contrast_ratio": a.min_contrast_ratio}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\necrit -> {a.out}")


if __name__ == "__main__":
    main()
