#!/usr/bin/env python3
"""Consolidation QUANTITATIVE des trois architectures MMFM.

Trois choses que les CSV publies ne contiennent pas, et qui changent la lecture
des chiffres :

  --mode identity   Temoin « ne rien faire » (recopier le volume source) sur les
                    TROIS contrastes. Il n'existait que pour T1W. Sans lui, un
                    nRMSE de 0.36 en T2W et de 0.44 en T1W semblent dire que T2W
                    est mieux traite ; c'est faux tant qu'on ne sait pas ce que
                    coutait la paire au depart.
  --mode sharpness  Energie haute frequence conservee par rapport a la verite.
                    Met un nombre sur le flou, que nRMSE et SSIM recompensent
                    au lieu de le penaliser (un predicteur qui rend la moyenne
                    conditionnelle minimise l'erreur quadratique).
  --mode scaled_identity
                    Temoin « recopier la source, rééchelonnee par UN scalaire »,
                    estime en leave-one-out sur les autres sujets apparies (donc
                    deployable). C'est LE temoin a battre : 80 % de l'energie de
                    l'erreur de ce pipeline est une erreur d'echelle, et un seul
                    nombre par paire vaut 0.2274 sur le volume entier contre
                    0.3795 pour le modele de production. L'identite brute (0.6235)
                    place la barre trop bas et credite au modele un gain qu'un
                    scalaire obtient sans apprentissage.
  --mode table      Tableau final : 3 architectures x 3 contrastes, gain sur le
                    temoin, taux de victoire paire a paire.

RÉGION ÉVALUÉE (--region, defaut `slab`). Le classement ne note PAS le volume
entier : il note la tranche axiale [150, 180), forme (364, 436, 30) — voir
common/io.py::Z_CLIP_RANGE. Tous les chiffres du CHANGELOG anterieurs au
2026-09-04 ont ete mesures sur les 364 coupes et ne sont donc pas comparables au
classement (ecart mesure sur le vectorise : 0.021 en nRMSE, 0.059 en SSIM). La
region est ecrite dans chaque CSV produit.

Les metriques du temoin sont recalculees ici avec les formules EXACTES de
l'evaluateur officiel (Evaluation/evaluate.py : nRMSE = ||p-g||/||g|| sur le
volume entier sans masque ; SSIM moyennee par coupe axiale avec data_range issu
de la verite). `--mode identity` verifie d'abord cette reimplementation contre le
temoin T1W deja produit par l'evaluateur officiel : si l'ecart depasse la
tolerance, le script s'arrete au lieu de publier des chiffres non comparables.

Usage :
    PYTHONPATH=src python src/cfm/eval_quantitative_3arch.py --mode identity
    PYTHONPATH=src python src/cfm/eval_quantitative_3arch.py --mode sharpness
    PYTHONPATH=src python src/cfm/eval_quantitative_3arch.py --mode table
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import json

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from common.io import Z_CLIP_RANGE, apply_z_clip
from cfm.eval_qualitative_3arch import (
    METHODS, brain_bbox, gt_path, hf_ratio, load_metrics, load_vol, pred_path,
    spectrum_of_volume,
)

SUBJECTS = ["0006", "0007", "0009"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
PAIRS = [f"{a}_to_{b}" for a in FIELDS for b in FIELDS if a != b]

# Répertoire de sortie PAR DÉFAUT seulement — surchargeable par `--outdir`.
# Codé en dur, il a fait écraser les CSV d'une campagne précédente par une
# nouvelle : les chiffres cités par son manifeste n'étaient plus étayés par les
# fichiers du dépôt. Toujours passer --outdir pour une nouvelle campagne.
DEFAULT_OUTDIR = Path("results/mmfm/qualitative_20260824")
OUTDIR = DEFAULT_OUTDIR
FIELD_NORM_STATS = Path("configs/mmfm/field_norm_stats.json")
REF_IDENTITY_T1W = Path("results/mmfm/comparison_20260813_unet_adagn/task3_identity_baseline_T1W.csv")


# --------------------------------------------------------------------------- #
#  Formules officielles, reimplementees
# --------------------------------------------------------------------------- #

# RÉGION NOTÉE. « slab » = la tranche axiale [150, 180) que le classement note
# réellement (voir common/io.py::Z_CLIP_RANGE) ; « full » = le volume entier, ce
# que TOUS les chiffres du CHANGELOG antérieurs au 2026-09-04 mesuraient. Le
# défaut est `slab` : c'est la seule région comparable au classement. `full`
# reste accessible pour rejouer un chiffre historique, jamais pour en publier
# un nouveau. Positionné par --region dans main().
REGION = "slab"


def _region(vol: np.ndarray) -> np.ndarray:
    """Restreint un volume à la région évaluée (voir REGION)."""
    return vol if REGION == "full" else apply_z_clip(vol)


def official_nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    p, t = _region(pred).astype(np.float64), _region(target).astype(np.float64)
    n = np.linalg.norm(t)
    return float(np.linalg.norm(p - t) / n) if n > 1e-10 else 0.0


def official_ssim(pred: np.ndarray, target: np.ndarray, slice_axis: int = 2) -> float:
    p, t = _region(pred).astype(np.float64), _region(target).astype(np.float64)
    dr = t.max() - t.min()
    if dr < 1e-10:
        return 1.0
    vals = []
    for i in range(p.shape[slice_axis]):
        s = [slice(None)] * p.ndim
        s[slice_axis] = i
        s = tuple(s)
        if t[s].max() - t[s].min() < 1e-10:
            continue
        vals.append(structural_similarity(p[s], t[s], data_range=dr))
    return float(np.mean(vals)) if vals else 1.0


class VolumeCache:
    """Les 45 volumes (3 contrastes x 5 champs x 3 sujets) sont relus en moyenne
    quatre fois par le balayage des 20 paires ; les garder coute ~10 Go de RAM et
    divise par quatre le temps de disque."""

    def __init__(self, root: Path, limit: int = 60):
        self.root, self.limit, self.d = root, limit, {}

    def __call__(self, mod: str, field: str, sid: str) -> np.ndarray:
        k = (mod, field, sid)
        if k not in self.d:
            if len(self.d) >= self.limit:
                self.d.pop(next(iter(self.d)))
            self.d[k] = load_vol(gt_path(self.root, mod, field, sid))
        return self.d[k]


# --------------------------------------------------------------------------- #
#  Mode identity
# --------------------------------------------------------------------------- #

def _aggregate(per_pair: Dict[str, List[Tuple[float, float]]],
               method: str = "identity") -> List[dict]:
    rows = []
    for pair in PAIRS:
        vals = per_pair.get(pair, [])
        if not vals:
            continue
        n = np.array([v[0] for v in vals]); s = np.array([v[1] for v in vals])
        rows.append({"method": method, "task": "task3", "pair": pair,
                     # La RÉGION entre dans le CSV : deux campagnes mesurées sur
                     # des régions différentes ne sont pas comparables, et rien
                     # dans un CSV ne le signalait jusqu'ici.
                     "region": _region_label(),
                     "n_subjects": len(vals),
                     "nrmse_mean": float(n.mean()), "ssim_mean": float(s.mean()),
                     "nrmse_std": float(n.std()), "ssim_std": float(s.std())})
    return rows


def _region_label() -> str:
    if REGION == "full":
        return "full_volume"
    z0, z1 = Z_CLIP_RANGE
    return f"slab_z{z0}_{z1}"


def _check_against_official(cache, check_pairs, tol: float) -> None:
    """Garde-fou : nos formules doivent retrouver le témoin produit par
    l'évaluateur officiel.

    Le CSV de référence (`REF_IDENTITY_T1W`) a été produit sur le VOLUME ENTIER.
    La conformité des formules se vérifie donc en `full`, quelle que soit la
    région sur laquelle on publie ensuite : ce test valide l'implémentation des
    métriques, pas le choix de la région.
    """
    global REGION
    ref = load_metrics(str(REF_IDENTITY_T1W))
    if not ref:
        raise SystemExit(f"temoin officiel de reference introuvable : {REF_IDENTITY_T1W}")
    saved, REGION = REGION, "full"
    try:
        print(f"verification des formules contre l'evaluateur officiel "
              f"({REF_IDENTITY_T1W.name}, volume entier) :")
        worst = 0.0
        for pair in check_pairs:
            src_f, tgt_f = pair.split("_to_")
            ns, ss = [], []
            for sid in SUBJECTS:
                g = cache("T1W", tgt_f, sid); s_ = cache("T1W", src_f, sid)
                ns.append(official_nrmse(s_, g)); ss.append(official_ssim(s_, g))
            dn = abs(float(np.mean(ns)) - ref[pair][0])
            ds = abs(float(np.mean(ss)) - ref[pair][1])
            worst = max(worst, dn, ds)
            print(f"  {pair:14s} nRMSE {np.mean(ns):.5f} vs {ref[pair][0]:.5f} (d={dn:.2e})   "
                  f"SSIM {np.mean(ss):.5f} vs {ref[pair][1]:.5f} (d={ds:.2e})")
        if worst > tol:
            raise SystemExit(f"ECART {worst:.2e} > tolerance {tol:.0e} — reimplementation "
                             "non conforme a l'evaluateur officiel, on n'ecrit rien.")
        print(f"conforme (ecart max {worst:.2e} <= {tol:.0e})")
    finally:
        REGION = saved
    print(f"region publiee : {_region_label()}\n")


def mode_identity(args) -> None:
    root = Path(load_env("local")["data_root"])
    cache = VolumeCache(root)

    _check_against_official(cache, args.check_pairs, args.tol)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for mod in args.modalities:
        per_pair: Dict[str, List[Tuple[float, float]]] = {}
        for pair in PAIRS:
            src_f, tgt_f = pair.split("_to_")
            for sid in SUBJECTS:
                g = cache(mod, tgt_f, sid); s_ = cache(mod, src_f, sid)
                per_pair.setdefault(pair, []).append((official_nrmse(s_, g), official_ssim(s_, g)))
            v = per_pair[pair]
            print(f"  {mod:8s} {pair:14s} nRMSE={np.mean([x[0] for x in v]):.4f} "
                  f"SSIM={np.mean([x[1] for x in v]):.4f}", flush=True)
        rows = _aggregate(per_pair)
        out = OUTDIR / f"identity_baseline_{mod}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"  -> {out}   moyenne nRMSE={np.mean([r['nrmse_mean'] for r in rows]):.4f} "
              f"SSIM={np.mean([r['ssim_mean'] for r in rows]):.4f}\n", flush=True)


# --------------------------------------------------------------------------- #
#  Mode scaled_identity — le temoin qui manquait
# --------------------------------------------------------------------------- #


FG_THRESHOLD = 0.02  # seuil de foreground sur les donnees natives, deja dans [0, 1]


def _global_gain(src: np.ndarray, tgt: np.ndarray) -> float:
    """Facteur multiplicatif global au sens des moindres carres : le `a` qui
    minimise ||a*src - tgt||. Un SEUL nombre par volume."""
    s = _region(src).astype(np.float64)
    t = _region(tgt).astype(np.float64)
    den = float((s * s).sum())
    return float((s * t).sum() / den) if den > 1e-12 else 1.0


def _fg_level(vol: np.ndarray) -> float:
    """Niveau d'intensite moyen du foreground, sur la region evaluee.

    C'est l'information que `normalize_volume` DETRUIT : elle divise chaque
    volume par son propre p99.5, donc le reseau ne voit jamais ce nombre. Mesure
    du 2026-09-04 : corr(niveau du foreground de la source, gain oracle requis)
    = -0.76 / -0.70 / -0.88 selon le contraste. Le gain a appliquer est donc
    largement lisible dans la source elle-meme.
    """
    v = _region(vol).astype(np.float64)
    m = v > FG_THRESHOLD
    return float(v[m].mean()) if m.any() else 0.0


def mode_scaled_identity(args) -> None:
    """Temoin « recopier la source, rééchelonnée par UN scalaire ».

    Pourquoi ce temoin et pas seulement l'identite : 80 % de l'energie de
    l'erreur de ce pipeline est une erreur d'ECHELLE d'intensite (mesure du
    2026-08-25, confirmee le 2026-08-30 par le facteur oracle). Un temoin qui
    ignore cette composante place la barre beaucoup trop bas — l'identite brute
    vaut 0.6235, mais la meme source rééchelonnee par un unique scalaire vaut
    0.2274 sur le volume entier, soit MIEUX que le modele de production (0.3795).
    Comparer un modele a l'identite brute revient donc a lui crediter un gain
    qu'un seul nombre obtient sans apprentissage.

    Deux variantes sont ecrites :

      scaled_identity_loo    — le scalaire est estime en LEAVE-ONE-OUT sur les
                               AUTRES sujets apparies de la meme cellule
                               (contraste, paire). Aucune information sur la
                               cible du sujet evalue n'entre dans son calcul :
                               c'est un temoin DEPLOYABLE, et c'est celui qu'un
                               modele doit battre.
      scaled_identity_srclevel
                             — le scalaire est le rapport entre le niveau
                               ATTENDU du champ cible (moyenne leave-one-out des
                               autres sujets) et le niveau REELLEMENT OBSERVE
                               dans le volume source. Aussi deployable que le
                               LOO, mais il exploite le niveau absolu de la
                               source — precisement l'information que
                               `normalize_volume` supprime avant que le reseau
                               ne la voie. C'est le temoin qui rend le defaut 04
                               de l'audit falsifiable : si cet estimateur bat le
                               LOO nu, c'est que la source contient bien le
                               signal, et que le pipeline le jette.
      scaled_identity_oracle — le scalaire optimal du sujet lui-meme. NON
                               deployable ; c'est une borne superieure sur ce
                               qu'une correction purement globale peut donner,
                               utile pour separer « erreur d'echelle » et
                               « erreur de structure ».

ATTENTION, n = 3. Avec trois sujets apparies, un LOO sur une cellule qui
contient un outlier est tres bruite : mesure sur la tranche, le sujet 0009 a un
7T environ deux fois plus sombre que les deux autres (gains 7T_to_* de
[2.19, 2.66, 5.31]), et le LOO nu lui attribue 2.43 au lieu de 5.31. Les
cellules impliquant 7T portent donc l'essentiel de l'erreur du temoin LOO. Ne
pas lire ces temoins comme des methodes, mais comme des BORNES.

    Le scalaire est ajuste SUR LA REGION EVALUEE : optimiser une echelle sur une
    region et la noter sur une autre n'aurait pas de sens.
    """
    root = Path(load_env("local")["data_root"])
    cache = VolumeCache(root)
    _check_against_official(cache, args.check_pairs, args.tol)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, float]] = {}

    if len(SUBJECTS) < 2:
        raise SystemExit(
            f"Le temoin LOO exige au moins 2 sujets apparies ; reçu {len(SUBJECTS)}."
        )

    for mod in args.modalities:
        per_pair_loo: Dict[str, List[Tuple[float, float]]] = {}
        per_pair_src: Dict[str, List[Tuple[float, float]]] = {}
        per_pair_orc: Dict[str, List[Tuple[float, float]]] = {}
        for pair in PAIRS:
            src_f, tgt_f = pair.split("_to_")
            # gains oracle et niveaux de foreground de tous les sujets, pour le LOO
            gains = {sid: _global_gain(cache(mod, src_f, sid), cache(mod, tgt_f, sid))
                     for sid in SUBJECTS}
            lvl_tgt = {sid: _fg_level(cache(mod, tgt_f, sid)) for sid in SUBJECTS}
            for sid in SUBJECTS:
                src = cache(mod, src_f, sid)
                tgt = cache(mod, tgt_f, sid)
                a_loo = float(np.mean([gains[o] for o in SUBJECTS if o != sid]))
                # Niveau cible ATTENDU (LOO, aucune info sur la cible du sujet
                # evalue) rapporte au niveau OBSERVE dans sa propre source.
                lvl_src_obs = _fg_level(src)
                exp_tgt = float(np.mean([lvl_tgt[o] for o in SUBJECTS if o != sid]))
                a_src = (exp_tgt / lvl_src_obs) if lvl_src_obs > 1e-9 else 1.0
                for store, a in ((per_pair_loo, a_loo), (per_pair_src, a_src),
                                 (per_pair_orc, gains[sid])):
                    store.setdefault(pair, []).append(
                        (official_nrmse(a * src, tgt), official_ssim(a * src, tgt)))
            vl = per_pair_loo[pair]; vs = per_pair_src[pair]; vo = per_pair_orc[pair]
            print(f"  {mod:8s} {pair:14s} LOO={np.mean([x[0] for x in vl]):.4f} "
                  f"srclevel={np.mean([x[0] for x in vs]):.4f} "
                  f"oracle={np.mean([x[0] for x in vo]):.4f} "
                  f"| gains={[round(gains[s], 3) for s in SUBJECTS]}", flush=True)

        for name, per_pair in (("scaled_identity_loo", per_pair_loo),
                               ("scaled_identity_srclevel", per_pair_src),
                               ("scaled_identity_oracle", per_pair_orc)):
            rows = _aggregate(per_pair, method=name)
            out = OUTDIR / f"{name}_{mod}.csv"
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
            m_n = float(np.mean([r["nrmse_mean"] for r in rows]))
            m_s = float(np.mean([r["ssim_mean"] for r in rows]))
            summary.setdefault(mod, {})[name] = m_n
            print(f"  -> {out}   moyenne nRMSE={m_n:.4f} SSIM={m_s:.4f}", flush=True)
        print(flush=True)

    keys = ["scaled_identity_loo", "scaled_identity_srclevel", "scaled_identity_oracle"]
    labels = ["LOO (deployable)", "srclevel (deployable)", "oracle (borne)"]
    print(f"=== Temoins rééchelonnes, region {_region_label()} ===")
    print(f"{'contraste':10s}" + "".join(f"{l:>24s}" for l in labels))
    for mod in args.modalities:
        print(f"{mod:10s}" + "".join(f"{summary[mod][k]:24.4f}" for k in keys))
    if len(args.modalities) > 1:
        print(f"{'MOYENNE':10s}" + "".join(
            f"{np.mean([summary[m][k] for m in args.modalities]):24.4f}" for k in keys))


# --------------------------------------------------------------------------- #
#  Mode sharpness
# --------------------------------------------------------------------------- #

def _shrink_bbox(bbox, frac: float):
    """Reduit la boite au coeur du cerveau. Sert a separer deux causes de haute
    frequence : le detail anatomique (partout) et les artefacts de BORD. Les
    predictions de l'INR ont une frontiere cerveau/fond en escalier qui injecte
    de la puissance haute frequence sur toute la bande — sans ce recadrage, son
    indice de nettete est SUR-estime alors qu'elle est la plus floue a
    l'interieur (mesure : 0.49 -> 0.21 en T1W->7T, 0.60 -> 0.15 en T2FLAIR->7T)."""
    if frac >= 1.0:
        return bbox
    out = []
    for sl in bbox:
        c = (sl.start + sl.stop) // 2
        h = max(1, int((sl.stop - sl.start) * frac / 2))
        out.append(slice(c - h, c + h))
    return tuple(out)


def mode_sharpness(args) -> None:
    root = Path(load_env("local")["data_root"])
    rows = []
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for mod in args.modalities:
        for pair in args.pairs:
            src_f, tgt_f = pair.split("_to_")
            for sid in args.subjects:
                gt = load_vol(gt_path(root, mod, tgt_f, sid))
                bbox = _shrink_bbox(brain_bbox(gt), args.interior_frac)
                centers, p_gt = spectrum_of_volume(gt, bbox, args.n_slices)
                # le temoin identite passe par le meme calcul : sa nettete est
                # celle d'une VRAIE image, donc ~1 — ce qui sert d'echelle.
                _, p_id = spectrum_of_volume(load_vol(gt_path(root, mod, src_f, sid)), bbox, args.n_slices)
                rows.append({"modality": mod, "pair": pair, "subject": sid,
                             "method": "identite", "hf_ratio": hf_ratio(p_id, p_gt, centers)})
                for label, per_mod in METHODS:
                    if mod not in per_mod:
                        continue
                    p = pred_path(per_mod[mod][0], mod, pair, tgt_f, sid)
                    if not p.exists():
                        continue
                    _, p_pr = spectrum_of_volume(load_vol(p), bbox, args.n_slices)
                    rows.append({"modality": mod, "pair": pair, "subject": sid,
                                 "method": label, "hf_ratio": hf_ratio(p_pr, p_gt, centers)})
                print(f"  {mod:8s} {pair:14s} {sid} " + "  ".join(
                    f"{r['method']}={r['hf_ratio']:.3f}" for r in rows[-4:]), flush=True)
    tag = "cerveau_entier" if args.interior_frac >= 1.0 else f"interieur_{args.interior_frac:g}"
    out = OUTDIR / f"sharpness_index_{tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["modality", "pair", "subject", "method", "hf_ratio"])
        w.writeheader(); w.writerows(rows)
    print(f"\n-> {out}")
    print("\nratio HF moyen (1.0 = nettete de la verite ; <1 = flou)")
    names = ["identite"] + [m[0] for m in METHODS]
    print(f"  {'contraste':10s} " + "".join(f"{n:>12s}" for n in names))
    for mod in args.modalities:
        line = f"  {mod:10s} "
        for n in names:
            v = [r["hf_ratio"] for r in rows if r["modality"] == mod and r["method"] == n]
            line += f"{np.mean(v):>12.3f}" if v else f"{'-':>12s}"
        print(line)


# --------------------------------------------------------------------------- #
#  Mode calibration
# --------------------------------------------------------------------------- #

def _best_affine(p: np.ndarray, g: np.ndarray) -> Tuple[float, float]:
    """(a, b) minimisant ||a*p + b - g||^2, forme fermee."""
    n = p.size
    sp, sg = float(p.sum()), float(g.sum())
    spp, spg = float(p @ p), float(p @ g)
    den = n * spp - sp * sp
    if abs(den) < 1e-12:
        return 1.0, 0.0
    a = (n * spg - sp * sg) / den
    return a, (sg - a * sp) / n


def _field_hi() -> Dict[str, Dict[str, float]]:
    """hi par (modalite, champ) — les memes que l'inference utilise pour
    DE-normaliser (`lo` vaut 0 partout, la denormalisation est donc une simple
    multiplication par `hi`)."""
    with open(FIELD_NORM_STATS) as f:
        d = json.load(f)["stats"]
    return {m: {k: float(v["hi"]) for k, v in fields.items()} for m, fields in d.items()}


def subject_scale(src_vol: np.ndarray, hi_pop_src: float, pct: float = 99.5) -> float:
    """Facteur d'echelle propre au sujet, estime SANS voir la verite.

    L'inference denormalise avec `hi` calcule sur les 1939 volumes
    d'entrainement : chaque sujet recoit donc l'echelle MOYENNE de sa classe
    (modalite, champ). Si l'acquisition de ce sujet-la est globalement plus
    sombre ou plus claire que la moyenne, la prediction herite de l'erreur.
    Le volume SOURCE du meme sujet est disponible a l'inference et porte cette
    information : le rapport entre son centile et le `hi` de population de SON
    champ estime l'ecart du sujet a la moyenne, qu'on reporte sur la cible.
    """
    return float(np.percentile(src_vol, pct)) / max(hi_pop_src, 1e-12)


def _methods_for(args):
    """Les methodes a scorer : celles codees en dur, OU une racine arbitraire.

    Sans cela, mesurer l'erreur STRUCTURELLE d'un nouveau run exigerait de
    modifier `METHODS` a chaque fois. Or c'est desormais la mesure de travail :
    en nRMSE brut, 80 % de ce qu'on mesure est l'erreur d'echelle d'intensite,
    declaree inaccessible avec 3 sujets apparies (voir
    results/mmfm/recalib_20260829/manifest.md §8) -- tout progres structurel y est
    noye. C'est exactement ce qui est arrive a R-best : geometrie du flow reparee
    d'un facteur 100, score inchange.
    """
    if getattr(args, "pred_root", None):
        name = args.name or Path(args.pred_root).parts[-3]
        return [(name, {m: (args.pred_root, "") for m in MODALITIES})]
    return METHODS


def mode_calibration(args) -> None:
    """Quelle part de l'erreur est une simple erreur d'ECHELLE D'INTENSITE ?

    Les figures montrent des predictions anatomiquement justes mais
    radiometriquement fausses (trop plates vers 7T, trop nettes vers 0.1T). On
    corrige donc chaque prediction par la meilleure transformation d'intensite
    GLOBALE possible — celle qu'un oracle choisirait en connaissant la verite —
    et on regarde ce qu'il reste. Deux variantes :
      - echelle seule (a*p) : preserve le fond a zero, donc physiquement
        realisable par une simple renormalisation d'intensite ;
      - affine (a*p + b) : borne inferieure absolue de toute correction globale.
    L'ecart entre l'erreur brute et l'erreur corrigee est, par construction, la
    part de l'erreur qu'aucun progres d'architecture n'est necessaire pour
    supprimer.
    """
    root = Path(load_env("local")["data_root"])
    cache = VolumeCache(root)
    hi_pop = _field_hi()
    rows = []
    for mod in args.modalities:
        for pair in args.pairs:
            src_f, tgt_f = pair.split("_to_")
            for sid in args.subjects:
                g = cache(mod, tgt_f, sid).ravel().astype(np.float64)
                ng = float(np.linalg.norm(g))
                r_sub = subject_scale(cache(mod, src_f, sid), hi_pop[mod][src_f])
                for label, per_mod in _methods_for(args):
                    if mod not in per_mod:
                        continue
                    pth = pred_path(per_mod[mod][0], mod, pair, tgt_f, sid)
                    if not pth.exists():
                        continue
                    pr = load_vol(pth).ravel().astype(np.float64)
                    raw = float(np.linalg.norm(pr - g)) / ng
                    a_s = float(pr @ g) / max(float(pr @ pr), 1e-12)
                    sca = float(np.linalg.norm(a_s * pr - g)) / ng
                    a, b = _best_affine(pr, g)
                    aff = float(np.linalg.norm(a * pr + b - g)) / ng
                    rel = float(np.linalg.norm(r_sub * pr - g)) / ng
                    # p_sq / pg / g_sq permettent de recalculer le nRMSE pour
                    # N'IMPORTE QUEL facteur d'echelle a en forme fermee :
                    #   nRMSE(a)^2 = (a^2*p_sq - 2*a*pg + g_sq) / g_sq
                    # Sans eux, tester une nouvelle regle de recalibration exigerait
                    # de relire les 540 volumes de prediction.
                    rows.append({"modality": mod, "pair": pair, "target": tgt_f,
                                 "subject": sid, "method": label,
                                 "p_sq": float(pr @ pr), "pg": float(pr @ g), "g_sq": float(g @ g),
                                 "nrmse_raw": round(raw, 5),
                                 "nrmse_subject_rel": round(rel, 5),
                                 "nrmse_scale_oracle": round(sca, 5),
                                 "nrmse_affine_oracle": round(aff, 5),
                                 "scale_a": round(a_s, 4),
                                 "scale_subject_rel": round(r_sub, 4)})
                    del pr
                print(f"  {mod:8s} {pair:14s} {sid} " + "  ".join(
                    f"{r['method']}: {r['nrmse_raw']:.3f}->{r['nrmse_scale_oracle']:.3f}"
                    for r in rows[-3:]), flush=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / "calibration_intensite.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n-> {out}")

    names = [m[0] for m in _methods_for(args)]
    print("\nnRMSE brut -> correction REALISABLE (relative au sujet) -> ORACLE d'echelle\n")
    print(f"  {'contraste':10s} " + "".join(f"{n:>34s}" for n in names))
    for mod in args.modalities:
        line = f"  {mod:10s} "
        for n in names:
            sub = [r for r in rows if r["modality"] == mod and r["method"] == n]
            if not sub:
                line += f"{'-':>34s}"; continue
            r0 = np.mean([r["nrmse_raw"] for r in sub])
            r1 = np.mean([r["nrmse_subject_rel"] for r in sub])
            r2 = np.mean([r["nrmse_scale_oracle"] for r in sub])
            line += f"{r0:>10.4f}{r1:>12.4f}{r2:>12.4f}"
        print(line)

    print("\npart de l'erreur (en energie) supprimee : REALISABLE | ORACLE")
    print(f"  {'contraste':10s} " + "".join(f"{n:>12s}" for n in names))
    for mod in args.modalities:
        line = f"  {mod:10s} "
        for n in names:
            sub = [r for r in rows if r["modality"] == mod and r["method"] == n]
            if not sub:
                line += f"{'-':>12s}"; continue
            e0 = np.mean([r["nrmse_raw"] ** 2 for r in sub])
            e1 = np.mean([r["nrmse_subject_rel"] ** 2 for r in sub])
            e2 = np.mean([r["nrmse_scale_oracle"] ** 2 for r in sub])
            line += f"{100 * (1 - e1 / e0):>7.1f}%|{100 * (1 - e2 / e0):>5.1f}%"
        print(line)

    print("\nventile par champ CIBLE (nRMSE brut -> echelle oracle)")
    for mod in args.modalities:
        print(f"  --- {mod}")
        for tgt in FIELDS:
            line = f"    {tgt:6s}"
            for n in names:
                sub = [r for r in rows if r["modality"] == mod and r["method"] == n and r["target"] == tgt]
                if sub:
                    line += (f"   {n}: {np.mean([r['nrmse_raw'] for r in sub]):.3f}"
                             f"->{np.mean([r['nrmse_subject_rel'] for r in sub]):.3f}"
                             f"->{np.mean([r['nrmse_scale_oracle'] for r in sub]):.3f}"
                             f" (a_or={np.mean([r['scale_a'] for r in sub]):.2f}"
                             f" a_re={np.mean([r['scale_subject_rel'] for r in sub]):.2f})")
            print(line)


# --------------------------------------------------------------------------- #
#  Mode recalib
# --------------------------------------------------------------------------- #

def mode_recalib(args) -> None:
    """La part d'echelle mal reglee est-elle SYSTEMATIQUE, donc corrigeable ?

    --mode calibration montre qu'un scalaire oracle enleve 80 % de l'energie de
    l'erreur, et que l'estimer depuis le volume source echoue. Reste la question
    utile : ce mauvais facteur est-il le MEME pour tous les sujets d'une meme
    cible ? Si oui, il s'estime sur des donnees d'entrainement une fois pour
    toutes — c'est une constante de dénormalisation a corriger, pas un oracle.

    On teste donc des regles de recalibration estimees en LAISSANT LE SUJET DE COTE
    (leave-one-subject-out) : le facteur applique au sujet s est la mediane des
    facteurs oracles des AUTRES sujets. Avec 3 sujets, cela veut dire 2 sujets
    par estimation : bruite, mais honnete — aucune information de la verite du
    sujet evalue n'entre dans son propre facteur.

    Le nRMSE pour un facteur quelconque se recalcule en forme fermee depuis les
    sommes stockees : nRMSE(a)^2 = (a^2*p_sq - 2*a*pg + g_sq) / g_sq.
    """
    src = OUTDIR / "calibration_intensite.csv"
    with open(src) as f:
        rows = [r for r in csv.DictReader(f)]
    if "p_sq" not in rows[0]:
        raise SystemExit(f"{src} ne contient pas p_sq/pg/g_sq — relancer --mode calibration.")
    for r in rows:
        for k in ("p_sq", "pg", "g_sq", "nrmse_raw", "scale_a", "nrmse_scale_oracle"):
            r[k] = float(r[k])

    def nrmse_at(r, a):
        return float(np.sqrt(max(a * a * r["p_sq"] - 2 * a * r["pg"] + r["g_sq"], 0.0) / r["g_sq"]))

    def loo(r, key):
        """Mediane des facteurs oracles des autres sujets, a `key` identique."""
        others = [x["scale_a"] for x in rows
                  if x["method"] == r["method"] and x["modality"] == r["modality"]
                  and key(x) == key(r) and x["subject"] != r["subject"]]
        return float(np.median(others)) if others else 1.0

    regles = [
        ("brut (a = 1)", lambda r: 1.0),
        ("constante par contraste", lambda r: loo(r, lambda x: True)),
        ("constante par champ cible", lambda r: loo(r, lambda x: x["target"])),
        ("constante par paire", lambda r: loo(r, lambda x: x["pair"])),
    ]
    names = [m[0] for m in METHODS]
    print("nRMSE moyen selon la regle de recalibration — facteurs estimes SANS le sujet evalue\n")
    print(f"  {'regle':28s}" + "".join(f"{n:>34s}" for n in names))
    print(f"  {'':28s}" + "".join(f"{'T1W':>11s}{'T2W':>11s}{'T2FLAIR':>12s}" for _ in names))
    for label, rule in regles + [("oracle par volume (borne)", None)]:
        line = f"  {label:28s}"
        for n in names:
            for mod in MODALITIES:
                sub = [r for r in rows if r["method"] == n and r["modality"] == mod]
                if not sub:
                    line += f"{'-':>11s}"; continue
                v = ([r["nrmse_scale_oracle"] for r in sub] if rule is None
                     else [nrmse_at(r, rule(r)) for r in sub])
                line += f"{np.mean(v):>11.4f}"
        print(line)

    print("\nfacteurs oracles medians par champ CIBLE (vectorise) et dispersion inter-sujet")
    print(f"  {'cible':7s}" + "".join(f"{m:>26s}" for m in MODALITIES))
    for tgt in FIELDS:
        line = f"  {tgt:7s}"
        for mod in MODALITIES:
            sub = [r for r in rows if r["method"] == names[0] and r["modality"] == mod and r["target"] == tgt]
            if not sub:
                line += f"{'-':>26s}"; continue
            a = [r["scale_a"] for r in sub]
            line += f"{np.median(a):>14.2f} +-{np.std(a):>9.2f}"
        print(line)


# --------------------------------------------------------------------------- #
#  Mode table
# --------------------------------------------------------------------------- #

def mode_table(args) -> None:
    metrics = {}          # (method, mod) -> {pair: (nrmse, ssim, lpips)}
    for label, per_mod in METHODS:
        for mod, (_, csv_path) in per_mod.items():
            metrics[(label, mod)] = load_metrics(csv_path)
    for mod in MODALITIES:
        p = OUTDIR / f"identity_baseline_{mod}.csv"
        if p.exists():
            with open(p) as f:
                metrics[("Identite", mod)] = {
                    x["pair"]: (float(x["nrmse_mean"]), float(x["ssim_mean"]), float("nan"))
                    for x in csv.DictReader(f)}

    names = [m[0] for m in METHODS]
    have_id = all(("Identite", m) in metrics for m in MODALITIES)
    order = (["Identite"] if have_id else []) + names

    for mi, mname in enumerate(["nRMSE", "SSIM", "LPIPS"]):
        print(f"\n### {mname}\n")
        print("| | " + " | ".join(MODALITIES) + " | moyenne |")
        print("|---|" + "---|" * (len(MODALITIES) + 1))
        for label in order:
            vals = []
            for mod in MODALITIES:
                d = metrics.get((label, mod), {})
                v = [t[mi] for t in d.values()]
                vals.append(float(np.mean(v)) if v and not np.isnan(v[0]) else float("nan"))
            cells = " | ".join("—" if np.isnan(v) else f"{v:.4f}" for v in vals)
            mean = float(np.mean(vals))
            print(f"| {label} | {cells} | " + ("—" if np.isnan(mean) else f"{mean:.4f}") + " |")

    print("\n### Duels paire a paire (nRMSE, 20 paires x 3 contrastes = 60 duels)\n")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            wins = tot = 0
            for mod in MODALITIES:
                da, db = metrics.get((a, mod), {}), metrics.get((b, mod), {})
                for pair in set(da) & set(db):
                    tot += 1
                    wins += da[pair][0] < db[pair][0]
            print(f"  {a} bat {b} : {wins}/{tot} paires")

    if have_id:
        print("\n### Gain relatif sur le temoin identite (1 - nRMSE_modele / nRMSE_identite)\n")
        print("| | " + " | ".join(MODALITIES) + " |")
        print("|---|" + "---|" * len(MODALITIES))
        for label in names:
            cells = []
            for mod in MODALITIES:
                d, di = metrics.get((label, mod), {}), metrics[("Identite", mod)]
                common = sorted(set(d) & set(di))
                g = np.mean([1.0 - d[p][0] / di[p][0] for p in common]) if common else float("nan")
                cells.append(f"{100 * g:.1f} %")
            print(f"| {label} | " + " | ".join(cells) + " |")


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["identity", "scaled_identity", "sharpness",
                                       "calibration", "recalib", "table"], required=True)
    # RÉGION. Défaut `slab` = la tranche axiale [150, 180) que le classement note
    # réellement. `full` ne sert qu'à rejouer un chiffre historique : tous les
    # nombres du CHANGELOG antérieurs au 2026-09-04 sont en `full`, donc NON
    # comparables au classement. Voir common/io.py::Z_CLIP_RANGE.
    ap.add_argument("--region", choices=["slab", "full"], default="slab",
                    help="region evaluee : 'slab' = tranche axiale [150,180) notee par "
                         "le classement (defaut) ; 'full' = volume entier (historique, "
                         "non comparable au classement)")
    ap.add_argument("--modalities", nargs="+", default=MODALITIES)
    ap.add_argument("--pairs", nargs="+", default=PAIRS)
    ap.add_argument("--subjects", nargs="+", default=["0006"])
    ap.add_argument("--n-slices", type=int, default=24)
    ap.add_argument("--interior-frac", type=float, default=1.0,
                    help="fraction centrale de la boite cerveau retenue pour le spectre "
                         "(1.0 = cerveau entier ; 0.5 = interieur seul, sans les bords)")
    ap.add_argument("--check-pairs", nargs="+", default=["3T_to_7T", "0.1T_to_1.5T", "7T_to_0.1T"])
    ap.add_argument("--tol", type=float, default=5e-4)
    ap.add_argument("--pred-root", default=None,
                    help="racine de predictions a scorer (.../predictions/task3) au lieu "
                         "des methodes codees en dur — pour mesurer l'erreur STRUCTURELLE "
                         "d'un nouveau run")
    ap.add_argument("--name", default=None, help="nom de la methode dans le CSV")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR),
                    help="repertoire de sortie ; en passer un NOUVEAU par campagne, "
                         "sous peine d'ecraser les CSV d'une campagne precedente")
    args = ap.parse_args()
    global OUTDIR, REGION
    OUTDIR = Path(args.outdir)
    REGION = args.region
    {"identity": mode_identity, "scaled_identity": mode_scaled_identity,
     "sharpness": mode_sharpness,
     "calibration": mode_calibration, "recalib": mode_recalib,
     "table": mode_table}[args.mode](args)


if __name__ == "__main__":
    main()
