#!/usr/bin/env python3
"""Evaluation QUALITATIVE des trois architectures MMFM, sur les trois contrastes.

Complement visuel aux tableaux de metriques : les CSV de l'evaluateur officiel
disent QUI gagne, pas EN QUOI les predictions different. Ce script produit deux
lectures complementaires, toutes deux calculees sur les volumes de prediction
DEJA ecrits (resolution native 0.5mm, 364x436x364) — aucun GPU, aucune
re-inference, donc rien qui puisse diverger des chiffres publies.

  1. Panorama (--mode panel) : source / verite / 3 predictions, avec la carte
     d'erreur absolue de chacune ET celle du temoin IDENTITE (recopier la source),
     toutes sur la MEME echelle. Le temoin dans la figure evite l'illusion
     classique : une carte d'erreur seule parait toujours mauvaise.
  2. Spectres (--mode spectrum) : puissance radiale moyenne, prediction contre
     verite. C'est la mesure du flou — le defaut visuel dominant de l'INR — sur
     une echelle ou il devient un nombre et non une impression.

Aucune normalisation n'est appliquee avant comparaison : l'evaluateur officiel
n'en applique aucune (voir Evaluation/evaluate.py), les predictions sont donc
deja dans l'echelle d'intensite de la verite terrain.

Usage :
    PYTHONPATH=src python src/cfm/eval_qualitative_3arch.py --mode panel \
        --modality T1W --pair 3T_to_7T --subject 0006
    PYTHONPATH=src python src/cfm/eval_qualitative_3arch.py --mode spectrum \
        --modalities T1W T2W T2FLAIR
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env

# Chaque methode est declaree PAR MODALITE : (racine des predictions, CSV officiel).
# Ce n'est pas de la verbosite gratuite — les predictions T1W du vectorise de
# PRODUCTION vivent dans `vectorized_flip/` (run du 2026-08-13, avec flip), alors
# que celles de `vectorized/predictions/task3/T1W` datent du 2026-08-09 et
# proviennent du checkpoint PRE-correction. Pointer la mauvaise racine
# afficherait un modele qui n'est plus celui dont on publie les chiffres.
RESULTS = Path("results/mmfm")
V_FLIP = "outputs/mmfm/vectorized_flip/predictions/task3"
V_PROD = "outputs/mmfm/vectorized/predictions/task3"
U_PROD = "outputs/mmfm/unet/predictions/task3"
I_PROD = "outputs/mmfm/inr_std/predictions/task3"   # INR CORRIGEE (audit phase B)
I_OLD = "outputs/mmfm/inr/predictions/task3"        # INR d'avant l'audit, gardee comme temoin
C14 = RESULTS / "comparison_20260814_all_contrasts"
C07 = RESULTS / "comparison_20260807_1mm"
AUD = RESULTS / "audit_20260825"

METHODS: List[Tuple[str, Dict[str, Tuple[str, str]]]] = [
    ("Vectorise", {
        "T1W": (V_FLIP, str(RESULTS / "comparison_20260814_vectorized_flip/task3_vectorized_flip_T1W.csv")),
        "T2W": (V_PROD, str(C14 / "task3_vectorized_T2W.csv")),
        "T2FLAIR": (V_PROD, str(C14 / "task3_vectorized_T2FLAIR.csv")),
    }),
    ("UNet", {
        "T1W": (U_PROD, str(C07 / "task3_unet_1mm_T1W.csv")),
        "T2W": (U_PROD, str(C14 / "task3_unet_T2W.csv")),
        "T2FLAIR": (U_PROD, str(C14 / "task3_unet_T2FLAIR.csv")),
    }),
    # INR de reference depuis l'audit du 2026-08-25/26 : checkpoint inr_std
    # (EMA echauffee + entree du flow standardisee) servi par une inference dont
    # l'orientation et la normalisation sont alignees sur le precompute.
    ("INR", {
        "T1W": (I_PROD, str(AUD / "task3_inr_std_T1W.csv")),
        "T2W": (I_PROD, str(AUD / "task3_inr_std_T2W.csv")),
        "T2FLAIR": (I_PROD, str(AUD / "task3_inr_std_T2FLAIR.csv")),
    }),
]

# Temoin AVANT correctifs, pour montrer sur une meme figure ce que les trois bugs
# faisaient (EMA sans correction de biais, orientation LAS/RAS, normalisation).
# `latent_dim = 129024` : ces predictions viennent bien du checkpoint de
# production de l'epoque, pas du run z=4096 (verifie en recalculant, 0.6383).
# Run du 2026-08-27 : time_scale 1000 + conditionnement FiLM + adjacent_only.
# C'est le seul checkpoint dont la trajectoire se courbe reellement
# (cos(v(0),v(1)) = -0.24 / 0.80 / 0.03 contre 1.000000 partout ailleurs ;
# courbure 0.50 / 0.97 / 0.86 de celle exigee contre 0.006-0.05).
S27 = RESULTS / "staircase_20260827"
R_BEST = ("Vectorise R-best", {
    m: ("outputs/mmfm/vec_rbest/predictions/task3", str(S27 / f"task3_vec_rbest_{m}.csv"))
    for m in ("T1W", "T2W", "T2FLAIR")
})

# PLAFOND DE REPRESENTATION : la verite terrain passee dans l'encodeur/decodeur
# de l'architecture, sans aucun transport (paires identite, dt = 0). C'est ce que
# l'architecture rendrait avec un flow PARFAIT. Mesure du 2026-08-27 : MedVAE
# 0.1048 de nRMSE moyen, INR 0.3395 -- et l'INR score 0.3749, donc 90.6 % de son
# erreur est sa representation, pas son flow.
CEIL = RESULTS / "ceiling_20260827"
CEIL_VAE = ("Plafond MedVAE", {
    m: ("outputs/mmfm/ceiling_vectorized/predictions/task3", str(CEIL / f"ceiling_vectorized_{m}.csv"))
    for m in ("T1W", "T2W", "T2FLAIR")
})
CEIL_INR = ("Plafond INR", {
    m: ("outputs/mmfm/ceiling_inr/predictions/task3", str(CEIL / f"ceiling_inr_{m}.csv"))
    for m in ("T1W", "T2W", "T2FLAIR")
})

# Run du 2026-09-02 : R-best entraine sur les latents du MedVAE PERCEPTUEL
# (fine-tuning LPIPS). Plafond de representation 0.0976 contre 0.1048, mais score
# de bout en bout inchange -- le gain de representation est absorbe par le flow.
LPIPS = ("Vectorise LPIPS", {
    m: ("outputs/mmfm/vec_lpips/predictions/task3", str(S27 / f"task3_vec_lpips_{m}.csv"))
    for m in ("T1W", "T2W", "T2FLAIR")
})
CEIL2 = RESULTS / "ceiling_20260901"
CEIL_LPIPS = ("Plafond LPIPS", {
    m: ("outputs/mmfm/ceiling_lpips/predictions/task3", str(CEIL2 / f"ceiling_lpips_{m}.csv"))
    for m in ("T1W", "T2W", "T2FLAIR")
})

INR_AVANT = ("INR avant audit", {
    "T1W": (I_OLD, str(C07 / "task3_inr_129k_T1W.csv")),
    "T2W": (I_OLD, str(C14 / "task3_inr_T2W.csv")),
    "T2FLAIR": (I_OLD, str(C14 / "task3_inr_T2FLAIR.csv")),
})

SUBJECTS = ["0006", "0007", "0009"]
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]


# --------------------------------------------------------------------------- #
#  E/S
# --------------------------------------------------------------------------- #

def load_vol(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))


def gt_path(root: Path, mod: str, field: str, sid: str) -> Path:
    return root / "Training_prospective" / mod / field / f"P_{mod}_{field}_{sid}.nii.gz"


def pred_path(method_root: str, mod: str, pair: str, tgt: str, sid: str) -> Path:
    # Les volumes de PLAFOND sont ranges par paire IDENTITE (`7T_to_7T`) : ils ne
    # dependent que du champ cible, puisqu'aucun transport n'a lieu. On y renvoie
    # donc la paire correspondante, quelle que soit la paire demandee — c'est ce
    # qui permet de poser le plafond a cote d'une prediction sur la meme figure.
    if "/ceiling_" in method_root:
        pair = f"{tgt}_to_{tgt}"
    return Path(method_root) / mod / pair / f"P_{mod}_{tgt}_{sid}.nii.gz"


def load_metrics(csv_path: str) -> Dict[str, Tuple[float, float, float]]:
    p = Path(csv_path)
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            # Deux schemas coexistent. Les CSV Task 3 sont indexes par PAIRE et
            # portent LPIPS ; les CSV de plafond sont indexes par CHAMP cible
            # (aucun transport n'a lieu) et n'ont pas LPIPS. On enregistre le
            # plafond sous la cle `X_to_X` pour qu'il s'aligne sur pred_path.
            lp = float(r["lpips_mean"]) if "lpips_mean" in r else float("nan")
            vals = (float(r["nrmse_mean"]), float(r["ssim_mean"]), lp)
            if "pair" in r:
                out[r["pair"]] = vals
            else:
                out[f"{r['field']}_to_{r['field']}"] = vals
    return out


# --------------------------------------------------------------------------- #
#  Geometrie d'affichage
# --------------------------------------------------------------------------- #

def brain_bbox(vol: np.ndarray, thr_frac: float = 0.08) -> Tuple[slice, slice, slice]:
    """Boite englobante du cerveau, seuil relatif au 99.5e centile (robuste aux
    points chauds isoles, contrairement au max)."""
    thr = thr_frac * float(np.percentile(vol, 99.5))
    mask = vol > thr
    sl = []
    for ax in range(3):
        proj = mask.any(axis=tuple(a for a in range(3) if a != ax))
        idx = np.flatnonzero(proj)
        sl.append(slice(int(idx[0]), int(idx[-1]) + 1) if idx.size else slice(0, vol.shape[ax]))
    return tuple(sl)  # type: ignore[return-value]


def best_axial(vol: np.ndarray, bbox) -> int:
    """Coupe axiale la plus « pleine » : maximise l'aire de cerveau, ce qui
    tombe sur les ventricules/noyaux gris plutot que sur le vertex ou le cou."""
    thr = 0.08 * float(np.percentile(vol, 99.5))
    area = (vol[bbox[0], bbox[1], :] > thr).sum(axis=(0, 1))
    return int(np.argmax(area))


def window(gt: np.ndarray) -> Tuple[float, float]:
    fg = gt[gt > 0.02 * float(np.percentile(gt, 99.9))]
    if fg.size == 0:
        fg = gt.ravel()
    return 0.0, float(np.percentile(fg, 99.0))


# --------------------------------------------------------------------------- #
#  Spectre radial
# --------------------------------------------------------------------------- #

def radial_power(sl: np.ndarray, nbins: int = 96) -> Tuple[np.ndarray, np.ndarray]:
    """Puissance moyenne par anneau de frequence d'une coupe 2D.

    Frequence exprimee en cycles/mm sous-entendus par la normalisation : l'axe
    retourne est en fraction de la frequence de Nyquist (0 = continu, 1 = detail
    a la limite de la grille), ce qui rend la courbe independante de la taille
    de la coupe.
    """
    h, w = sl.shape
    f = np.fft.fftshift(np.fft.fft2(sl - sl.mean()))
    p = (f.real ** 2 + f.imag ** 2)
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    # rayon normalise par le Nyquist de chaque axe : sans cela une coupe non
    # carree melangerait des frequences differentes dans le meme anneau.
    r = np.sqrt(((yy - cy) / (h / 2.0)) ** 2 + ((xx - cx) / (w / 2.0)) ** 2)
    bins = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < nbins)
    sums = np.bincount(idx[ok], weights=p.ravel()[ok], minlength=nbins)
    cnts = np.bincount(idx[ok], minlength=nbins).astype(np.float64)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, sums / np.maximum(cnts, 1.0)


def spectrum_of_volume(vol: np.ndarray, bbox, n_slices: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    z0, z1 = bbox[2].start, bbox[2].stop
    zs = np.linspace(z0 + 0.15 * (z1 - z0), z0 + 0.85 * (z1 - z0), n_slices).astype(int)
    acc = None
    for z in zs:
        c, p = radial_power(vol[bbox[0], bbox[1], z])
        acc = p if acc is None else acc + p
    return c, acc / len(zs)


def hf_ratio(p_pred: np.ndarray, p_gt: np.ndarray, centers: np.ndarray,
             lo: float = 0.25, hi: float = 0.75,
             lf_lo: float = 0.02, lf_hi: float = 0.15) -> float:
    """Indice de nettete relative : (HF/BF)_prediction / (HF/BF)_verite.

    1.0 = meme repartition de l'energie entre details et structures larges que la
    verite ; < 1 = flou ; > 1 = trop de details (soit du bruit, soit — cas
    frequent ici — une image qui a garde la nettete de sa SOURCE au lieu de
    prendre celle de sa cible).

    Le rapport HF/BF est pris AVANT la comparaison, et c'est essentiel : une
    puissance haute frequence brute est proportionnelle au carre de l'echelle
    d'intensite de l'image. Or les predictions de ce projet se trompent
    justement d'echelle d'intensite (voir eval_quantitative_3arch.py
    --mode calibration) — un ratio de puissances brutes mesurerait donc surtout
    cette erreur-la, pas la nettete. Normaliser chaque spectre par sa propre
    bande basse frequence rend l'indice invariant a tout facteur multiplicatif.

    La bande 0.25-0.75 Nyquist exclut le continu (que tout le monde reussit) et
    l'extreme (domine par le bruit d'acquisition, que personne ne doit
    reproduire) ; la bande 0.02-0.15 sert de reference basse frequence.
    """
    hf = (centers >= lo) & (centers <= hi)
    bf = (centers >= lf_lo) & (centers <= lf_hi)
    num = float(p_pred[hf].sum()) / max(float(p_pred[bf].sum()), 1e-30)
    den = float(p_gt[hf].sum()) / max(float(p_gt[bf].sum()), 1e-30)
    return num / den if den > 0 else float("nan")


# --------------------------------------------------------------------------- #
#  Mode « panel »
# --------------------------------------------------------------------------- #

def make_panel(args, data_root: Path, out: Path) -> None:
    src_f, tgt_f = args.pair.split("_to_")
    mod, sid = args.modality, args.subject

    gt = load_vol(gt_path(data_root, mod, tgt_f, sid))
    src = load_vol(gt_path(data_root, mod, src_f, sid))
    preds = []
    methods = _selected_methods(args)
    for label, per_mod in methods:
        if mod not in per_mod:
            raise SystemExit(f"{label} : modalite {mod} non declaree")
        root, csv_path = per_mod[mod]
        p = pred_path(root, mod, args.pair, tgt_f, sid)
        if not p.exists():
            raise SystemExit(f"prediction manquante : {p}")
        preds.append((label, load_vol(p), load_metrics(csv_path).get(args.pair)))

    bbox = brain_bbox(gt)
    z = args.slice if args.slice is not None else best_axial(gt, bbox)
    lo, hi = window(gt)
    # la source est a un AUTRE champ : son echelle d'intensite n'est pas celle de
    # la cible, on la fenetre donc sur ses propres centiles (sinon elle parait
    # noire ou saturee pour une raison qui n'a rien a voir avec le modele).
    slo, shi = window(src)

    def ax_slice(v):
        return np.rot90(v[bbox[0], bbox[1], z])

    zoom = args.zoom
    if zoom is None:
        h, w = ax_slice(gt).shape
        # region corticale : quart superieur-gauche, la ou le ruban cortical est
        # tangent au plan de coupe et donc le plus exigeant en finesse.
        zoom = (int(0.12 * h), int(0.42 * h), int(0.15 * w), int(0.50 * w))

    ncol = 2 + len(preds)
    fig, axes = plt.subplots(3, ncol, figsize=(3.1 * ncol, 9.6))
    err_hi = args.err_scale * hi

    cols = [("Source " + src_f, ax_slice(src), slo, shi, "gray"),
            (f"Verite {tgt_f}", ax_slice(gt), lo, hi, "gray")]
    for label, v, m in preds:
        # m vient du CSV officiel : c'est la moyenne de la PAIRE sur les 3 sujets,
        # pas la valeur de ce sujet-ci. On ne recalcule pas de metrique maison ici
        # (cf. en-tete), donc le titre le dit explicitement.
        t = label if m is None else f"{label}\nnRMSE {m[0]:.3f}  SSIM {m[1]:.3f}  (moy. 3 suj.)"
        cols.append((t, ax_slice(v), lo, hi, "gray"))

    for c, (title, img, a, b, cm) in enumerate(cols):
        axes[0, c].imshow(img, cmap=cm, vmin=a, vmax=b)
        axes[0, c].set_title(title, fontsize=9)
        y0, y1, x0, x1 = zoom
        axes[0, c].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           fill=False, ec="#ff4d4d", lw=1.0))
        axes[2, c].imshow(img[y0:y1, x0:x1], cmap=cm, vmin=a, vmax=b,
                          interpolation="nearest")
        axes[2, c].set_title("zoom cortex", fontsize=8)

    gt_s = ax_slice(gt)
    # colonne 0 de la rangee d'erreur = temoin IDENTITE (source recopiee), sur la
    # meme echelle que les autres : c'est la reference « ne rien faire ».
    errs = [("|source - verite|  (temoin identite)", np.abs(ax_slice(src) - gt_s))]
    errs += [(f"|{label} - verite|", np.abs(ax_slice(v) - gt_s)) for label, v, _ in preds]

    axes[1, 0].imshow(errs[0][1], cmap="inferno", vmin=0, vmax=err_hi)
    axes[1, 0].set_title(errs[0][0], fontsize=8)
    axes[1, 1].axis("off")
    axes[1, 1].text(0.5, 0.5, f"echelle d'erreur\n0 - {err_hi:.3g}\n(commune aux 4 cartes)",
                    ha="center", va="center", fontsize=8, transform=axes[1, 1].transAxes)
    for k, (title, e) in enumerate(errs[1:]):
        im = axes[1, 2 + k].imshow(e, cmap="inferno", vmin=0, vmax=err_hi)
        axes[1, 2 + k].set_title(title, fontsize=8)
    fig.colorbar(im, ax=axes[1, 2:].tolist(), fraction=0.02, pad=0.01)

    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{mod} — {args.pair} — sujet {sid}   (coupe axiale {z}, 0.5 mm natif)",
                 fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"ecrit : {out}")


# --------------------------------------------------------------------------- #
#  Mode « calib » — la meme prediction, avant et apres correction d'echelle
# --------------------------------------------------------------------------- #

def make_calib_demo(args, data_root: Path, out: Path) -> None:
    """Montre ce que « 80 % de l'erreur est une erreur d'echelle » veut dire.

    Colonne 2 = prediction telle qu'elle sort du modele ; colonne 3 = la MEME
    prediction multipliee par le scalaire qui minimise l'erreur (un oracle : il
    connait la verite). Si la colonne 3 ressemble a la verite, alors le modele
    avait raison sur l'anatomie et tort sur un seul nombre.
    """
    src_f, tgt_f = args.pair.split("_to_")
    mod, sid = args.modality, args.subject
    gt = load_vol(gt_path(data_root, mod, tgt_f, sid))
    bbox = brain_bbox(gt)
    z = args.slice if args.slice is not None else best_axial(gt, bbox)
    lo, hi = window(gt)
    g = gt.ravel().astype(np.float64)
    ng = float(np.linalg.norm(g))

    rows = []
    for label, per_mod in _selected_methods(args):
        if mod not in per_mod:
            continue
        p = pred_path(per_mod[mod][0], mod, args.pair, tgt_f, sid)
        if not p.exists():
            continue
        v = load_vol(p)
        f = v.ravel().astype(np.float64)
        a = float(f @ g) / max(float(f @ f), 1e-12)
        rows.append((label, v, a,
                     float(np.linalg.norm(f - g)) / ng,
                     float(np.linalg.norm(a * f - g)) / ng))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9.6, 3.2 * len(rows)), squeeze=False)
    for r, (label, v, a, e0, e1) in enumerate(rows):
        sl = np.rot90(v[bbox[0], bbox[1], z])
        gsl = np.rot90(gt[bbox[0], bbox[1], z])
        for c, (img, title) in enumerate([
                (gsl, f"verite {tgt_f}"),
                (sl, f"{label} brut\nnRMSE {e0:.3f}"),
                (a * sl, f"{label} x {a:.2f} (echelle oracle)\nnRMSE {e1:.3f}")]):
            axes[r][c].imshow(img, cmap="gray", vmin=lo, vmax=hi)
            axes[r][c].set_title(title, fontsize=9)
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    fig.suptitle(f"{mod} — {args.pair} — sujet {sid} : un seul scalaire separe la colonne 2 de la colonne 3",
                 fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"ecrit : {out}")


# --------------------------------------------------------------------------- #
#  Mode « spectrum »
# --------------------------------------------------------------------------- #

def make_spectrum(args, data_root: Path, out: Path) -> None:
    mods = args.modalities
    pairs = args.spectrum_pairs
    fig, axes = plt.subplots(1, len(mods), figsize=(5.2 * len(mods), 4.4), squeeze=False)
    rows = []

    for ci, mod in enumerate(mods):
        ax = axes[0][ci]
        for pi, pair in enumerate(pairs):
            tgt = pair.split("_to_")[1]
            gt = load_vol(gt_path(data_root, mod, tgt, args.subject))
            bbox = brain_bbox(gt)
            centers, p_gt = spectrum_of_volume(gt, bbox, args.n_slices)
            style = "-" if pi == 0 else "--"
            if pi == 0:
                ax.plot(centers, p_gt, color="k", lw=2.0, ls=style, label="verite terrain")
            else:
                ax.plot(centers, p_gt, color="k", lw=2.0, ls=style)
            for mi, (label, per_mod) in enumerate(_selected_methods(args)):
                if mod not in per_mod:
                    continue
                p = pred_path(per_mod[mod][0], mod, pair, tgt, args.subject)
                if not p.exists():
                    continue
                _, p_pr = spectrum_of_volume(load_vol(p), bbox, args.n_slices)
                col = plt.cm.tab10(mi)
                ax.plot(centers, p_pr, color=col, lw=1.4, ls=style,
                        label=label if pi == 0 else None)
                rows.append({"modality": mod, "pair": pair, "method": label,
                             "hf_ratio": round(hf_ratio(p_pr, p_gt, centers), 4)})
        ax.set_yscale("log")
        ax.set_xlabel("frequence radiale (fraction du Nyquist)")
        if ci == 0:
            ax.set_ylabel("puissance moyenne (log)")
        ax.set_title(f"{mod}   —   {pairs[0]} (plein)"
                     + ("" if len(pairs) < 2 else f" / {pairs[1]} (tirets)"), fontsize=10)
        ax.axvspan(0.25, 0.75, color="0.85", zorder=0)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.suptitle("Puissance spectrale radiale — le flou, en nombre\n"
                 "bande grisee : 0.25-0.75 Nyquist, celle de l'indice de nettete",
                 fontsize=11, y=0.99)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"ecrit : {out}")

    csv_out = out.with_suffix(".csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["modality", "pair", "method", "hf_ratio"])
        w.writeheader(); w.writerows(rows)
    print(f"ecrit : {csv_out}")
    print("\nratio d'energie haute frequence conservee (1.0 = finesse de la verite)")
    for mod in mods:
        for pair in pairs:
            sub = [r for r in rows if r["modality"] == mod and r["pair"] == pair]
            if sub:
                print(f"  {mod:8s} {pair:14s} " +
                      "  ".join(f"{r['method']}={r['hf_ratio']:.3f}" for r in sub))


# --------------------------------------------------------------------------- #

def _selected_methods(args) -> List[Tuple[str, Dict[str, Tuple[str, str]]]]:
    """Les trois architectures de production, plus les temoins demandes.

    L'ordre place les plafonds en DERNIER : ils ne sont pas des methodes
    concurrentes mais la borne que chaque architecture ne peut pas franchir.
    """
    m = list(METHODS)
    if getattr(args, "with_rbest", False):
        m.append(R_BEST)
    if getattr(args, "with_lpips", False):
        m.append(LPIPS)
    if getattr(args, "with_before", False):
        m.append(INR_AVANT)
    if getattr(args, "with_ceiling", False):
        m += [CEIL_VAE, CEIL_INR]
    if getattr(args, "with_ceiling_lpips", False):
        m.append(CEIL_LPIPS)
    only = getattr(args, "only", None)
    if only:
        # selection par nom, dans l'ordre demande : au-dela de 6 colonnes un
        # panneau devient illisible, et toutes les comparaisons n'ont pas besoin
        # des memes temoins.
        by = {lbl: pair for lbl, pair in m}
        missing = [o for o in only if o not in by]
        if missing:
            raise SystemExit(f"--only : inconnu(s) {missing} ; disponibles {sorted(by)}")
        m = [(o, by[o]) for o in only]
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["panel", "spectrum", "calib"], default="panel")
    ap.add_argument("--with-rbest", action="store_true",
                    help="ajoute le run corrige du 2026-08-27 (time_scale + FiLM + adjacent_only)")
    ap.add_argument("--with-lpips", action="store_true",
                    help="ajoute le run sur latents MedVAE perceptuel (2026-09-02)")
    ap.add_argument("--with-ceiling-lpips", action="store_true",
                    help="ajoute le plafond du MedVAE perceptuel")
    ap.add_argument("--only", nargs="+", default=None,
                    help="ne garder que ces methodes, dans cet ordre (par leur nom)")
    ap.add_argument("--with-ceiling", action="store_true",
                    help="ajoute le PLAFOND de representation : la verite terrain passee dans "
                         "l'encodeur/decodeur, sans transport. Montre ce que l'architecture "
                         "rendrait avec un flow parfait.")
    ap.add_argument("--modality", default="T1W")
    ap.add_argument("--modalities", nargs="+", default=["T1W", "T2W", "T2FLAIR"])
    ap.add_argument("--pair", default="3T_to_7T")
    ap.add_argument("--spectrum-pairs", nargs="+", default=["3T_to_7T", "3T_to_0.1T"])
    ap.add_argument("--subject", default="0006")
    ap.add_argument("--slice", type=int, default=None)
    ap.add_argument("--zoom", type=int, nargs=4, default=None, metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--with-before", action="store_true",
                    help="ajoute une colonne avec l'INR d'AVANT l'audit (temoin visuel des bugs)")
    ap.add_argument("--err-scale", type=float, default=0.6,
                    help="haut de l'echelle des cartes d'erreur, en fraction du 99e centile de la verite")
    ap.add_argument("--n-slices", type=int, default=24)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--outdir", default="results/mmfm/qualitative_20260824")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    data_root = Path(load_env("local")["data_root"])
    outdir = Path(args.outdir)
    if args.mode == "panel":
        out = Path(args.output) if args.output else outdir / f"panel_{args.modality}_{args.pair}_{args.subject}.png"
        make_panel(args, data_root, out)
    elif args.mode == "calib":
        out = Path(args.output) if args.output else outdir / f"calib_{args.modality}_{args.pair}_{args.subject}.png"
        make_calib_demo(args, data_root, out)
    else:
        out = Path(args.output) if args.output else outdir / "spectres_radiaux.png"
        make_spectrum(args, data_root, out)


if __name__ == "__main__":
    main()
