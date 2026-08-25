#!/usr/bin/env python3
"""Phase A de l'audit — de combien le latent vu a l'INFERENCE s'ecarte-t-il de
celui sur lequel le flow a ete ENTRAINE ?

Le flow INR est entraine sur un cache de latents (`precompute_inr_latents.py`) et
ne rappelle jamais `prep_latent` (`cache_prebakes_prep=True`), tandis que
l'inference le rappelle en direct. Rien, cote entrainement, ne peut donc detecter
une derive entre les deux chemins. Ce script la mesure.

Pour chaque volume du cache on refait le fit sous plusieurs variantes de
pretraitement, et on compare au `z` du cache (L2 relatif et cosinus).

Variantes :
  inference   chemin d'inference actuel : resample_to_output (RAS) + field_fixed + bf16
  orient      idem mais rechantillonnage du PRECOMPUTE (garde l'orientation native LAS)
  orient_norm idem + normalisation par volume (celle du cache et du backbone)
  precompute  idem + fp32 : reproduction fidele du precompute

`precompute` donne le PLANCHER : il reste un ecart parce que
`inr_backbone.sample_points` tire ses points avec `generator=None`, donc le fit
n'est pas deterministe. Tout ce qui depasse ce plancher est une derive reelle.

Usage :
    PYTHONPATH=src python src/cfm/audit_inr_latent.py --n 6
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
import torch
from nibabel import processing as nib_proc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env, load_yaml_with_include, resolve_paths
from common.io import center_crop_or_pad_np, load_nifti_volume
from cfm.arch_inr import load_inr_backbone, _backbone_config_from_cfg
from cfm.inr_backbone import fit_new_volume, make_coord_grid

OUT = Path("results/mmfm/audit_20260825")


def _norm_fixed(vol: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32) * 2.0 - 1.0


def path_precompute(p: Path, spacing, volume_size, lo, hi, per_volume: bool) -> np.ndarray:
    """Chemin du PRECOMPUTE : resample_volume (scipy.zoom) sur le tableau brut,
    donc l'orientation NATIVE du fichier est conservee."""
    if per_volume:
        vol, _ = load_nifti_volume(p, target_spacing=spacing, volume_size=None,
                                   normalize=True, lo_pct=0.5, hi_pct=99.5)
    else:
        raw, _ = load_nifti_volume(p, target_spacing=spacing, volume_size=None, normalize=False)
        vol = _norm_fixed(raw, lo, hi)
    return center_crop_or_pad_np(vol, volume_size)


def path_inference(p: Path, spacing, volume_size, lo, hi, per_volume: bool) -> np.ndarray:
    """Chemin de l'INFERENCE : nibabel.resample_to_output, qui reoriente en
    canonique RAS — l'axe 0 est inverse par rapport au precompute."""
    img = nib.load(str(p))
    res = nib_proc.resample_to_output(img, voxel_sizes=spacing, order=1)
    raw = res.get_fdata(dtype=np.float32)
    if per_volume:
        a, b = np.percentile(raw, 0.5), np.percentile(raw, 99.5)
        vol = np.clip((raw - a) / max(b - a, 1e-8), 0.0, 1.0).astype(np.float32) * 2.0 - 1.0
    else:
        vol = _norm_fixed(raw, lo, hi)
    return center_crop_or_pad_np(vol, volume_size)


VARIANTS: List[Tuple[str, bool, bool, bool]] = [
    # (nom, chemin_precompute, normalisation_par_volume, fp32)
    ("inference",   False, False, False),
    ("orient",      True,  False, False),
    ("orient_norm", True,  True,  False),
    ("precompute",  True,  True,  True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mmfm/inr.yaml")
    ap.add_argument("--n", type=int, default=6, help="volumes tires du cache")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = load_env("local")
    cfg = resolve_paths(load_yaml_with_include(args.config), env)
    data_cfg = cfg["data"]
    data_root = Path(env["data_root"])
    spacing = tuple(float(s) for s in data_cfg["target_spacing"])
    volume_size = tuple(int(v) for v in data_cfg["volume_size"])
    split = data_cfg.get("split", "retro_train")
    split_dir = {"retro_train": "Training_retrospective"}.get(split, split)

    with open(Path(data_cfg["latent_cache_dir"]) / "index.json") as f:
        index = json.load(f)
    cache_root = Path(data_cfg["latent_cache_root"])
    stats = json.load(open("configs/mmfm/field_norm_stats.json"))["stats"]

    rng = np.random.default_rng(args.seed)
    samples = [index["samples"][i] for i in rng.choice(len(index["samples"]), args.n, replace=False)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = load_inr_backbone(cfg, device)
    bcfg = _backbone_config_from_cfg(cfg)
    coords = make_coord_grid(volume_size, device=device)
    fit_points = int(cfg["inr_backbone"]["fit_points"])

    print(f"{args.n} volumes  |  fit : {bcfg.inner_steps_eval} pas, lr={bcfg.inner_lr}, "
          f"{fit_points} points\n")
    header = f"  {'volume':26s}" + "".join(f"{v[0]:>14s}" for v in VARIANTS)
    print(header); print("  " + "-" * (len(header) - 2))

    rows = []
    for s in samples:
        z_ref = torch.load(cache_root / s["path"], map_location=device).float()
        stem = Path(s["path"]).stem
        nii = data_root / split_dir / s["modality"] / s["field"] / f"{stem}.nii.gz"
        if not nii.exists():
            print(f"  {stem:26s}  MANQUANT : {nii}")
            continue
        entry = stats[s["modality"]][s["field"]]
        line = f"  {stem:26s}"
        for name, use_pre, per_vol, fp32 in VARIANTS:
            build = path_precompute if use_pre else path_inference
            vol = build(nii, spacing, volume_size, entry["lo"], entry["hi"], per_vol)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                   if (not fp32 and device.type == "cuda") else torch.enable_grad())
            with ctx:
                z = fit_new_volume(backbone, x, coords, num_steps=bcfg.inner_steps_eval,
                                   lr=bcfg.inner_lr, num_points=fit_points)
            z = z.reshape(-1).float()
            rel = float((z - z_ref).norm() / z_ref.norm())
            cos = float(torch.nn.functional.cosine_similarity(z[None], z_ref[None])[0])
            rows.append({"volume": stem, "modality": s["modality"], "field": s["field"],
                         "variant": name, "rel_l2": round(rel, 4), "cos": round(cos, 5)})
            line += f"{rel:>14.3f}"
            del x, z
            torch.cuda.empty_cache()
        print(line, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "latent_drift.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n  MOYENNES")
    print(f"  {'variante':14s}{'L2 relatif':>13s}{'cosinus':>11s}   lecture")
    notes = {
        "inference": "ce que le flow recoit AUJOURD'HUI",
        "orient": "orientation alignee",
        "orient_norm": "+ normalisation alignee",
        "precompute": "plancher (tirage de points non graine)",
    }
    for name, *_ in VARIANTS:
        sub = [r for r in rows if r["variant"] == name]
        print(f"  {name:14s}{np.mean([r['rel_l2'] for r in sub]):>13.3f}"
              f"{np.mean([r['cos'] for r in sub]):>11.4f}   {notes[name]}")
    print(f"\n  CSV : {out}")


if __name__ == "__main__":
    main()
