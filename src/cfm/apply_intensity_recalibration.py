#!/usr/bin/env python3
"""Applique un recalage d'intensite a un arbre de predictions DEJA ecrit.

Le recalage est une simple multiplication apres denormalisation : le refaire en
relancant l'inference couterait ~3.5 h de GPU pour un resultat identique au bit
pres. Ce script relit les volumes, multiplie, et reecrit dans un nouvel arbre —
quelques minutes, sans GPU.

`src/cfm/infer_mmfm_unified.py --intensity_recalibration` reste le chemin de
PRODUCTION (le recalage y est applique au moment de la prediction) ; celui-ci
sert a evaluer une table de facteurs sur des predictions existantes.

Usage :
    PYTHONPATH=src python src/cfm/apply_intensity_recalibration.py \\
        --pred-root outputs/mmfm/vec_rbest/predictions/task3 \\
        --out-root outputs/mmfm/vec_rbest_recal/predictions/task3 \\
        --table configs/mmfm/intensity_recalibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def factor(table: dict, mod: str, pair: str) -> float:
    """Table par PAIRE en priorite (0.3231 contre 0.3345 au leave-one-out du
    2026-08-25), repli sur le champ cible, puis 1.0."""
    tgt = pair.split("_to_")[1]
    v = (table.get("by_pair", {}).get(mod, {}).get(pair)
         or table.get("by_target_field", {}).get(mod, {}).get(tgt))
    return float(v) if v else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--level", choices=["pair", "target_field"], default="pair")
    a = ap.parse_args()

    table = json.load(open(a.table))
    if a.level == "target_field":
        table = {"by_target_field": table.get("by_target_field", {})}

    src_root, out_root = Path(a.pred_root), Path(a.out_root)
    n, used = 0, {}
    for mdir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        mod = mdir.name
        for pdir in sorted(p for p in mdir.iterdir() if p.is_dir()):
            f = factor(table, mod, pdir.name)
            used[f"{mod}/{pdir.name}"] = f
            dst = out_root / mod / pdir.name
            dst.mkdir(parents=True, exist_ok=True)
            for p in sorted(pdir.glob("*.nii.gz")):
                img = nib.load(str(p))
                v = np.asarray(img.get_fdata(dtype=np.float32)) * f
                nib.save(nib.Nifti1Image(v.astype(np.float32), img.affine, img.header),
                         str(dst / p.name))
                n += 1
    print(f"{n} volumes reecrits -> {out_root}")
    fs = np.array(list(used.values()))
    print(f"facteurs : min {fs.min():.3f}  median {np.median(fs):.3f}  max {fs.max():.3f}  "
          f"| a 1.000 (aucun facteur trouve) : {(fs == 1.0).sum()}/{len(fs)}")


if __name__ == "__main__":
    main()
