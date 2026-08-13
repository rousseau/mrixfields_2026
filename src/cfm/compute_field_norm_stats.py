#!/usr/bin/env python3
"""Statistiques de normalisation FIXES par (modalité, champ) — remplace la
normalisation percentile PAR VOLUME qui efface les différences de dynamique
réelles entre champs (voir diagnostic : à 5T/7T, le GT natif n'occupe qu'une
fraction de [0,1] alors que la normalisation par volume étire chaque champ
pour remplir tout [0,1], effaçant le signal que le modèle devrait apprendre).

Pour chaque (modalité, champ), calcule la MÉDIANE (sur tous les sujets de
retro_train) des percentiles bas/haut par-volume — un résumé robuste de la
dynamique typique de ce champ, au lieu d'une normalisation individuelle par
volume qui gomme les écarts inter-champs.

Usage :
    PYTHONPATH=src python src/cfm/compute_field_norm_stats.py \
        --config configs/mmfm/vectorized.yaml --env local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from common.io import FILE_RE, MODALITIES, DOMAINS, SPLIT_MAP, resample_volume


def main():
    ap = argparse.ArgumentParser(description="Statistiques de normalisation fixes par champ")
    ap.add_argument("--config", required=True)
    ap.add_argument("--env", default="local")
    ap.add_argument("--split", default=None, help="Défaut : data.split de la config")
    ap.add_argument("--out", default="configs/field_norm_stats.json")
    args = ap.parse_args()

    cfg = load_yaml_with_include(args.config)
    cfg = resolve_paths(cfg, load_env(args.env))
    data_cfg = cfg["data"]

    modalities = data_cfg.get("modalities", MODALITIES)
    fields = data_cfg.get("fields", DOMAINS)
    split = args.split or data_cfg.get("split", "retro_train")
    p_lo = data_cfg.get("percentile_lower", 0.5)
    p_hi = data_cfg.get("percentile_upper", 99.5)
    raw_ts = data_cfg.get("target_spacing", None)
    target_spacing = tuple(float(v) for v in raw_ts) if raw_ts else None

    data_root = Path(data_cfg["data_root"])
    split_dir = SPLIT_MAP.get(split, split)

    stats = {}
    for modality in modalities:
        stats[modality] = {}
        for field in fields:
            src_dir = data_root / split_dir / modality / field
            if not src_dir.exists():
                print(f"[WARN] absent : {src_dir}")
                continue
            files = sorted(src_dir.glob("*.nii.gz"))
            los, his = [], []
            for p in files:
                if FILE_RE.match(p.name) is None:
                    continue
                img = nib.load(str(p))
                vol = img.get_fdata(dtype=np.float32)
                if target_spacing is not None:
                    spacing = np.abs(np.diag(img.affine)[:3])
                    vol = resample_volume(vol, spacing, target_spacing)
                los.append(float(np.percentile(vol, p_lo)))
                his.append(float(np.percentile(vol, p_hi)))
            if not los:
                print(f"[WARN] aucun volume valide : {src_dir}")
                continue
            lo_med, hi_med = float(np.median(los)), float(np.median(his))
            stats[modality][field] = {"lo": lo_med, "hi": hi_med, "n": len(los)}
            print(f"{modality}/{field}: n={len(los):4d}  lo(médiane)={lo_med:.4f}  "
                  f"hi(médiane)={hi_med:.4f}  [par-volume: lo∈[{min(los):.3f},{max(los):.3f}] "
                  f"hi∈[{min(his):.3f},{max(his):.3f}]]")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "split": split,
        "percentile_lower": p_lo,
        "percentile_upper": p_hi,
        "target_spacing": list(target_spacing) if target_spacing else None,
        "stats": stats,
    }
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n✅ Statistiques sauvegardées : {out_path}")


if __name__ == "__main__":
    main()
