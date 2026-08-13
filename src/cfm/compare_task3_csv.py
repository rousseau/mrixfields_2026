#!/usr/bin/env python3
"""Compare deux CSV Task 3 produits par `src/evaluation/evaluate.py`.

Ce projet compare des variantes d'architecture en permanence, et la moyenne
globale seule a déjà induit en erreur : l'écart du UNet face au vectorisé est
CONCENTRÉ sur les transitions ->5T/->7T (0.821 contre 0.676) alors que les
moyennes ne diffèrent que de 0.027. Ce script ventile donc systématiquement par
champ CIBLE, en plus du total et du décompte de paires gagnées.

Usage:
    PYTHONPATH=src python src/cfm/compare_task3_csv.py \\
        --ref results/mmfm/comparison_20260807_1mm/task3_unet_1mm_T1W.csv \\
        --new results/mmfm/.../task3_unet_adagn_T1W.csv \\
        --ref-name "UNet (décalage additif)" --new-name "UNet AdaGN"
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

METRICS = ("nrmse", "ssim", "lpips")
LOWER_IS_BETTER = {"nrmse": True, "ssim": False, "lpips": True}


def load(path: Path) -> Dict[str, Dict[str, float]]:
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} est vide")
    return {r["pair"]: {m: float(r[f"{m}_mean"]) for m in METRICS} for r in rows}


def target_field(pair: str) -> str:
    m = re.match(r"^(.+?)_to_(.+)$", pair)
    return m.group(2) if m else "?"


def _fmt_delta(d: float, metric: str) -> str:
    better = (d < 0) if LOWER_IS_BETTER[metric] else (d > 0)
    return f"{d:+.4f} {'✓' if better else '✗'}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparaison de deux CSV Task 3")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--ref-name", default="référence")
    ap.add_argument("--new-name", default="nouveau")
    args = ap.parse_args()

    ref, new = load(Path(args.ref)), load(Path(args.new))
    pairs = sorted(set(ref) & set(new))
    missing = (set(ref) ^ set(new))
    if missing:
        print(f"⚠ {len(missing)} paires non communes, ignorées : {sorted(missing)[:5]}")
    if not pairs:
        raise SystemExit("aucune paire commune — les deux CSV ne décrivent pas le même protocole")

    print(f"{len(pairs)} paires communes\n")
    print(f"{'':22s} {args.ref_name:>22s} {args.new_name:>22s} {'écart':>14s}")
    for m in METRICS:
        a = float(np.mean([ref[p][m] for p in pairs]))
        b = float(np.mean([new[p][m] for p in pairs]))
        print(f"  {m.upper():20s} {a:22.4f} {b:22.4f} {_fmt_delta(b - a, m):>14s}")

    wins = sum(1 for p in pairs if new[p]["nrmse"] < ref[p]["nrmse"])
    print(f"\n  paires gagnées en nRMSE : {wins}/{len(pairs)}")

    # Ventilation par champ CIBLE : c'est là que se cachent les écarts réels.
    by_field: Dict[str, List[str]] = defaultdict(list)
    for p in pairs:
        by_field[target_field(p)].append(p)

    def field_key(f: str) -> float:
        try:
            return float(f.replace("T", ""))
        except ValueError:
            return 1e9

    print(f"\n--- nRMSE par champ CIBLE ---")
    print(f"{'cible':>8s} {'n':>3s} {args.ref_name:>22s} {args.new_name:>22s} {'écart':>14s}")
    for f in sorted(by_field, key=field_key):
        ps = by_field[f]
        a = float(np.mean([ref[p]["nrmse"] for p in ps]))
        b = float(np.mean([new[p]["nrmse"] for p in ps]))
        print(f"{f:>8s} {len(ps):3d} {a:22.4f} {b:22.4f} {_fmt_delta(b - a, 'nrmse'):>14s}")

    print(f"\n--- détail par paire (nRMSE) ---")
    for p in sorted(pairs, key=lambda x: new[x]["nrmse"] - ref[x]["nrmse"]):
        d = new[p]["nrmse"] - ref[p]["nrmse"]
        print(f"  {p:22s} {ref[p]['nrmse']:8.4f} -> {new[p]['nrmse']:8.4f}   {_fmt_delta(d, 'nrmse')}")


if __name__ == "__main__":
    main()
