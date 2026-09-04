#!/usr/bin/env python3
"""La porte [3/5] et le contrôle négatif [5/5] du smoke test, appliqués au
backbone INR de PRODUCTION (configs/mmfm/inr.yaml).

Le smoke test valide le mécanisme sur un backbone jetable (4096-d, 2000 pas,
8 volumes). Il ne dit rien du backbone réellement consommé par le flow
(`outputs/mmfm/inr_backbone/weights/model_final.pth`, 129024-d, 50k pas).
Ce script pose exactement les mêmes questions à celui-là :

  [3/5] le fit de z bat-il la moyenne LEAVE-ONE-OUT des autres volumes,
        c'est-à-dire la meilleure prédiction qui n'a PAS accès à z ?
  [5/5] un z=0 (aucune information par volume) est-il nettement PIRE que le
        fit ? Sans ça, rien n'établit que [3/5] sache séparer sain et cassé.

PARITÉ DE CALCUL : les métriques, le sous-échantillonnage et le chargement des
volumes sont IMPORTÉS de test_inr_backbone_smoke — pas réimplémentés. Les deux
backbones sont jugés par le même code, aux mêmes points, sur les mêmes volumes.
Seul le backbone change (et, par lui, sa config de fitting).

Usage:
    PYTHONPATH=src python src/cfm/gate_inr_production.py
    PYTHONPATH=src python src/cfm/gate_inr_production.py --num-steps 50
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_yaml_with_include, load_env, resolve_paths
from cfm.arch_inr import load_inr_backbone
from cfm.inr_backbone import fit_new_volume, make_coord_grid
# Code PARTAGÉ avec le smoke test — ne pas recopier ces fonctions ici.
from cfm.test_inr_backbone_smoke import (
    VOLUME_SIZE, _fg_nrmse, _load_smoke_volumes, _resample_idx,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/mmfm/inr.yaml")
    ap.add_argument("--num-steps", type=int, default=None,
                    help="Pas de fit de z. Défaut = inner_steps_eval de la config "
                         "(le réglage de PRODUCTION). 50 = le réglage du smoke test.")
    ap.add_argument("--out", default="results/mmfm/gate_inr_production_20260904")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = resolve_paths(load_yaml_with_include(args.config), load_env("local"))

    print("=" * 70)
    print(f"PORTE [3/5]+[5/5] — backbone de PRODUCTION | device={device}")
    print("=" * 70)
    backbone = load_inr_backbone(cfg, device)   # exerce aussi le garde-fou D0
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    b = cfg.get("inr_backbone", {})
    n_steps = args.num_steps if args.num_steps is not None else int(b.get("inner_steps_eval", 20))
    lr = float(b.get("inner_lr", 1e-2))
    n_points = int(b.get("fit_points", 1_048_576))
    print(f"  latent_dim={backbone.cfg.latent_dim}  fit: {n_steps} pas, lr={lr}, "
          f"{n_points} points  (checkpoint: {b.get('checkpoint')})")

    volumes = _load_smoke_volumes(device)       # mêmes volumes que le smoke test
    n_vol = volumes.shape[0]
    print(f"Volumes chargés : {tuple(volumes.shape)}")

    coords_full = make_coord_grid(VOLUME_SIZE, device=device)
    n_eval = min(1_000_000, coords_full.shape[0])
    idx = _resample_idx(coords_full, n_eval)    # mêmes points query que le smoke test
    coords_sub = coords_full[idx]

    z_zero = torch.zeros(1, backbone.cfg.latent_dim, device=device, dtype=torch.float32)
    with torch.no_grad():
        pred_z0 = backbone.decode(coords_sub.unsqueeze(0), z_zero).reshape(-1)

    rows = []
    for i in range(n_vol):
        vol = volumes[i:i + 1]
        gt = vol.reshape(1, -1)[0, idx]

        z = fit_new_volume(backbone, vol, coords_full, num_steps=n_steps, lr=lr,
                           num_points=min(n_points, coords_full.shape[0]))
        with torch.no_grad():
            pred_fit = backbone.decode(coords_sub.unsqueeze(0), z).reshape(-1)

        # baseline LEAVE-ONE-OUT : la meilleure prédiction sans accès à z.
        mean_loo = torch.cat([volumes[:i], volumes[i + 1:]]).mean(dim=0, keepdim=True)
        mean_sub = mean_loo.reshape(1, -1)[0, idx]

        r = {
            "volume": i,
            "nrmse_fit": _fg_nrmse(pred_fit, gt),
            "nrmse_loo": _fg_nrmse(mean_sub, gt),
            "nrmse_z0": _fg_nrmse(pred_z0, gt),
        }
        r["gate3_pass"] = r["nrmse_fit"] < r["nrmse_loo"]
        r["ratio_z0_fit"] = r["nrmse_z0"] / max(r["nrmse_fit"], 1e-9)
        rows.append(r)
        print(f"  vol {i}: fit={r['nrmse_fit']:.4f}  LOO={r['nrmse_loo']:.4f}  "
              f"z0={r['nrmse_z0']:.4f}  ratio(z0/fit)={r['ratio_z0_fit']:.2f}  "
              f"{'✅' if r['gate3_pass'] else '❌'}")

    fit = np.mean([r["nrmse_fit"] for r in rows])
    loo = np.mean([r["nrmse_loo"] for r in rows])
    z0 = np.mean([r["nrmse_z0"] for r in rows])
    ratio = np.mean([r["ratio_z0_fit"] for r in rows])
    wins = sum(r["gate3_pass"] for r in rows)

    print("\n" + "-" * 70)
    print(f"[3/5] fit={fit:.4f}  vs  moyenne LOO={loo:.4f}   -> {wins}/{n_vol} volumes gagnés")
    print(f"[5/5] z=0 (cassé)={z0:.4f}  vs  fit={fit:.4f}   -> ratio moyen {ratio:.2f} "
          f"(seuil 1.05)")
    gate3 = fit < loo
    gate5 = ratio >= 1.05
    print(f"\n[3/5] {'✅ PASSE' if gate3 else '❌ ÉCHOUE'}    "
          f"[5/5] {'✅ PASSE' if gate5 else '❌ ÉCHOUE'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"gate_production_{n_steps}steps.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n💾 {csv_path}")
    return 0 if (gate3 and gate5) else 1


if __name__ == "__main__":
    sys.exit(main())
