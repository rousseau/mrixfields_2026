#!/usr/bin/env python3
"""Evalue un checkpoint du harnais synthetique contre la reponse EXACTE.

Deux mesures, celles qui ont revele le defaut du 2026-08-27 :

  1. nRMSE par marginale — le transport parfait est calculable echantillon par
     echantillon (meme sujet, position du champ cible sur la courbe), donc on
     ne se contente pas d'une distance entre distributions.
  2. courbure de la trajectoire integree / courbure exigee — un modele aveugle
     au temps integre une droite et ne peut atteindre aucune ancre intermediaire.

PORTE. Le correctif est valide si ratio de courbure >= 0.80 et
nRMSE intermediaire <= 2x le plancher de bruit. Tant qu'elle n'est pas
franchie, inutile de payer 25 000 iterations sur donnees reelles.

Usage :
    PYTHONPATH=src python src/cfm/eval_synthetic.py --config configs/mmfm/synthetic.yaml
    PYTHONPATH=src python src/cfm/eval_synthetic.py --config ... --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfm.arch_synthetic import spec_from_cfg
from cfm.arch_vector import build_vector_mmfm
from cfm.synthetic_marginals import SyntheticMarginals, evaluate

BEND_GATE = 0.80          # fraction de la courbure exigee
NRMSE_GATE_FACTOR = 2.0   # multiple du plancher de bruit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mmfm/synthetic.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--weights", default="ema", choices=["ema", "model"])
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    spec = spec_from_cfg(cfg)
    ckpt_path = a.checkpoint or str(
        Path("outputs") / cfg["data"]["output_subdir"] / "weights" / "model_final.pth")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_vector_mmfm(cfg, spec.dim, spec.n_contrasts).to(dev).eval()
    model.load_state_dict(ck[a.weights])

    dgp = SyntheticMarginals(spec)
    res = evaluate(lambda z, zs, t, y: model(z, zs, t, y), dgp, dev)

    ts = float(cfg["model"].get("time_scale", 1.0))
    print(f"Checkpoint : {ckpt_path}  (iter {ck.get('iter')}, poids '{a.weights}')")
    print(f"time_scale = {ts:g} | courbure imposee = {spec.bend} | bruit = {spec.noise}\n")

    floor = spec.noise
    bend_ratios, mid_nrmse = [], []
    for c, r in res.items():
        print(f"  contraste {c}")
        print(f"    nRMSE par champ (f0..f4) : "
              + "  ".join(f"{v:.4f}" for v in r["nrmse"]))
        print(f"    courbure integree        : "
              + "  ".join(f"{v:.4f}" for v in r["bend"]))
        print(f"    courbure exigee          : "
              + "  ".join(f"{v:.4f}" for v in r["true_bend"]))
        print(f"    fraction atteinte        : "
              + "  ".join(f"{v:.3f}" for v in r["bend_ratio"]))
        bend_ratios += r["bend_ratio"]
        mid_nrmse += r["nrmse"][1:-1]

    mean_bend = sum(bend_ratios) / len(bend_ratios)
    worst_mid = max(mid_nrmse)
    print(f"\n  fraction de courbure moyenne : {mean_bend:.3f}  (porte >= {BEND_GATE})")
    print(f"  pire nRMSE intermediaire     : {worst_mid:.4f}  "
          f"(porte <= {NRMSE_GATE_FACTOR * floor:.4f} = {NRMSE_GATE_FACTOR:g}x le bruit)")

    passed = mean_bend >= BEND_GATE and worst_mid <= NRMSE_GATE_FACTOR * floor
    print(f"\n  PORTE {'FRANCHIE' if passed else 'NON FRANCHIE'}")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"checkpoint": ckpt_path, "time_scale": ts,
                   "results": {str(k): v for k, v in res.items()},
                   "mean_bend_ratio": mean_bend, "worst_mid_nrmse": worst_mid,
                   "passed": passed}, open(a.json_out, "w"), indent=2)
        print(f"  ecrit -> {a.json_out}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
