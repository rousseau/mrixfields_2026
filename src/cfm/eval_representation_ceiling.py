#!/usr/bin/env python3
"""Plafond de REPRESENTATION par architecture : que vaudrait-elle si le flow
etait parfait ?

Sans cette borne, aucun ecart de score n'est interpretable : on ne sait pas si
une architecture perd parce que son flow transporte mal ou parce que sa
representation ne peut pas restituer le volume, meme sans rien transporter.

METHODE — reutilisation exacte, pas de chemin paralle. Le plafond s'obtient en
lancant l'inference REELLE sur des paires IDENTITE (`--pairs 7T_to_7T,...`) :
`_make_flow_spec` donne alors `t_start == t_end`, donc `euler_integrate` a un
`dt` nul et laisse le latent intact. Il reste exactement
encode -> (rien) -> decode, par le meme code, la meme normalisation, le meme
tuilage et le meme reechantillonnage 1 mm -> 0.5 mm que les predictions. Chaque
architecture emprunte SON propre chemin : MedVAE pour le vectorise et l'UNet,
l'ajustement INR (Algorithme 2) pour l'INR — c'est la parite de calcul exigee
par la norme de comparaison equitable du projet.

L'evaluateur officiel code en dur les 20 paires HORS identite
(`src/evaluation/evaluate.py:43`), donc il ne peut pas scorer ces volumes. On
utilise ici la reimplementation des formules officielles de
`eval_quantitative_3arch.py`, deja verifiee bit a bit contre l'evaluateur
(ecart max 2.2e-16).

CONTROLE D'EQUITE : le vectorise et l'UNet partagent MedVAE, donc leurs plafonds
DOIVENT etre identiques. S'ils different, le chemin d'encodage differe quelque
part et aucune comparaison entre eux n'est valide. Le script le verifie.

Usage :
  # 1) produire les volumes de plafond (identite) pour une architecture
  PYTHONPATH=src python src/cfm/infer_mmfm_unified.py \
      --config configs/mmfm/vectorized.yaml \
      --checkpoint outputs/mmfm/vectorized/weights/model_final.pth \
      --output_dir outputs/mmfm/ceiling_vectorized/predictions \
      --pairs 0.1T_to_0.1T,1.5T_to_1.5T,3T_to_3T,5T_to_5T,7T_to_7T \
      --modalities T1W T2W T2FLAIR \
      --field_norm_stats configs/mmfm/field_norm_stats.json

  # 2) scorer
  PYTHONPATH=src python src/cfm/eval_representation_ceiling.py \
      --pred-root outputs/mmfm/ceiling_vectorized/predictions/task3 \
      --name vectorized --outdir results/mmfm/ceiling_20260827
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.config import load_env
from cfm.eval_qualitative_3arch import gt_path, load_vol
from cfm.eval_quantitative_3arch import (
    FIELDS, MODALITIES, SUBJECTS, official_nrmse, official_ssim,
)

IDENTITY_PAIRS = ",".join(f"{f}_to_{f}" for f in FIELDS)


def score(pred_root: Path, name: str, outdir: Path) -> dict:
    root = Path(load_env("local")["data_root"])
    outdir.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    for mod in MODALITIES:
        rows = []
        for f in FIELDS:
            vals = []
            for sid in SUBJECTS:
                p = pred_root / mod / f"{f}_to_{f}" / f"P_{mod}_{f}_{sid}.nii.gz"
                g = gt_path(root, mod, f, sid)
                if not p.exists():
                    print(f"  [manquant] {p}")
                    continue
                pv, gv = load_vol(p), load_vol(g)
                vals.append((official_nrmse(pv, gv), official_ssim(pv, gv)))
            if not vals:
                continue
            n = np.array([v[0] for v in vals])
            s = np.array([v[1] for v in vals])
            rows.append({"method": f"ceiling_{name}", "task": "ceiling", "field": f,
                         "n_subjects": len(vals),
                         "nrmse_mean": float(n.mean()), "ssim_mean": float(s.mean()),
                         "nrmse_std": float(n.std()), "ssim_std": float(s.std())})
        if not rows:
            continue
        out = outdir / f"ceiling_{name}_{mod}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        summary[mod] = rows
        print(f"  {mod:8s} " + "  ".join(
            f"{r['field']}:{r['nrmse_mean']:.4f}" for r in rows)
            + f"   -> {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-root", required=True,
                    help="racine des volumes de plafond, p.ex. .../predictions/task3")
    ap.add_argument("--name", required=True, help="vectorized | unet | inr")
    ap.add_argument("--outdir", default="results/mmfm/ceiling_20260827")
    ap.add_argument("--compare", default=None,
                    help="autre nom deja score, pour le controle d'equite "
                         "(le vectorise et l'UNet DOIVENT avoir le meme plafond)")
    a = ap.parse_args()

    print(f"Plafond de representation — {a.name}")
    s = score(Path(a.pred_root), a.name, Path(a.outdir))
    if not s:
        raise SystemExit("aucun volume score — lancer d'abord l'inference identite "
                         f"avec --pairs {IDENTITY_PAIRS}")

    if a.compare:
        print(f"\nControle d'equite : {a.name} contre {a.compare}")
        worst = 0.0
        for mod in s:
            other = Path(a.outdir) / f"ceiling_{a.compare}_{mod}.csv"
            if not other.exists():
                print(f"  {mod}: {other} absent, controle impossible")
                continue
            ref = {r["field"]: float(r["nrmse_mean"]) for r in csv.DictReader(open(other))}
            for r in s[mod]:
                d = abs(r["nrmse_mean"] - ref.get(r["field"], r["nrmse_mean"]))
                worst = max(worst, d)
                if d > 1e-6:
                    print(f"  ECART {mod} {r['field']}: {r['nrmse_mean']:.6f} "
                          f"contre {ref[r['field']]:.6f}  (delta {d:.2e})")
        if worst <= 1e-6:
            print("  plafonds identiques — le chemin d'encodage est bien partage")
        else:
            print(f"  ATTENTION : ecart max {worst:.2e}. Les deux architectures ne "
                  f"partagent pas le meme encodage ; toute comparaison entre elles "
                  f"melange representation et flow.")


if __name__ == "__main__":
    main()
