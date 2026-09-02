#!/usr/bin/env python3
"""Le flow n'est-il qu'une TRANSLATION dependante de `t` ?

L'OBSERVATION A EXPLIQUER (diagnostic du 2026-08-31,
`results/mmfm/contraction_20260831/manifest.md`). Deux faits, laisses alors sans
hypothese :

  1. la dispersion inter-sujets des latents PREDITS est quasi identique quel que
     soit le champ cible -- en T2FLAIR : 2651, 2651, 2651, 2650, identique a
     quatre chiffres, alors que les reels valent 2208, 2277, 2465, 2171 en T1W ;
  2. le rang effectif est la seule quantite deficitaire (0.862).

L'HYPOTHESE QUE CE SCRIPT TESTE. Une pure translation expliquerait les DEUX d'un
coup. Si la vitesse ne depend pas de `z`, alors

    z_pred,i = z_src,i + A(t_cible, y)     pour tout sujet i

donc (a) les ecarts au centroide sont preserves EXACTEMENT, ce qui rend la
dispersion identique pour toutes les cibles, et (b) le rang effectif du nuage
predit est exactement celui du nuage SOURCE -- qui n'a aucune raison d'egaler
celui de la cible.

Le diagnostic du 2026-08-31 comparait predit contre reel-a-la-cible. Il manquait
le temoin decisif : le nuage **source**. Si predit ~ source sur les deux
quantites, l'observation se reduit a « le flow translate », et ce n'est plus une
enigme mais un defaut nomme.

LA MESURE DIRECTE, qui ne depend d'aucun temoin. Le deplacement de chaque sujet
est `D_i = z_pred,i - z_src,i`. On le decompose en une part COMMUNE et une part
PROPRE AU SUJET :

    part propre = moyenne_i ||D_i - D_moyen||  /  ||D_moyen||

Une pure translation donne 0. Plus ce rapport est grand, plus le flow adapte son
deplacement au sujet. C'est le nombre qui tranche.

Usage :
    PYTHONPATH=src python src/cfm/diagnose_flow_translation.py
    PYTHONPATH=src python src/cfm/diagnose_flow_translation.py \\
        --config configs/mmfm/vectorized_lpips.yaml --json-out /tmp/trans_lpips.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfm.arch_vector import build_vector_mmfm
from cfm.diagnose_flow_contraction import CACHE, FIELDS, LATENT_DIM, MODALITIES, dispersion, load_cached
from cfm.mmfm_core import _field_to_time, euler_integrate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mmfm/vectorized_rbest.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--src-field", default="0.1T")
    ap.add_argument("--cache", default=CACHE,
                    help="le cache doit correspondre au VAE de la config")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ckpt = a.checkpoint or str(
        Path("outputs") / cfg["data"]["output_subdir"] / "weights" / "model_final.pth")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_vector_mmfm(cfg, LATENT_DIM, len(MODALITIES)).to(dev).eval()
    model.load_state_dict(ck["ema"])

    base = Path(a.cache).parent.parent
    idx = json.load(open(Path(a.cache) / "index.json"))["samples"]
    use_amp = bool(cfg["train"].get("use_amp", True))

    def integrate(z_src: torch.Tensor, mi: int, t0: float, t1: float) -> torch.Tensor:
        outs = []
        for i in range(0, z_src.shape[0], a.chunk):
            zc = z_src[i:i + a.chunk].to(dev)
            y = torch.full((zc.shape[0],), mi, dtype=torch.long, device=dev)
            outs.append(euler_integrate(lambda z, zs, t, yy: model(z, zs, t, yy),
                                        zc, y, t0, t1, a.n_steps, dev,
                                        use_amp=use_amp).cpu())
        return torch.cat(outs)

    print(f"Checkpoint : {ckpt}  (iter {ck.get('iter')})")
    print(f"Cache      : {a.cache}")
    print(f"{a.n} latents par nuage, source {a.src_field}, {a.n_steps} pas\n")

    t_src = _field_to_time(FIELDS.index(a.src_field), len(FIELDS))
    res: dict = {}

    print(f"  {'contraste':10s}{'cible':7s}{'part propre au sujet':>22s}"
          f"{'disp. pred/src':>16s}{'rang pred/src':>15s}{'rang reel':>11s}")
    for mi, mod in enumerate(MODALITIES):
        z_src = load_cached(idx, base, mod, a.src_field, a.n, seed=0)
        d_src = dispersion(z_src)
        res[mod] = {"source": d_src, "targets": {}}
        for f in FIELDS:
            if f == a.src_field:
                continue
            t_tgt = _field_to_time(FIELDS.index(f), len(FIELDS))
            z_pred = integrate(z_src, mi, t_src, t_tgt)
            z_real = load_cached(idx, base, mod, f, a.n, seed=1)
            dp, dr = dispersion(z_pred), dispersion(z_real)

            # LA mesure : quelle part du deplacement est propre au sujet ?
            D = z_pred - z_src
            Dm = D.mean(0, keepdim=True)
            own = float(((D - Dm).norm(dim=1).mean() / Dm.norm().clamp_min(1e-12)).item())

            res[mod]["targets"][f] = {"own_share": own, "pred": dp, "real": dr,
                                      "disp_pred_over_src": dp["inter"] / d_src["inter"],
                                      "rank_pred_over_src": dp["eff_rank"] / d_src["eff_rank"]}
            print(f"  {mod:10s}{f:7s}{own:22.4f}"
                  f"{dp['inter']/d_src['inter']:16.4f}"
                  f"{dp['eff_rank']/d_src['eff_rank']:15.4f}"
                  f"{dr['eff_rank']:11.1f}")
        print(f"  {'':10s}{'(source)':7s}{'':22s}{'1.0000':>16s}"
              f"{'1.0000':>15s}{d_src['eff_rank']:11.1f}")

    owns = [t["own_share"] for m in res.values() for t in m["targets"].values()]
    dsp = [t["disp_pred_over_src"] for m in res.values() for t in m["targets"].values()]
    rnk = [t["rank_pred_over_src"] for m in res.values() for t in m["targets"].values()]
    import numpy as np
    print(f"\n  part du deplacement propre au sujet : {np.mean(owns):.4f} "
          f"(min {min(owns):.4f}, max {max(owns):.4f})")
    print(f"  dispersion predite / source          : {np.mean(dsp):.4f} "
          f"(min {min(dsp):.4f}, max {max(dsp):.4f})")
    print(f"  rang effectif predit / source        : {np.mean(rnk):.4f} "
          f"(min {min(rnk):.4f}, max {max(rnk):.4f})")
    print()
    if np.mean(dsp) > 0.97 and np.mean(rnk) > 0.97:
        print("  => Le nuage predit est celui de la SOURCE, transporte en bloc.")
        print("     L'observation du 2026-08-31 se reduit a : le flow TRANSLATE.")
        print("     Le deficit de rang (0.862) n'est alors pas un defaut du flow mais")
        print("     le rang du nuage SOURCE, qui n'a aucune raison d'egaler celui de")
        print("     la cible.")
    else:
        print("  => Le nuage predit n'est PAS celui de la source : le flow deforme")
        print("     bien le nuage. L'hypothese de translation pure est ECARTEE, et le")
        print("     deficit de rang reste inexplique.")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"checkpoint": ckpt, "cache": a.cache, "n": a.n,
                   "n_steps": a.n_steps, "src_field": a.src_field, "per_contrast": res,
                   "mean": {"own_share": float(np.mean(owns)),
                            "disp_pred_over_src": float(np.mean(dsp)),
                            "rank_pred_over_src": float(np.mean(rnk))}},
                  open(a.json_out, "w"), indent=2)
        print(f"\n  ecrit -> {a.json_out}")


if __name__ == "__main__":
    main()
