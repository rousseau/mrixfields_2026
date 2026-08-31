#!/usr/bin/env python3
"""Le flow contracte-t-il ses latents vers la moyenne de la classe cible ?

L'HYPOTHESE. Le flow est entraine en L1 -- il estime donc la MEDIANE
conditionnelle -- sur un couplage INDEPENDANT (l'OT est vacuous a 129 024
dimensions : mesure du 2026-08-30, activer le couplage a donne +0.2 % au lieu des
-19 % annonces par le harnais). Un estimateur de tendance centrale sur un
couplage aleatoire produit des latents CONTRACTES vers la moyenne de la classe
cible.

Si la contraction existe, elle explique d'un coup les deux symptomes observes
depuis le debut du projet : images trop lisses (une moyenne est lisse) et
contraste incompletement transporte (une moyenne est a mi-chemin). Et elle rend
le passage a L2 ARGUMENTE, au lieu de « les 9 scripts de la reference le font ».

Si elle n'existe pas, l'hypothese L1 tombe pour la DEUXIEME fois -- elle avait
deja ete refutee le 2026-08-26 par un autre chemin (||mediane||/||moyenne|| =
0.98-1.01, la mediane conservant 106-144 % des hautes frequences) -- et on ne
paie pas les 4 h de reentrainement.

COMMENT, SANS APPARIEMENT ET SANS DECODAGE. Purement en espace latent, sur les
caches existants : on passe N latents source dans le flow, on obtient N latents
predits au champ cible, et on compare la DISPERSION de ce nuage a celle de N
latents REELS du meme champ. Comparer deux nuages ne demande aucun appariement --
ce qui compte ici, puisque aucun sujet n'existe a deux champs. Et aucun decodage
MedVAE, donc des minutes au lieu d'heures.

TEMOIN. Le script mesure la dispersion des latents REELS dans la meme execution :
sans elle, un rapport de 0.8 ne serait pas interpretable. Et il verifie sur un pas
d'IDENTITE (t_i = t_j, donc dt = 0, donc le flow ne deplace rien) que le rapport
vaut exactement 1.000 -- si ce controle echoue, la mesure est fausse et non le
modele.

Usage :
    PYTHONPATH=src python src/cfm/diagnose_flow_contraction.py \\
        --config configs/mmfm/vectorized_rbest.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfm.arch_vector import build_vector_mmfm
from cfm.mmfm_core import _field_to_time, euler_integrate

CACHE = "outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/retro_train"
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
LATENT_DIM = 129024


def load_cached(idx: list, base: Path, mod: str, field: str, n: int,
                seed: int) -> torch.Tensor:
    s = [r for r in idx if r["modality"] == mod and r["field"] == field]
    random.Random(seed).shuffle(s)
    out = []
    for r in s[:n]:
        o = torch.load(base / r["path"], map_location="cpu", weights_only=False)
        z = o["latent"] if isinstance(o, dict) and "latent" in o else o
        out.append(torch.as_tensor(z).float().reshape(-1))
    return torch.stack(out)


def dispersion(z: torch.Tensor) -> dict:
    """Trois facons de mesurer l'etalement d'un nuage de latents.

    `elem_std` : ecart-type par element, moyenne sur les elements — l'amplitude
        brute du signal.
    `inter` : distance moyenne au centroide — la dispersion INTER-SUJETS, c'est-a-
        dire ce qui distingue un sujet d'un autre. C'est la quantite qu'un
        estimateur de tendance centrale detruit en premier.
    `eff_rank` : participation ratio des valeurs singulieres du nuage centre — une
        perte de DIVERSITE, pas seulement d'amplitude. Un nuage contracte
        uniformement garde son rang ; un nuage effondre sur quelques directions le
        perd.
    """
    zc = z - z.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(zc.double())
    return {
        "elem_std": float(z.std(0).mean().item()),
        "inter": float(zc.norm(dim=1).mean().item()),
        "eff_rank": float((sv.sum() ** 2 / (sv ** 2).sum()).item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mmfm/vectorized_rbest.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n", type=int, default=24, help="latents par nuage")
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--src-field", default="0.1T")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    ckpt = a.checkpoint or str(
        Path("outputs") / cfg["data"]["output_subdir"] / "weights" / "model_final.pth")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_vector_mmfm(cfg, LATENT_DIM, len(MODALITIES)).to(dev).eval()
    model.load_state_dict(ck["ema"])

    base = Path(CACHE).parent.parent
    idx = json.load(open(Path(CACHE) / "index.json"))["samples"]
    use_amp = bool(cfg["train"].get("use_amp", True))

    def model_fn(z, zs, t, y):
        return model(z, zs, t, y)

    def integrate(z_src: torch.Tensor, mi: int, t0: float, t1: float) -> torch.Tensor:
        outs = []
        for i in range(0, z_src.shape[0], a.chunk):
            zc = z_src[i:i + a.chunk].to(dev)
            y = torch.full((zc.shape[0],), mi, dtype=torch.long, device=dev)
            outs.append(euler_integrate(model_fn, zc, y, t0, t1, a.n_steps, dev,
                                        use_amp=use_amp).cpu())
        return torch.cat(outs)

    print(f"Checkpoint : {ckpt}  (iter {ck.get('iter')})")
    print(f"{a.n} latents par nuage, source {a.src_field}, {a.n_steps} pas d'Euler\n")

    # -- CONTROLE DE COHERENCE : un pas d'identite ne doit rien deplacer --------
    z0 = load_cached(idx, base, "T2W", a.src_field, a.chunk, seed=0)
    t_src = _field_to_time(FIELDS.index(a.src_field), len(FIELDS))
    z_id = integrate(z0, 1, t_src, t_src)
    rel = float((z_id - z0).norm() / z0.norm().clamp_min(1e-12))
    print(f"  controle identite (dt = 0) : ecart relatif {rel:.2e} "
          f"-> {'OK' if rel < 1e-6 else 'ECHEC — la mesure est fausse, pas le modele'}")
    if rel >= 1e-6:
        raise SystemExit("le controle d'identite a echoue")

    # -- MESURE ----------------------------------------------------------------
    res: dict = {}
    print(f"\n  {'contraste':10s}{'cible':7s}{'std elem':>20s}{'inter-sujets':>20s}{'rang eff.':>18s}")
    print(f"  {'':10s}{'':7s}{'predit/reel':>20s}{'predit/reel':>20s}{'predit/reel':>18s}")
    ratios = {"elem_std": [], "inter": [], "eff_rank": []}
    for mi, mod in enumerate(MODALITIES):
        z_src = load_cached(idx, base, mod, a.src_field, a.n, seed=0)
        for tgt in FIELDS:
            if tgt == a.src_field:
                continue
            t_tgt = _field_to_time(FIELDS.index(tgt), len(FIELDS))
            z_pred = integrate(z_src, mi, t_src, t_tgt)
            z_real = load_cached(idx, base, mod, tgt, a.n, seed=1)
            dp, dr = dispersion(z_pred), dispersion(z_real)
            row = {k: dp[k] / dr[k] if dr[k] > 1e-12 else float("nan") for k in dp}
            res[f"{mod}/{tgt}"] = {"pred": dp, "real": dr, "ratio": row}
            for k in ratios:
                ratios[k].append(row[k])
            print(f"  {mod:10s}{tgt:7s}"
                  f"{dp['elem_std']:8.3f}/{dr['elem_std']:<7.3f}{row['elem_std']:5.2f}"
                  f"{dp['inter']:9.0f}/{dr['inter']:<7.0f}{row['inter']:5.2f}"
                  f"{dp['eff_rank']:8.1f}/{dr['eff_rank']:<5.1f}{row['eff_rank']:5.2f}")

    print(f"\n  MOYENNE des rapports predit/reel")
    for k, lab in (("elem_std", "amplitude (ecart-type par element)"),
                   ("inter", "dispersion INTER-SUJETS"),
                   ("eff_rank", "rang effectif (diversite)")):
        v = float(np.nanmean(ratios[k]))
        print(f"    {lab:38s} {v:.3f}")

    inter = float(np.nanmean(ratios["inter"]))
    print(f"\n  VERDICT")
    if inter < 0.85:
        print(f"    CONTRACTION MESUREE : la dispersion inter-sujets des latents")
        print(f"    predits ne vaut que {100*inter:.0f} % de celle des latents reels.")
        print(f"    Le flow rend une tendance centrale -- ce qui explique a la fois")
        print(f"    le lissage et le contraste incompletement transporte, et justifie")
        print(f"    d'essayer L2 (partie D du plan).")
    elif inter > 1.15:
        print(f"    EXPANSION ({100*inter:.0f} %) — l'inverse de l'hypothese.")
    else:
        print(f"    PAS DE CONTRACTION ({100*inter:.0f} %, dans [85, 115] %).")
        print(f"    L'hypothese L1/tendance centrale tombe pour la DEUXIEME fois.")
        print(f"    La partie D du plan ne doit PAS etre lancee.")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"checkpoint": ckpt, "n": a.n, "n_steps": a.n_steps,
                   "src_field": a.src_field, "identity_check": rel,
                   "per_cell": res,
                   "mean_ratio": {k: float(np.nanmean(v)) for k, v in ratios.items()}},
                  open(a.json_out, "w"), indent=2)
        print(f"\n  ecrit -> {a.json_out}")


if __name__ == "__main__":
    main()
