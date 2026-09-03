#!/usr/bin/env python3
"""L'OT retrouve-t-il le BON sujet, sur nos latents reels ?

POURQUOI CE TEST PLUTOT QUE LE HARNAIS. Le harnais synthetique ne reproduit pas
l'effondrement de l'OT (0.767 de qualite de couplage a dim 129 024 le 2026-09-03,
la ou la production ne gagne rien du tout). La raison est identifiable : son
facteur sujet est UNIDIMENSIONNEL — une seule direction commune a tous les sujets
— alors que l'anatomie reelle est de haute dimension. Conclure depuis ce regime
serait conclure depuis un probleme plus facile que le notre.

Or il existe un test DIRECT sur nos donnees. Les 3 sujets de
`Training_prospective` sont les seuls du jeu a exister a plusieurs champs. Pour
chaque (contraste, paire de champs), on a donc 3 latents source et 3 latents cible
dont on CONNAIT l'appariement veritable. On demande a l'OT d'apparier, et on
compte combien de fois il tombe juste.

Hasard : pour une permutation uniforme de 3 elements, le nombre attendu de points
fixes vaut exactement 1, soit **1/3**. Au-dessus, l'OT porte de l'information ;
a 1/3, il n'en porte aucune — et c'est ce que la mesure du 2026-08-30 (+0.2 % en
activant le couplage) laisse presager.

On compare, sur les memes cellules :
  - OT sur les latents PLEINS (129 024 dimensions) ;
  - OT sur une ACP a k composantes, calculee sur le nuage source+cible ;
  - OT sur une projection ALEATOIRE de meme dimension (temoin : elle preserve les
    distances, donc leur concentration, mais pas les directions de variation).

Usage :
    PYTHONPATH=src python src/cfm/diagnose_coupling_real.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CACHE = Path("outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/pro_train")
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]


def assignment(x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
    """Affectation OT exacte : pour chaque source, l'indice cible retenu."""
    from torchcfm.optimal_transport import OTPlanSampler
    pi = np.asarray(OTPlanSampler(method="exact").get_map(x0, x1))
    # affectation 1-1 par l'algorithme hongrois sur -pi (le plan est doublement
    # stochastique ; argmax par ligne pourrait choisir deux fois la meme cible)
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-pi)
    out = np.empty(len(r), dtype=int)
    out[r] = c
    return out


def reduce_pca(x0, x1, k):
    z = torch.cat([x0, x1], 0).double()
    z = z - z.mean(0, keepdim=True)
    _, _, vt = torch.linalg.svd(z, full_matrices=False)
    v = vt[:min(k, vt.shape[0])].T
    return (x0.double() @ v).float(), (x1.double() @ v).float()


def reduce_random(x0, x1, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(x0.shape[1], k, generator=g) / np.sqrt(k)
    return x0 @ p, x1 @ p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    cache = Path(a.cache)
    base = cache.parent.parent
    idx = json.load(open(cache / "index.json"))["samples"]

    def load(mod, field):
        out = {}
        for r in idx:
            if r["modality"] == mod and r["field"] == field:
                sid = Path(r["path"]).stem.split("_")[-1]
                o = torch.load(base / r["path"], map_location="cpu", weights_only=False)
                z = o["latent"] if isinstance(o, dict) and "latent" in o else o
                out[sid] = torch.as_tensor(z).float().reshape(-1)
        return out

    zs = {(m, f): load(m, f) for m in MODALITIES for f in FIELDS}
    n_sub = len(zs[(MODALITIES[0], FIELDS[0])])
    print(f"Cache : {cache}")
    print(f"{n_sub} sujets apparies — hasard = 1/{n_sub} = {1/n_sub:.3f}\n")

    methods = ["plein"] + [f"acp{k}" for k in a.k] + [f"alea{k}" for k in a.k]
    hits = {m: 0 for m in methods}
    trials = 0
    per_mod = {m: {mm: 0 for mm in methods} for m in MODALITIES}
    per_mod_n = {m: 0 for m in MODALITIES}

    for mod in MODALITIES:
        for i in range(len(FIELDS)):
            for j in range(len(FIELDS)):
                if i == j:
                    continue
                A, B = zs[(mod, FIELDS[i])], zs[(mod, FIELDS[j])]
                sids = sorted(set(A) & set(B))
                if len(sids) < 2:
                    continue
                x0 = torch.stack([A[s] for s in sids])
                x1 = torch.stack([B[s] for s in sids])
                truth = np.arange(len(sids))          # meme ordre = meme sujet
                cand = {"plein": (x0, x1)}
                for k in a.k:
                    cand[f"acp{k}"] = reduce_pca(x0, x1, k)
                    cand[f"alea{k}"] = reduce_random(x0, x1, k, seed=i * 5 + j)
                for name, (p0, p1) in cand.items():
                    perm = assignment(p0, p1)
                    h = int((perm == truth).sum())
                    hits[name] += h
                    per_mod[mod][name] += h
                trials += len(sids)
                per_mod_n[mod] += len(sids)

    print(f"  {'methode':10s}{'justes':>9s}{'sur':>6s}{'taux':>9s}")
    res = {}
    for m in methods:
        res[m] = hits[m] / trials
        print(f"  {m:10s}{hits[m]:9d}{trials:6d}{res[m]:9.3f}")
    print(f"  {'hasard':10s}{'':9s}{'':6s}{1/n_sub:9.3f}")

    print(f"\n  par contraste (taux) :")
    print(f"  {'':10s}" + "".join(f"{m:>9s}" for m in methods))
    for mod in MODALITIES:
        print(f"  {mod:10s}" + "".join(
            f"{per_mod[mod][m]/max(per_mod_n[mod],1):9.3f}" for m in methods))

    from scipy.stats import binomtest
    print()
    for m in methods:
        p = binomtest(hits[m], trials, 1.0 / n_sub, alternative="greater").pvalue
        verdict = "porte de l'information" if p < 0.05 else "indiscernable du hasard"
        print(f"  {m:10s} contre le hasard : p = {p:.4f}  -> {verdict}")

    best = max([f"acp{k}" for k in a.k], key=lambda m: res[m])
    print()
    if res["plein"] < 1.5 / n_sub and res[best] > 2.0 / n_sub:
        print("  => L'OT plein n'apparie pas mieux que le hasard, et l'ACP le RESTAURE.")
        print("     Le couplage en dimension reduite est un levier reel a implementer.")
    elif res["plein"] > 2.0 / n_sub:
        print("  => L'OT plein apparie DEJA correctement. L'echec du couplage en")
        print("     production ne vient donc pas de l'appariement lui-meme.")
    else:
        print("  => Ni l'OT plein ni l'ACP n'apparient mieux que le hasard sur nos")
        print("     latents. Le levier « dimension reduite » ne tient pas ainsi.")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n_subjects": n_sub, "trials": trials, "rates": res,
                   "chance": 1.0 / n_sub}, open(a.json_out, "w"), indent=2)
        print(f"\n  ecrit -> {a.json_out}")


if __name__ == "__main__":
    main()
