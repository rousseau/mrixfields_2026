#!/usr/bin/env python3
"""Le couplage OT retrouve-t-il les bons appariements, et jusqu'a quelle dimension ?

CONTEXTE. Le 2026-09-02 a etabli que le flow appris est une translation (3.5 % du
deplacement propre au sujet) alors que le vrai transport champ->champ l'est a
71.7 %. La cause est l'entrainement NON APPARIE : sans couplage, la composante
individuelle du deplacement est du bruit du point de vue du modele. Le couplage OT
existe pour reparer cela -- et il est vacuous a 129 024 dimensions (+0.2 % mesure
le 2026-08-30), la concentration des distances rendant la structure de plus proche
voisin essentiellement aleatoire.

L'IDEE TESTEE ICI : calculer le plan OT dans un espace REDUIT, ou les distances
redeviennent informatives, puis appliquer la permutation obtenue aux latents
pleins.

POURQUOI CE TEST NE DEMANDE AUCUN ENTRAINEMENT. Le probleme synthetique connait le
facteur sujet `s` de chaque echantillon, et la vraie carte le PRESERVE. Le bon
couplage est donc celui qui apparie les `s` voisins, et on peut le noter
directement : correlation de Spearman entre `s` de la source et `s` du partenaire
choisi. Aleatoire = 0, parfait = 1. Quelques secondes par point.

DEUX REDUCTIONS COMPAREES, et ce n'est pas indifferent :
  - projection ALEATOIRE (Johnson-Lindenstrauss) : elle PRESERVE les distances,
    donc elle preserve aussi leur concentration. Si l'hypothese « c'est la
    concentration qui tue l'OT » est juste, elle ne doit RIEN rapporter. C'est le
    temoin qui rend le test falsifiable.
  - ACP : elle projette sur les directions ou les donnees varient reellement. Si
    l'identite du sujet y vit, l'OT doit y redevenir informatif.

Usage :
    PYTHONPATH=src python src/cfm/diagnose_coupling_dimension.py
    PYTHONPATH=src python src/cfm/diagnose_coupling_dimension.py --real
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfm.synthetic_marginals import SyntheticMarginals, SyntheticSpec


def ot_permutation(x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
    """Plan OT exact entre deux nuages, rendu comme une permutation de x1."""
    from torchcfm.optimal_transport import OTPlanSampler
    s = OTPlanSampler(method="exact")
    pi = s.get_map(x0, x1)
    # pour chaque source, le partenaire de masse maximale
    return np.asarray(pi).argmax(axis=1)


def reduce_pca(x0: torch.Tensor, x1: torch.Tensor, k: int):
    """ACP sur le nuage POOLE (source + cible), k composantes."""
    z = torch.cat([x0, x1], 0).double()
    z = z - z.mean(0, keepdim=True)
    # SVD economique : k << n < d
    u, s, vt = torch.linalg.svd(z, full_matrices=False)
    v = vt[:k].T
    return (x0.double() @ v).float(), (x1.double() @ v).float()


def reduce_random(x0: torch.Tensor, x1: torch.Tensor, k: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(x0.shape[1], k, generator=g) / np.sqrt(k)
    return x0 @ p, x1 @ p


def coupling_quality(s0: np.ndarray, s1: np.ndarray, perm: np.ndarray) -> float:
    """Spearman entre le facteur sujet de la source et celui du partenaire choisi.
    La vraie carte preserve `s`, donc 1.0 = couplage parfait, 0 = aleatoire."""
    r = spearmanr(s0, s1[perm]).statistic
    return float(r) if np.isfinite(r) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+",
                    default=[64, 512, 4096, 32768, 129024])
    ap.add_argument("--k", type=int, nargs="+", default=[4, 16, 64],
                    help="dimensions reduites a tester")
    ap.add_argument("--n", type=int, default=64, help="echantillons par nuage")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    print("Qualite du couplage OT — Spearman entre le facteur sujet de la source")
    print("et celui du partenaire choisi. 1.0 = parfait, 0 = aleatoire.")
    print(f"{a.n} echantillons par nuage, {a.repeats} tirages, marginales 0.1T -> 7T.\n")
    head = f"  {'dim':>8s}{'OT plein':>11s}" + "".join(f"{'ACP '+str(k):>11s}" for k in a.k) \
           + "".join(f"{'alea '+str(k):>11s}" for k in a.k)
    print(head)
    res = {}
    for d in a.dims:
        spec = SyntheticSpec(dim=d, bend=0.6, noise=0.05, seed=0)
        dgp = SyntheticMarginals(spec)
        acc = {"full": []} | {f"pca{k}": [] for k in a.k} | {f"rand{k}": [] for k in a.k}
        for rep in range(a.repeats):
            g = torch.Generator().manual_seed(100 + rep)
            x0, s0 = dgp.sample(0, 0, a.n, g)
            x1, s1 = dgp.sample(0, 4, a.n, g)
            s0, s1 = s0.squeeze().numpy(), s1.squeeze().numpy()
            acc["full"].append(coupling_quality(s0, s1, ot_permutation(x0, x1)))
            for k in a.k:
                p0, p1 = reduce_pca(x0, x1, k)
                acc[f"pca{k}"].append(coupling_quality(s0, s1, ot_permutation(p0, p1)))
                r0, r1 = reduce_random(x0, x1, k, seed=rep)
                acc[f"rand{k}"].append(coupling_quality(s0, s1, ot_permutation(r0, r1)))
        res[d] = {kk: float(np.mean(v)) for kk, v in acc.items()}
        print(f"  {d:8d}{res[d]['full']:11.3f}"
              + "".join(f"{res[d]['pca'+str(k)]:11.3f}" for k in a.k)
              + "".join(f"{res[d]['rand'+str(k)]:11.3f}" for k in a.k))

    print()
    lo, hi = min(a.dims), max(a.dims)
    print(f"  OT plein : {res[lo]['full']:.3f} a dim {lo}  ->  {res[hi]['full']:.3f} a dim {hi}")
    best_k = max(a.k, key=lambda k: res[hi][f"pca{k}"])
    print(f"  ACP {best_k:<3d} : {res[hi]['pca'+str(best_k)]:.3f} a dim {hi}")
    print(f"  alea {best_k:<2d} : {res[hi]['rand'+str(best_k)]:.3f} a dim {hi}  "
          f"(temoin : une projection aleatoire preserve les distances, donc leur concentration)")
    print()
    if res[hi]["full"] < 0.3 and res[hi][f"pca{best_k}"] > 0.6:
        print("  => L'OT s'effondre en haute dimension et l'ACP le RESTAURE.")
        print("     Le couplage en dimension reduite est un levier reel : a implementer.")
    elif res[hi]["full"] > 0.6:
        print("  => L'OT ne s'effondre PAS sur ce probleme synthetique : il ne reproduit")
        print("     donc pas le regime reel, et ce test ne conclut rien sur nos donnees.")
    else:
        print("  => L'ACP ne restaure pas le couplage. Le levier « dimension reduite »")
        print("     ne tient pas, au moins sous cette forme.")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"n": a.n, "repeats": a.repeats, "results": res}, open(a.json_out, "w"), indent=2)
        print(f"\n  ecrit -> {a.json_out}")


if __name__ == "__main__":
    main()
