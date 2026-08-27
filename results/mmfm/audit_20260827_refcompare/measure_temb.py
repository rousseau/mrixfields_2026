import math, torch, sys
sys.path.insert(0, "src")
from cfm.mmfm_vectorized import sinusoidal_time_embedding

T = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
FIELDS = ["0.1T", "1.5T", "3T", "5T", "7T"]

for dim, name in [(256, "vectorise/INR (time_embed_dim=256)")]:
    print(f"===== {name} =====")
    for scale, tag in [(1.0, "ACTUEL   (t in [0,1])"), (1000.0, "REFERENCE (t*1000)")]:
        e = sinusoidal_time_embedding(T * scale, dim)
        # distance entre champs adjacents, normalisee par la norme de l'embedding
        d = torch.cdist(e, e)
        adj = torch.tensor([d[i, i+1] for i in range(4)])
        nrm = e.norm(dim=1).mean()
        # rang effectif : energie de la matrice centree
        ec = e - e.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(ec)
        er = (sv.sum()**2 / (sv**2).sum()).item()   # rang effectif (participation ratio)
        # amplitude du SIGNAL (ecart-type par canal a travers les 5 champs)
        chan_std = ec.std(0)
        print(f"  {tag}")
        print(f"    ||emb|| moyen                 : {nrm:.3f}")
        print(f"    distance champs adjacents     : {adj.min():.4f} .. {adj.max():.4f}")
        print(f"    distance relative (d/||emb||) : {(adj.mean()/nrm).item():.5f}")
        print(f"    rang effectif (max 4)         : {er:.2f}")
        print(f"    canaux avec std > 1e-3        : {(chan_std > 1e-3).sum().item()} / {dim}")
        print(f"    std moyen par canal           : {chan_std.mean():.2e}")
