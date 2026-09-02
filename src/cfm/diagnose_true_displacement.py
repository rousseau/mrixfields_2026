"""Le VRAI deplacement entre champs est-il, lui aussi, commun a tous les sujets ?

Le flow appris est une translation a 3.5 % pres (mesure du 2026-09-02). Ce n'est
un DEFAUT que si la vraie transformation champ -> champ est, elle, propre au
sujet. Mesurable seulement sur les 3 sujets APPARIES (Training_prospective), les
seuls du jeu a exister a plusieurs champs.

Meme decomposition que pour le flow :
    part propre = moyenne_i ||D_i - D_moyen|| / ||D_moyen||   avec D_i = z_j,i - z_i,i
Une transformation purement commune donne 0.
"""
import json, torch, numpy as np
from pathlib import Path
CACHE = Path("outputs/mmfm/latent_cache/vectorized/medvae_finetune_c4d1e200/pro_train")
base = CACHE.parent.parent
idx = json.load(open(CACHE / "index.json"))["samples"]
F = ["0.1T", "1.5T", "3T", "5T", "7T"]
def get(mod, field):
    out = {}
    for r in idx:
        if r["modality"] == mod and r["field"] == field:
            sid = Path(r["path"]).stem.split("_")[-1]
            o = torch.load(base / r["path"], map_location="cpu", weights_only=False)
            z = o["latent"] if isinstance(o, dict) and "latent" in o else o
            out[sid] = torch.as_tensor(z).float().reshape(-1)
    return out
print("Part du VRAI deplacement propre au sujet (3 sujets apparies)\n")
print(f"  {'contraste':10s}{'paire':16s}{'part propre':>13s}{'||D moyen||':>13s}{'ecart-type':>12s}")
allv = []
for mod in ("T1W", "T2W", "T2FLAIR"):
    zs = {f: get(mod, f) for f in F}
    for i in range(len(F)):
        for j in range(len(F)):
            if i == j: continue
            sids = sorted(set(zs[F[i]]) & set(zs[F[j]]))
            if len(sids) < 3: continue
            D = torch.stack([zs[F[j]][s] - zs[F[i]][s] for s in sids])
            Dm = D.mean(0, keepdim=True)
            own = float(((D - Dm).norm(dim=1).mean() / Dm.norm().clamp_min(1e-12)).item())
            allv.append(own)
            if F[i] == "0.1T":
                print(f"  {mod:10s}{F[i]+'->'+F[j]:16s}{own:13.4f}{Dm.norm():13.1f}"
                      f"{(D-Dm).norm(dim=1).mean():12.1f}")
print(f"\n  sur les 20 paires x 3 contrastes : part propre moyenne {np.mean(allv):.4f} "
      f"(min {min(allv):.4f}, max {max(allv):.4f})")
print(f"  le FLOW appris, lui, vaut 0.0348")
print()
if np.mean(allv) > 0.20:
    print("  => Le vrai deplacement est LARGEMENT propre au sujet, alors que le flow")
    print("     ne l'est qu'a 3.5 %. Le flow rate donc une composante reelle et")
    print("     dominante : c'est un DEFAUT nomme, et un levier a instruire.")
else:
    print("  => Le vrai deplacement est lui aussi majoritairement COMMUN. La")
    print("     translation apprise est alors le bon comportement, et cette piste")
    print("     se referme : il n'y a pas de composante propre au sujet a capturer.")
