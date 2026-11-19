"""Choix du checkpoint par (repli, contraste) : min du nRMSE latent moyen du
contraste (4 cibles depuis 0.1T) sur le sujet tenu a l'ecart, parmi les
checkpoints fine-tunes (ni le point de depart, ni le doublon « final »).
Regle verifiee le 2026-09-25 : retrouve EXACTEMENT les 6 choix de la reference
0.2843 (T1W long_lpips 3750/250/2250, T2FLAIR 5k 2750/2000/1000)."""
import json, sys
import numpy as np
MI = {"T1W": 0, "T2W": 1, "T2FLAIR": 2}
path, mod = sys.argv[1], sys.argv[2]
r = [x for x in json.load(open(path))["results"][1:] if x["label"] != "iter final"]
i = MI[mod]
b = min(r, key=lambda x: np.mean(x["nrmses"][4 * i:4 * i + 4]))
print(b["checkpoint"])
