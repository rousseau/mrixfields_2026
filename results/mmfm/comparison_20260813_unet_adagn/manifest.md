# UNet + conditionnement AdaGN — résultat : NÉGATIF (aucun effet mesurable)

**Date** : 2026-08-13
**Verdict** : AdaGN ne change rien. nRMSE 0.4623 contre 0.4617 pour le UNet de
référence — les deux modèles produisent des prédictions distantes de 0.2-0.3 %.

---

## Ce qui était testé

`DiffusionUNetResnetBlock` (MONAI) conditionne par un DÉCALAGE ADDITIF injecté
avant `norm2` :

    h = h + time_emb_proj(silu(emb))
    h = conv2(silu(norm2(h)))

Deux limites : la modulation ne porte jamais sur le GAIN, et `norm2` — qui
soustrait la moyenne par groupe de canaux — efface une partie du décalage qu'on
vient d'injecter. AdaGN (Dhariwal & Nichol, arXiv:2105.05233,
`use_scale_shift_norm`) module APRÈS la normalisation :

    scale, shift = emb_proj(silu(emb)).chunk(2)
    h = norm2(h) * (1 + scale) + shift

Hypothèse : l'écart documenté du UNet se concentre sur les transitions
->5T/->7T (0.821 contre 0.676 pour le vectorisé), donc aux extrémités de l'axe
de champ, là où un conditionnement qui ne sait que décaler est le plus limité.

Implémentation : `src/models/adagn_conditioning.py` (21 blocs convertis,
+3.41 M paramètres, +1.9 %). Le bloc REPREND les sous-modules de MONAI plutôt
que de les reconstruire — aucune dépendance à la signature de son constructeur,
qui a déjà changé entre versions.

## Protocole

Identique en tout point à la référence : mêmes latents (cache 1mm
`medvae_finetune_c4d1e200`), même config `data:`/`train:`/`inference:`, 25000
itérations from scratch, même inférence (`--norm_mode field_fixed
--center_crop_only`), même évaluateur officiel, 20 paires x 3 sujets, T1W.
La reprise depuis l'ancien checkpoint est impossible (`time_emb_proj` -> `emb_proj`)
et échoue bruyamment — vérifié.

## Résultat

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| UNet référence | **0.4617** | **0.8955** | **0.1014** |
| UNet AdaGN | 0.4623 | 0.8954 | 0.1015 |
| *(vectorisé, meilleure archi du projet)* | *0.4354* | *0.8995* | *0.0985* |

3/20 paires gagnées. Écarts par champ CIBLE uniformément entre +0.0003 et
+0.0010 — **y compris sur 5T et 7T**, que le conditionnement visait :

| cible | UNet réf | AdaGN | écart |
|---|---|---|---|
| 0.1T | 0.3017 | 0.3027 | +0.0010 |
| 1.5T | 0.3526 | 0.3530 | +0.0004 |
| 3T | 0.3299 | 0.3302 | +0.0003 |
| 5T | 0.5286 | 0.5293 | +0.0007 |
| 7T | 0.7956 | 0.7966 | +0.0009 |

## Le fait le plus instructif

Les prédictions des deux modèles ne diffèrent que de **0.2-0.3 % en L2 relatif**
(mesuré sur 8 paires). Deux mécanismes de conditionnement différents, 3.4 M
paramètres d'écart, des loss d'entraînement séparées de ~2.4 points en médiane
(7.4 contre 9.9) — et une sortie quasi identique.

**Le conditionnement n'était pas le goulot.** Ce qui limite le UNet est ailleurs.

À noter : la loss d'AdaGN est restée systématiquement PLUS HAUTE (+2.4) sans que
le résultat Task 3 se dégrade pour autant (+0.0007, soit du bruit). Cinquième
confirmation que la loss d'entraînement ne prédit rien sur ce pipeline.

## Témoin IDENTITÉ — nouveau, et il manquait

Aucun modèle de ce projet n'avait jamais été comparé à « ne rien faire ». On a
donc évalué, dans le protocole EXACT, un jeu de prédictions où la prédiction est
le volume SOURCE recopié sous le nom de la cible.

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **identité (source)** | **0.9273** | 0.8699 | 0.1055 |
| vectorisé | 0.4354 | 0.8995 | 0.0985 |

Le vectorisé gagne 18/20 paires, et l'écart explose là où c'est difficile :
**5T 1.3596 -> 0.5072**, **7T 1.9391 -> 0.7568**. Les architectures apportent donc
une valeur massive et réelle — ce témoin le prouve pour la première fois.

⚠️ **Piège rencontré en produisant ce témoin** : une première mesure « maison »,
normalisée par percentiles PAR VOLUME, semblait montrer que la source battait le
modèle sur 5 paires sur 8. C'était un artefact — cette normalisation recale déjà
l'intensité de la source sur celle de la cible, ce qui accomplit une bonne part
de la tâche de translation. Seule l'évaluation dans le protocole officiel
(`field_fixed`) fait foi. Second piège, plus bête : le `nrmse: 0.4391` affiché en
fin de sortie de l'évaluateur est la DERNIÈRE PAIRE (7T_to_5T), pas une moyenne
globale.

## Fichiers

| chemin | contenu |
|---|---|
| `src/models/adagn_conditioning.py` | AdaGN + 6 auto-tests (`__main__`) |
| `src/cfm/arch_unet.py` | conversion derrière `model.use_adagn` |
| `configs/mmfm/unet.yaml` | `use_adagn: true`, sortie `mmfm/unet_adagn` |
| `src/cfm/compare_task3_csv.py` | comparaison de deux CSV Task 3, ventilée par champ |
| `task3_unet_adagn_T1W.csv` | résultat AdaGN |
| `task3_identity_baseline_T1W.csv` | témoin identité |
| `comparaison_vs_unet.txt`, `temoin_identite.txt` | sorties détaillées |

## Piège des auto-tests, à retenir

MONAI initialise à zéro `conv2` de chaque bloc résiduel ET la convolution de
sortie (`zero_module`) : **à l'initialisation, le réseau entier sort exactement
zéro**. Tout test de conditionnement mené sur un modèle fraîchement initialisé
passe donc trivialement — y compris sur une branche morte. Mes trois premiers
tests étaient creux pour cette raison ; ils réveillent maintenant les 140
tenseurs nuls avant de mesurer.
