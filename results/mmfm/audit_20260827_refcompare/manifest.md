# Comparaison au code de référence — l'échelle du temps

**Date** : 2026-08-27
**Demande** : comparer notre code à `Genentech/MMFM` (le papier multi-marginal
flow matching) et `NVIDIA-Medtech/NV-Generate-CTMR` pour localiser la source des
deux symptômes persistants — erreurs de gestion de contraste, images trop lisses.

**Verdict** : cause racine trouvée, mesurée à chaque maillon. **Le temps `t`
n'atteint pas le modèle.** Les deux références multiplient `t` par 1000 avant
l'embedding sinusoïdal ; nous avons copié la même fonction sans le facteur.

---

## 1. Le défaut

Les trois codes utilisent *la même* fonction, tracée à
`atong01/conditional-flow-matching` (`torchcfm/models/unet/nn.py`), avec
`max_period = 10000` :

| | appel |
|---|---|
| Genentech/MMFM | `timestep_embedding(t * self.step_scale, dim)`, `step_scale = 1000` (`models.py:330`, défaut `models.py:119`) |
| NVIDIA / MONAI | `timesteps = torch.randint(0, num_train_timesteps, ...)` — entiers dans `[0, 1000)` (`diff_model_train.py:301`) |
| **nous** | `sinusoidal_time_embedding(timesteps, dim)` avec **`t ∈ [0,1]` brut** (`mmfm_vectorized.py:159`) |

Avec `t ∈ [0,1]` et `freqs ∈ [1e-4, 1]`, l'argument des sinusoïdes ne dépasse
jamais 1 : `cos ≈ 1` partout et `sin(x) ≈ x`. L'embedding, prévu multi-échelle,
s'effondre en une **rampe scalaire de faible amplitude**.

Portée : `mmfm_vectorized.py` sert **le vectorisé et l'INR** (`arch_inr.py:33`
importe `build_vector_mmfm`). L'UNet passe `t ∈ [0,1]` à MONAI
(`arch_unet.py:180`), dont `get_timestep_embedding` est la même formule.
**Les trois architectures sont touchées.** Introduit au premier commit du flow
vectorisé (`2f96c13`) ; `step_scale` n'apparaît nulle part dans le projet.

## 2. Dégénérescence de l'embedding (`measure_temb.py`, `measure_temb2.py`)

Aux 5 temps de champ `t ∈ {0, 0.25, 0.5, 0.75, 1}`, `dim = 256` :

| | actuel | référence (×1000) |
|---|---|---|
| distance entre champs adjacents | 0.682 | 13.024 (**×19**) |
| distance relative `d/‖emb‖` | 0.060 | 1.151 |
| **rang effectif** (max 4) | **1.22** | **3.98** |
| écart-type moyen par canal | 2.68e-2 | 5.15e-1 |

Valeurs singulières de la matrice 5×256 centrée :

| | σ1 | σ2 | σ3 | σ4 | conditionnement |
|---|---|---|---|---|---|
| actuel, fp32 | 2.12e+00 | 2.30e-01 | 8.34e-03 | 2.34e-04 | 9.0e+03 |
| actuel, **bf16** | 2.12e+00 | 2.31e-01 | 1.18e-02 | 8.11e-03 | 2.6e+02 |
| référence ×1000 | 1.08e+01 | 9.62e+00 | 9.00e+00 | 9.00e+00 | 1.2e+00 |

L'entraînement tourne en **AMP bf16** (`use_amp: true`, `amp_dtype: bf16`). En
bf16 les 3ᵉ et 4ᵉ directions *remontent* (2.3e-4 → 8.1e-3) : le signal a été
remplacé par du bruit de quantification. Deux des quatre directions du temps
sont sous le plancher numérique. Avec ×1000, les quatre valeurs sont
équivalentes — base quasi orthogonale, insensible à la précision.

## 3. Ce que le modèle entraîné en fait (`weight_share.py`)

Part de variance apportée à la première couche, **EMA de production du
vectorisé** (`outputs/mmfm/vectorized/weights/model_final.pth`, 24 999 iters) :

| bloc d'entrée | part de variance | ‖W‖ moyen par canal |
|---|---|---|
| `z_t` | 53.19 % | 1.242e-03 |
| `z_src` | 46.80 % | 1.218e-03 |
| **temps** | **0.0000 %** (1.2e-7) | **1.658e-03** |
| contraste | 0.0039 % | 2.917e-03 |

Le poids **par canal** sur le temps est le plus élevé des quatre blocs : le
réseau a bien essayé de remonter le signal, et n'a pas pu. Le temps n'est
injecté qu'à la concaténation d'entrée — pas d'AdaLN, pas de réinjection par
bloc — donc ce tableau est toute l'histoire.

S'y ajoute un second déséquilibre : `latent_mean`/`latent_scale` n'existent que
dans `inr.yaml`. Le vectorisé et l'UNet tournent avec `scale = 1`, donc un
latent d'écart-type **21.98** contre un embedding de temps à **0.027**. C'est
l'image en miroir du bug INR corrigé en phase B (là, le latent était trop petit
devant le conditionnement) ; ce côté-ci n'avait jamais été regardé.

## 4. Conséquence sur la vitesse (`t_sens.py`, `unet_t.py`)

`v(z, z_src, t, y)` à `z` fixe, seul `t` varie :

| | variation de `v` selon `t` | cos(v(t=0), v(t=1)) | variation selon le SUJET |
|---|---|---|---|
| Vectorisé T1W | **0.00 %** | **1.000000** | 10.96 % |
| Vectorisé T2W | **0.00 %** | **1.000000** | 14.25 % |
| Vectorisé T2FLAIR | **0.00 %** | **1.000000** | 29.80 % |
| UNet T1W | 1.71 % | 0.999443 | 14.65 % |
| UNet T2W | 2.59 % | 0.996959 | 22.30 % |

Le vectorisé de production est **littéralement aveugle au temps** : sa vitesse
est numériquement identique à `t = 0` et à `t = 1`. L'UNet s'en tire un peu
mieux — MONAI fait passer l'embedding par un MLP appris et l'**additionne** dans
chaque resblock, au lieu de le noyer dans une concaténation à 258 432 canaux —
mais reste ~9× plus sensible au sujet qu'au champ visé.

## 5. Conséquence sur la trajectoire (`traj_bend.py`, `unet_bend.py`)

Une vitesse constante en `t` intègre une **droite**. Déviation à la corde
0.1T→7T aux trois ancres intermédiaires (1.5T, 3T, 5T), en fraction de la
longueur de corde — intégration Euler 200 pas depuis des latents réels :

| | 1.5T | 3T | 5T |
|---|---|---|---|
| Vectorisé T1W | 0.005 | 0.007 | 0.006 |
| *données T1W* | *0.896* | *0.748* | *0.609* |
| Vectorisé T2W | 0.017 | 0.023 | 0.017 |
| *données T2W* | *0.598* | *0.450* | *0.371* |
| Vectorisé T2FLAIR | 0.013 | 0.018 | 0.014 |
| *données T2FLAIR* | *0.533* | *0.586* | *0.553* |
| UNet T2W (40 pas) | 0.015 | 0.020 | 0.015 |
| *données T2W* | *0.582* | *0.451* | *0.366* |

**Le modèle produit 1 à 5 % de la courbure que les données exigent.** Les champs
intermédiaires sont donc obtenus comme des points d'un segment de droite, à
37–90 % d'une corde de leur position réelle : c'est exactement l'apparence
moyennée entre 0.1T et 7T — contraste faux **et** lissé.

**Réserve.** La courbure « données » est mesurée sur les **moyennes de classe**
(aucun appariement n'existe : 0 sujet sur 1056 à deux champs). Elle situe
correctement les marginales, mais n'est pas une trajectoire individuelle
couplée. La courbure « modèle » est, elle, mesurée par sujet puis moyennée.

## 6. Ce que cela change pour `adjacent_only` (2026-08-26)

Le test `adjacent_only: true` avait dégradé surtout le saut long 0.1T↔7T
(+0.0153, 2 victoires sur 6), et j'en avais conclu que le mécanisme
« cibles contradictoires » était réfuté. **Cette lecture était fausse.**
`adjacent_only` oblige le saut long à s'obtenir en intégrant un chemin *courbé*
à travers les intermédiaires — précisément ce qu'un modèle aveugle au temps ne
peut pas faire ; entraîner la paire directement lui donne une cible en ligne
droite qu'il *peut* représenter. L'échec d'`adjacent_only` est donc un **second
symptôme du même défaut**, pas une réfutation.

Le fait mesuré — marginales non colinéaires — tient. Le remède était bloqué par
un défaut situé en amont de lui. Et à vitesse constante, « paires adjacentes »
et « toutes paires » convergent vers la même droite : l'écart de 0.0018 mesuré
alors est cohérent avec du bruit de run.

## 7. Autres écarts aux références, non testés

Par ordre décroissant d'appui externe.

1. **Loss.** Les **9** scripts d'entraînement de Genentech/MMFM utilisent
   `torch.mean((vt - ut) ** 2)`. Nous : `F.l1_loss` (`mmfm_core.py:753`).
   Dette n°7, désormais corroborée de l'extérieur.
2. **Guidance sans classifieur.** Absente de notre code (aucune occurrence).
   MMFM : `p_unconditional = 0.2` à l'entraînement, `guidance` balayée et
   *sélectionnée* à l'évaluation (`trajectory.py`, `MMFMModelGuidanceWrapper`).
   NVIDIA : `cfg_guidance_scale` de premier plan à l'inférence.
3. **Interpolant.** La référence enchaîne des transports OT
   (`ot.da.EMDTransport`) pour former **une** trajectoire couplée traversant
   toutes les marginales, puis ajuste **une spline cubique** ; la cible est sa
   dérivée — continue en `t`. Nous tirons des droites indépendantes par paire :
   cible constante par morceaux, **discontinue aux ancres**. En interpolation
   *linéaire* les deux coïncident segment par segment ; l'ingrédient réellement
   non testé est donc le caractère **cubique** de l'interpolant.
4. **`identity_prob: 0.1`.** Apprend `v = 0` à un `t` tiré uniformément sur
   `[0,1]`, pour un latent réel. Aucun équivalent dans l'une ou l'autre
   référence. Effet non mesuré.
5. **Normalisation d'intensité.** NVIDIA dénormalise sur une plage **globale
   fixe** (`a_min, a_max = 0, 1000` pour l'IRM, `utils_infer.py:185-190`). Nous
   utilisons des percentiles par volume ou par champ.

## 8. Correctif

Une ligne dans `mmfm_vectorized.py` (vectorisé + INR) et une dans
`arch_unet.py` : multiplier `t` par 1000 avant l'embedding. **Mais le correctif
invalide les trois checkpoints** — le sens de l'entrée temporelle change — et
impose de réentraîner (~25 000 itérations chacun). Non appliqué : décision à
prendre.

## Reproduction

```
python results/mmfm/audit_20260827_refcompare/measure_temb.py     # §2
python results/mmfm/audit_20260827_refcompare/measure_temb2.py    # §2 spectre
python results/mmfm/audit_20260827_refcompare/weight_share.py     # §3
python results/mmfm/audit_20260827_refcompare/t_sens.py           # §4 vectorisé
python results/mmfm/audit_20260827_refcompare/unet_t.py           # §4 UNet
python results/mmfm/audit_20260827_refcompare/traj_bend.py        # §5 vectorisé
python results/mmfm/audit_20260827_refcompare/unet_bend.py        # §5 UNet
```

Références clonées à titre de lecture seule, non versionnées :
`Genentech/MMFM` @ HEAD, `NVIDIA-Medtech/NV-Generate-CTMR` @ HEAD (2026-08-27).
