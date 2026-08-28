# Correctifs du flow : R1, R-best, et le plafond de représentation

**Date** : 2026-08-27 → 2026-08-28
**Verdict** : **la géométrie du flow est réparée, le score ne bouge pas de façon
significative** — et la décomposition représentation/flow réoriente la suite.

---

## 1. Ce qui a été entraîné

| run | changements par rapport à la production | coût |
|---|---|---|
| **R1** | `time_scale: 1000` seul | 2,0 h |
| **R-best** | `time_scale: 1000` + `time_cond: film` + `adjacent_only: true` | 2,0 h |

Combinaison désignée par le harnais synthétique
(`../synthetic_20260827/manifest.md`), qui l'avait sélectionnée en runs de
2 minutes au lieu de 2 heures.

## 2. Géométrie du flow — le défaut est corrigé

Mesure sur latents réels (`src/cfm/measure_flow_geometry.py`) :

| | production | R1 | **R-best** |
|---|---|---|---|
| cos(v(0), v(1)) T1W | 1.000000 | 1.000000 | **−0.240** |
| cos T2W | 1.000000 | 1.000000 | **0.802** |
| cos T2FLAIR | 1.000000 | 1.000000 | **0.031** |
| courbure / exigée, T1W | 0.006 | 0.003 | **0.498** |
| T2W | 0.051 | 0.054 | **0.965** |
| T2FLAIR | 0.031 | 0.027 | **0.864** |

**R1 est un échec net** : `time_scale` seul ne déplace rien (part de variance du
temps à la 1ʳᵉ couche 0.00000 % → 0.00037 %, cos inchangé à 1.000000).
**R-best restaure la courbure d'un facteur 100 à 500.**

## 3. Score Task 3 — aucun gain significatif

20 paires × 3 sujets × 3 contrastes, évaluateur officiel.

| | T1W | T2W | T2FLAIR | moyenne | SSIM | LPIPS |
|---|---|---|---|---|---|---|
| production | 0.4353 | 0.3376 | 0.3654 | **0.3794** | 0.8975 | 0.0941 |
| R1 | 0.4205 | 0.3347 | 0.3645 | **0.3733** | +0.0065 | +0.0055 |
| R-best | 0.4441 | **0.3181** | **0.3589** | **0.3737** | +0.0072 | +0.0049 |

| test apparié sur 60 paires | victoires | signes | Wilcoxon |
|---|---|---|---|
| R1 contre production | 35/60 | p = 0.245 | p = 0.124 |
| R-best contre production | 36/60 | p = 0.155 | p = 0.082 |

**Ni l'un ni l'autre n'est significatif.** SSIM monte de +0.007, LPIPS se dégrade
de +0.005 : les deux métriques ne vont pas dans le même sens.

**Le harnais l'avait annoncé** : en dimension 4096, son nRMSE restait plat
(0.0114–0.0119) sur des variantes dont la courbure variait d'un facteur 53.

## 4. Le seul signal net : courbure restaurée et gain vont ensemble

| contraste | courbure atteinte | écart de nRMSE | victoires |
|---|---|---|---|
| T1W | 0.498 | **+0.0088** (pire) | 9/20 |
| T2FLAIR | 0.864 | −0.0065 | 12/20 |
| T2W | 0.965 | **−0.0195** (meilleur) | 15/20 |

**Relation monotone sur les trois contrastes.** Là où la géométrie est restaurée,
le score s'améliore ; là où elle ne l'est qu'à moitié (T1W), il se dégrade. C'est
la prédiction mécanistique confirmée — et cela suggère que T1W a besoin de plus
que ces correctifs, pas qu'ils sont inutiles.

## 5. Auto-reconstruction : la décomposition représentation / flow

La vérité terrain passée dans l'encodeur/décodeur de chaque architecture, **sans
aucun transport** (paires identité, `dt = 0`), par le chemin de code strictement
identique aux prédictions.

| | nRMSE | SSIM |
|---|---|---|
| **MedVAE** (vectorisé et UNet) | **0.1048** | **0.9517** |
| **INR** (ajustement 20 pas) | 0.3395 | 0.8334 |

**Pour MedVAE c'est un vrai plafond** — meilleur que les prédictions sur les deux
métriques. Décomposition : score 0.3737, plafond 0.1048, donc **≈ 72 % de
l'erreur du vectorisé et de l'UNet vient du flow**, 28 % de la représentation.
Il y a de la marge, et elle est du côté du flow.

**Pour l'INR ce n'est PAS un plafond**, et il faut le dire : son
auto-reconstruction est **moins bonne en SSIM que sa propre prédiction de bout en
bout** sur les trois contrastes (0.845 / 0.793 / 0.863 contre 0.861 / 0.858 /
0.871). Le flow atterrit donc sur des latents qui décodent mieux que l'ajustement
de la vraie cible. La quantité mesurée n'est pas une borne : c'est la **qualité
de l'ajustement INR à 20 pas** (`inner_steps_eval: 20`).

Ce qu'elle établit tout de même, et c'est frappant : **l'ajustement INR ne retient
que 20.6 % du contraste de la cible** (écart-type sur le cerveau, T2W@3T), contre
73.7 % pour sa propre prédiction et **87.3 % pour MedVAE**. Le handicap de l'INR
est bien dans son chemin de représentation — mais `inner_steps_eval` est un
levier non testé, et il faut le tester avant de conclure à une limite de capacité.

## 6. Fichiers

| chemin | contenu |
|---|---|
| `task3_vec_r1_time_{T1W,T2W,T2FLAIR}.csv` | R1, 20 paires × 3 sujets |
| `task3_vec_rbest_{T1W,T2W,T2FLAIR}.csv` | R-best |
| `../ceiling_20260827/ceiling_{vectorized,inr}_*.csv` | auto-reconstruction |
| `../qualitative_20260828/*.png` | figures (voir son manifeste) |
| `../../../configs/mmfm/vectorized_r1_time.yaml`, `vectorized_rbest.yaml` | configs |

**Le plafond UNet a ÉCHOUÉ** : `remap_monai_attention_keys` ne traite pas la
renomination `time_emb_proj` → `emb_proj` entre versions de MONAI, donc le
checkpoint de référence ne se charge pas avec la config actuelle. L'UNet partage
MedVAE avec le vectorisé, donc son plafond devrait être identique — mais **c'est
une déduction, pas une mesure**, et le contrôle d'équité prévu n'a pas pu tourner.

## 7. Reproduction

```
PYTHONPATH=src python src/cfm/measure_flow_geometry.py \
    --config configs/mmfm/vectorized_rbest.yaml --tag R-best
bash scripts/run_task3_eval.sh configs/mmfm/vectorized_rbest.yaml \
    mmfm3d_vectorized vec_rbest results/mmfm/staircase_20260827
PYTHONPATH=src python src/cfm/infer_mmfm_unified.py --config configs/mmfm/vectorized.yaml \
    --checkpoint outputs/mmfm/vectorized/weights/model_final.pth \
    --output_dir outputs/mmfm/ceiling_vectorized/predictions \
    --pairs 0.1T_to_0.1T,1.5T_to_1.5T,3T_to_3T,5T_to_5T,7T_to_7T \
    --modalities T1W T2W T2FLAIR --field_norm_stats configs/mmfm/field_norm_stats.json
PYTHONPATH=src python src/cfm/eval_representation_ceiling.py \
    --pred-root outputs/mmfm/ceiling_vectorized/predictions/task3 --name vectorized \
    --outdir results/mmfm/ceiling_20260827
```
