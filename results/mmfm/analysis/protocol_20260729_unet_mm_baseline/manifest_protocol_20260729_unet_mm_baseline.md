# Manifest — Porte 0 : baseline fraîche de la lignée UNet multi-marginal (2026-07-29)

## Objectif
Établir une référence fiable pour `mmfm3d_multimarginal` (UNet MONAI, conditionnement additif,
sans field_norm_stats/cycle/edge) avant toute modification de code, conformément au plan
`temporal-plotting-engelbart.md`. Aucune éval fiable n'existait auparavant pour cette lignée (les
CSV `results/task*_mmfm_unet_v2.csv` de juin/juillet évaluent une méthode différente — 15 classes
plates, pas le design multi-marginal à temps continu — et les runs multi-marginaux `run1`/
`run2_local`/`run3c` n'avaient jamais été scorés).

## Modèles triés

1. `mmfm3d_multimarginal_medvae_run1`
   - Config: `configs/mmfm3d_multimarginal_medvae_run1.yaml`
   - Checkpoint: `outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run1/weights/model_final.pth`
2. `mmfm3d_multimarginal_medvae_run2_local`
   - Config: `configs/mmfm3d_multimarginal_medvae_run2_local.yaml`
   - Checkpoint: `outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run2_local/weights/model_final.pth`
3. `mmfm3d_multimarginal_medvae_run3c`
   - Config: `configs/mmfm3d_multimarginal_medvae_run3c.yaml`
   - Checkpoint: `outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run3c/weights/model_final.pth`

## Triage (4 paires `{0.1T,1.5T,3T,5T}→7T`, 3 sujets, T1W)

Commande : `src/cfm/infer_mmfm_unified.py` (batch, `--pairs`) puis `src/evaluation/evaluate.py --task task1`.

Pred dirs : `outputs/predictions/mmfm_unet_mm_{run}_triage/task3/T1W`
CSV : `results/mmfm/analysis/protocol_20260729_unet_mm_baseline/triage_{run}.csv`

### Ratio d'intensité moyenne pred/GT (masque cerveau, calcul direct numpy)
| Run | 0.1T→7T | 1.5T→7T | 3T→7T | 5T→7T | Moyenne |
|---|---|---|---|---|---|
| run1 | 4.256 | 4.896 | 4.759 | 4.831 | **4.69** |
| run2_local | 1.387 | 2.311 | 3.791 | 4.595 | **3.02** |
| run3c | 2.912 | 3.826 | 4.652 | 4.894 | **4.07** |

Confirme le bug documenté dans l'en-tête de `run3c.yaml` ("amplifie l'intensité au lieu de
l'atténuer") — présent et sévère sur les **3** runs, pas spécifique à un run. run2_local est le
moins affecté.

### nRMSE/SSIM/LPIPS (moyenne sur les 4 paires)
| Run | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| run1 | 3.511 | 0.7925 | 0.1607 |
| **run2_local** | **2.456** | 0.7928 | 0.1707 |
| run3c | 3.070 | 0.7916 | 0.1634 |

**Verdict** : run2_local retenu (nRMSE nettement meilleur, SSIM quasi-identique, LPIPS légèrement
moins bon mais non déterminant).

## Éval complète — run2_local (20 paires × 3 sujets, T1W)

Commande : `infer_mmfm_unified.py` batch (toutes paires) → `evaluate.py --task task3`.
Pred dir : `outputs/predictions/mmfm_unet_mm_run2_local_full/task3/T1W`
CSV : `results/mmfm/analysis/protocol_20260729_unet_mm_baseline/task3_run2_local_T1W.csv`

### Gate de complétude
60/60 prédictions présentes (PASS).

### nRMSE moyen par champ CIBLE
| Cible | nRMSE moyen | min | max |
|---|---|---|---|
| →0.1T | 0.487 | 0.455 | 0.515 |
| →1.5T | 0.625 | 0.394 | 0.752 |
| →3T | 0.454 | 0.396 | 0.500 |
| **→5T** | **1.881** | 1.080 | 2.515 |
| **→7T** | **2.455** | 1.506 | 3.410 |

**nRMSE global (20 paires) : 1.180** — à comparer à la référence vectorisée régularisée v2
(`results/task3_mmfm_v2_regularized_v2_T1W.csv`, nRMSE moyen ≈0.53).

## Conclusion et prochaine étape

Signature très nette : nRMSE raisonnable (0.45-0.63, comparable/compétitif avec le vectorisé)
quand la cible est 0.1T/1.5T/3T, mais **4-5x pire** (1.9-2.5) quand la cible est 5T/7T — exactement
la signature attendue d'un problème de normalisation par volume (dynamique native étroite à 5T/7T,
effacée par la normalisation percentile-par-volume au lieu d'une normalisation fixe par champ).
Confirme directement l'hypothèse de la Phase 1 du plan (port de `field_norm_stats`) comme prochaine
étape à fort ROI, avant d'investir dans le travail architectural FiLM (Phase 3).

Référence croisée : `field_norm_stats_1mm.json` déjà calculé (quasi-identique à la version 2mm
existante), prêt à être branché en Phase 1.

## Phase 1 — field_norm_stats + fix de dénormalisation (run4_fieldnorm)

Config : `configs/mmfm3d_multimarginal_medvae_run4_fieldnorm.yaml` — reprise (`--resume_weights_only`)
depuis `run2_local`, `field_norm_stats_1mm.json` branché (entraînement + cache latent spatial dédié
`outputs/latent_cache_unet_spatial/`), 15000 itérations (~60min, débit ~4.2 it/s — net gain vs
~0.55 it/s historique, grâce au cache pré-croppé qui élimine le crop CPU à la volée).

### Bug additionnel découvert et corrigé en cours de route

Premier essai (field_norm_stats côté entrée uniquement) : le ratio d'intensité pred/GT a *empiré*
sur 3 des 4 paires triage (nRMSE moyen 3.65 vs 2.46 baseline). Diagnostic : `infer_mmfm_unified.py`
normalise la source en l'étirant vers [-1,1] via `field_norm_stats`, mais ne recompressait jamais
la sortie du modèle vers l'échelle native étroite du champ CIBLE avant sauvegarde — le modèle
apprend à prédire dans le domaine étiré (les latents cibles du cache sont aussi étirés via
`field_norm_stats`), donc sans l'étape inverse, la prédiction reste ~`1/(hi-lo)` fois trop
"amplifiée" par rapport au GT natif — vérifié empiriquement (ratio observé ≈2.96 pour T1W/7T,
prédiction théorique `1/(hi-lo)`≈3.09). Fix : ajout de `denormalize_from_01(pred, tgt_lo, tgt_hi)`
dans `process_volume_unified` (miroir exact de ce que `train_mmfm_3d.py::infer()` fait déjà côté
vectorisé) — `src/cfm/infer_mmfm_unified.py`.

### Résultats après fix (triage, 4 paires vers 7T)

| Paire | run2_local (baseline) | run4_fieldnorm (fix complet) |
|---|---|---|
| 0.1T→7T | nRMSE 1.506 / SSIM 0.770 / LPIPS 0.214 | nRMSE 0.666 / SSIM 0.878 / LPIPS 0.145 |
| 1.5T→7T | nRMSE 2.044 / SSIM 0.805 / LPIPS 0.175 | nRMSE 0.985 / SSIM 0.874 / LPIPS 0.123 |
| 3T→7T | nRMSE 2.862 / SSIM 0.804 / LPIPS 0.150 | nRMSE 0.879 / SSIM 0.894 / LPIPS 0.103 |
| 5T→7T | nRMSE 3.410 / SSIM 0.793 / LPIPS 0.144 | nRMSE 0.533 / SSIM 0.916 / LPIPS 0.082 |
| **Moyenne** | **2.456** | **0.766** (>3x meilleur) |

CSV : `triage_run4_fieldnorm_fixed.csv`. Amélioration cohérente sur les 3 métriques et les 4 paires.
Le 0.1T→7T (nRMSE 0.666) dépasse même le vectorisé régularisé v2 sur cette paire précise (nRMSE
0.758 dans `results/task3_mmfm_v2_regularized_v2_T1W.csv`). La loss d'entraînement décroissait
encore nettement à la fin des 15000 itérations (pas de plateau) — marge probable avec plus
d'entraînement.

**Statut Porte 1 : validée.** Prochaine étape : éval complète (20 paires) pour confirmer sur
l'ensemble, puis décision (prolonger l'entraînement vs passer à la Phase 2 cycle/edge).

### Éval complète (20 paires × 3 sujets, T1W) — run4_fieldnorm

Pred dir : `outputs/predictions/mmfm_unet_mm_run4_fieldnorm_full/task3/T1W`
CSV : `results/mmfm/analysis/protocol_20260729_unet_mm_baseline/task3_run4_fieldnorm_T1W.csv`
Gate de complétude : 60/60 (PASS).

nRMSE moyen par champ CIBLE :
| Cible | nRMSE moyen | min | max |
|---|---|---|---|
| →0.1T | 0.275 | 0.186 | 0.360 |
| →1.5T | 0.338 | 0.197 | 0.459 |
| →3T | 0.320 | 0.229 | 0.400 |
| →5T | 0.493 | 0.400 | 0.662 |
| →7T | 0.766 | 0.533 | 0.985 |

**nRMSE global (20 paires) : 0.438 | SSIM : 0.906 | LPIPS : 0.106**

| Modèle | nRMSE global |
|---|---|
| run2_local (baseline UNet, avant Phase 1) | 1.180 |
| **run4_fieldnorm (Phase 1 : field_norm_stats + fix dénormalisation)** | **0.438** |
| Vectorisé régularisé v2 (référence session, `task3_mmfm_v2_regularized_v2_T1W.csv`) | ≈0.530 |

Le UNet+Phase1, avec seulement les corrections de pipeline de données (pas encore de FiLM, pas de
cycle/edge-consistency), **dépasse déjà** le modèle vectorisé fortement régularisé de la première
partie de cette session. Reste un gradient de difficulté par champ cible (7T le plus dur, cohérent
avec sa dynamique native la plus étroite), mais le niveau global a changé d'ordre de grandeur
(nRMSE ÷2.7 vs le baseline UNet).

**Prochaine étape** : confirmation qualitative (figures), puis décision entre (a) prolonger
l'entraînement de run4_fieldnorm (la loss ne plafonnait pas à 15k itérations), (b) passer à la
Phase 2 (cycle/edge-consistency, réutilisables tels quels), ou (c) passer directement à la Phase 3
(FiLM) étant donné que ce backbone dépasse déjà la référence vectorisée.
