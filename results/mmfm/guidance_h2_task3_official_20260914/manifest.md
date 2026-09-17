# Guidance conditionnelle (CFG-style) sur le flow mmfm — expérience H2, CLÔTURE NÉGATIVE

**Date** : 2026-09-14/15
**Motivation** : comparaison avec NVIDIA/NV-Generate-CTMR (rectified flow via MONAI),
qui utilise un dropout de conditionnement à l'entraînement (10 %) et une guidance
scale=15 à l'inférence — mécanisme absent de notre flow. Voir le plan
`ok-donc-si-on-quizzical-sifakis.md`.

**Mécanisme testé** : `model.cond_dropout_prob: 0.1` réserve un id de classe NUL
dans la table d'embedding du contraste (voir `arch_vector.py::build_vector_mmfm`) ;
`mmfm_core.py::train()` y bascule `y_tgt` (label de contraste, PAS celui du
régularisateur de cycle) avec cette probabilité. `euler_integrate` accepte un
`guidance_scale` : `v = v_uncond + scale·(v_cond - v_uncond)`. Code
rétrocompatible : `cond_dropout_prob=0.0` et `guidance_scale=1.0` (défauts)
reproduisent exactement le comportement historique.

**Config** : `configs/mmfm/vectorized_trajectory_cfg.yaml` — copie EXACTE de
`configs/mmfm/vectorized_trajectory.yaml` (production/R-best), une seule clé
change (`model.cond_dropout_prob: 0.1`). Checkpoint :
`outputs/mmfm/vectorized_trajectory_cfg/weights/model_final.pth` (25000 pas,
3h50, ralenti par un partage GPU avec une autre tâche locale).

## Balayage exploratoire (sous-ensemble réduit, NON officiel)

3 paires de champs (0.1T↔7T, 1.5T→5T) × 1 sujet × 3 contrastes, géométrie `cc`,
évaluées avec `scripts/eval_pairs_adhoc.py` (même calcul que l'évaluateur
officiel, sans la porte de complétude — voir ce script pour la raison). Sur
CE sous-ensemble, un optimum apparaissait à `guidance_scale≈2.0` (nRMSE moyen
0.2908 contre 0.2948 à `scale=1.0`, −1.4 %), avec dégradation nette au-delà de
4.0. **Ce signal ne survit PAS à l'échelle officielle (voir ci-dessous) — leçon
méthodologique : un n=3 sous-alimenté a suggéré un optimum qui s'est inversé
dès qu'on l'a mesuré correctement.**

## Évaluation officielle complète (20 paires × 3 sujets × 3 contrastes, 8win, région slab)

Comparaison appariée (60 paires) contre R-best/production
(`results/mmfm/geometry_20260907/task3_rbest_8win_slab_*.csv`, nRMSE 0.3516 —
la référence géométrie-vérifiée, PAS l'ancien chiffre 0.3794 périmé).

| comparaison | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **gs=1.0** (dropout seul, sans guidance) vs production | indiscernable (30/60, signe p=1.0, Wilcoxon p=0.84) | **significativement pire** (18/60, p=0.0027) | **significativement pire** (7/60, p<1e-8) |
| **gs=2.0** (candidat du balayage réduit) vs production | indiscernable (26/60, p=0.37/0.49) | **significativement pire** (16/60, p=0.0004) | **significativement pire** (5/60, p<1e-10) |
| **gs=2.0 vs gs=1.0** (même modèle, guidance seule) | **significativement PIRE** (23/60, Wilcoxon p=0.037) | **significativement pire** (11/60, p<1e-6) | **significativement pire** (5/60, p<1e-10) |

Chiffres moyens (20 paires × 3 sujets × 3 contrastes, officiel) :

| variante | T1W nRMSE | T2W nRMSE | T2FLAIR nRMSE | moyenne nRMSE |
|---|---|---|---|---|
| **production (R-best)** | 0.4096 | 0.2988 | 0.3464 | **0.3516** |
| **H2 gs=1.0** (dropout, sans guidance) | 0.4129 | 0.2825 | 0.3148 | 0.3367 |
| **H2 gs=2.0** (dropout + guidance) | 0.4663 | 0.3078 | 0.2956 | 0.3566 |

## Verdict

**NÉGATIF, sur les trois axes.**

1. `cond_dropout_prob=0.1` seul (sans guidance) ne change pas le nRMSE
   (statistiquement indiscernable de la production) mais dégrade
   significativement SSIM et LPIPS.
2. La guidance (`scale=2.0`) n'améliore RIEN à l'échelle officielle : elle
   dégrade le nRMSE lui-même (contrairement au signal du balayage réduit) en
   plus d'aggraver encore SSIM/LPIPS.
3. **Le balayage exploratoire sur sous-ensemble réduit (n=3 paires) a
   activement mal orienté** : son optimum apparent à `scale=2.0` s'inverse en
   dégradation nette dès qu'on mesure sur les 60 paires officielles. Un
   sous-ensemble de cette taille ne doit plus servir à choisir une valeur
   d'hyperparamètre sans un test apparié sur l'échelle complète.

**Clôture de la piste H2.** Le mécanisme (dropout + guidance) est conservé
dans le code (rétrocompatible, défauts neutres), réutilisable pour toute
reprise future, mais aucune configuration testée ne bat la production.
H1 (refonte pairwise, champ en conditionnement plutôt qu'en temps) reste
documentée et non engagée — ce résultat négatif ne se prononce pas sur elle,
mais ne fournit pas non plus d'argument pour l'ouvrir sans un nouveau feu vert
explicite.

## Réserves

- **Un seul entraînement par variante** (pas de répétition avec graines
  différentes) : le run-to-run variance de ce pipeline stochastique
  (`batch_size=1`) n'est pas quantifié ici : l'écart gs=1.0 vs production
  pourrait en partie refléter cela plutôt que le seul effet du dropout.
- Entraînement ralenti (~3.85 it/s en moyenne, contre ~5.9 pour un run
  similaire non contendu) par un partage GPU avec une tâche locale
  indépendante (`FlashVSR`) — sans effet attendu sur la qualité du résultat,
  seulement sur sa durée.
- Guidance implémentée uniquement pour l'architecture vectorisée
  (`arch_vector.py`/`mmfm_vectorized.py`) ; UNet et INR non touchés.
