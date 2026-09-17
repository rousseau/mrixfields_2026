# H1 — refonte pairwise (t local, champ en conditionnement) — NON ÉTABLI

**Date** : 2026-09-15
**Motivation** : comparaison avec NVIDIA/NV-Generate-CTMR et torchcfm, où `t`
ne porte jamais de sémantique (interpolation pure entre 2 points fixes, toute
information supplémentaire passant par un conditionnement séparé). Notre
design historique (`marginal_mode: trajectory`) conflate `t` avec la position
du champ (`_field_to_time`) et intègre en continu à travers les 5 marginales.
Voir CHANGELOG 2026-09-14, plan `ok-donc-si-on-quizzical-sifakis.md`.

**Mécanisme** (`marginal_mode: pairwise_ot`, nouveau, rétrocompatible) :
réutilise le MÊME couplage OT chaîné que `trajectory` (sujets appariés à
travers les 5 champs, donc pas de régression vers le bug « aucun couplage »
de l'ancien `pairwise`) ; pour chaque paire (champ source, champ cible)
tirée par `_sample_step_plan`, applique un CFM standard à 2 points entre les
marginales OT-appariées, `t` restant local dans [0,1]
(`_compute_flow(..., 0.0, 1.0, ...)`, sans rescale). Conditionnement de
classe = (contraste, champ source, champ cible) via `_flat_triple`
(num_classes = 3×5×5 = 75, contre 3).

**Validé au préalable sur le harnais synthétique** à réponse connue
(`configs/mmfm/synthetic_pairwise_ot.yaml`) : nRMSE 0.053 au bon
conditionnement de paire de champs (plancher de bruit 0.05) contre 0.086 à un
conditionnement délibérément faux (ratio 1.62×) — le mécanisme fonctionne
correctement au niveau mécanistique.

**Config réelle** : `configs/mmfm/vectorized_pairwise_cond.yaml` — copie
EXACTE de `configs/mmfm/vectorized_trajectory.yaml` (production/R-best), seule
clé qui change : `train.marginal_mode: pairwise_ot`. 25000 pas, 2h32 (aucune
contention GPU), loss ~48-50 en fin d'entraînement.

## Évaluation officielle (20 paires × 3 sujets × 3 contrastes, 8win, région slab)

Comparaison appariée (60 paires) contre R-best/production
(`results/mmfm/geometry_20260907/task3_rbest_8win_slab_*.csv`, la référence
géométrie-vérifiée, nRMSE 0.3516).

| métrique | H1 | production | H1 meilleur sur | signe p | Wilcoxon p |
|---|---|---|---|---|---|
| **nRMSE** | 0.3209 | 0.3516 | 36/60 | 0.155 (NS) | **0.057 (à la limite, NON significatif)** |
| **SSIM** | 0.8272 | 0.8499 | 16/60 | **0.0004** | **0.0002** |
| **LPIPS** | 0.2119 | 0.1880 | 6/60 | **<1e-10** | **<1e-9** |

Par contraste (nRMSE, H1 meilleur sur X/20) :

| contraste | production | H1 | H1 meilleur sur |
|---|---|---|---|
| T1W | 0.4096 | 0.3850 | 11/20 |
| T2W | 0.2988 | 0.2753 | 12/20 |
| T2FLAIR | 0.3464 | 0.3024 | 13/20 |

## Verdict

**NON ÉTABLI — l'amélioration apparente du nRMSE agrégé (−8.7 %) ne résiste
PAS au test apparié** (Wilcoxon p=0.057, juste au-dessus du seuil ; 60 % des
paires gagnantes, pas assez pour conclure). Dans les 3 contrastes pris
séparément, H1 gagne à peine plus de la moitié des paires (11-13/20) : la
direction est cohérente mais l'effet est faible et diffus, pas concentré.

**Et SSIM/LPIPS se dégradent significativement**, exactement comme pour H2
(`cond_dropout` seul) — un troisième résultat du même type (voir CHANGELOG
2026-09-14/15 §H2) : un changement qui touche le conditionnement/la
représentation de classe améliore ou n'affecte pas le nRMSE tout en
dégradant nettement la qualité perceptuelle.

**Ce résultat ne tranche PAS l'hypothèse H1 elle-même** (« t ne devrait pas
porter de sémantique ») — il montre que CETTE implémentation particulière
(num_classes 75, spline abandonnée pour un CFM à 2 points par paire) ne bat
pas la production sur l'ensemble des métriques. Deux causes plausibles, non
départagées :
1. La table de classe à 75 entrées (25× plus grande que les 3 du design
   `trajectory`) pourrait diluer ce que FiLM apprend par entrée, chacune vue
   moins souvent pendant l'entraînement à budget d'itérations égal (25000).
2. Le CFM à 2 points par paire perd la contrainte de cohérence globale que la
   spline cubique imposait à travers les 5 marginales — chaque transition est
   apprise indépendamment, ce qui pourrait favoriser le nRMSE brut au prix de
   la régularité perceptuelle.

## Clôture de la piste H1 (pour l'instant)

Le mécanisme (`marginal_mode: pairwise_ot`) est conservé dans le code
(rétrocompatible, validé sur le harnais synthétique, réutilisable). Aucune
configuration testée ne bat la production sur l'ensemble des 3 métriques.
Reprendre cette piste supposerait soit un budget d'itérations plus grand
(pour compenser la table de classe élargie), soit une régularisation de
cohérence entre paires (pour retrouver ce que la spline offrait), ni l'un ni
l'autre testés ici.

## Réserves

- Un seul entraînement (pas de répétition, variance run-to-run non
  quantifiée) — comme pour H2.
- `class_embed_dim=128` inchangé malgré le passage de num_classes=3 à 75 :
  jamais testé à une dimension plus grande.
- `num_targets_per_step=2` (vs 1 pour `trajectory`, où il est sans objet) :
  jamais balayé, choisi par analogie de coût de calcul.
