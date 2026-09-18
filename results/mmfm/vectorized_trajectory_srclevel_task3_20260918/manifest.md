# Conditionnement du flow par le niveau observé de la source (`level_cond`)

**Date** : 2026-09-18
**Verdict** : NÉGATIF — le modèle apprend à utiliser le signal, sans que cela
déplace le score agrégé.

## Motivation

Un témoin trivial (`eval_quantitative_3arch.py --mode scaled_identity`,
recopier la source rééchelonnée par un scalaire lu depuis son propre niveau
observé) bat les trois architectures entraînées :
`results/mmfm/region_20260904/scaled_identity_srclevel_*.csv`,
nRMSE moyen **0.2561**, SSIM **0.8707** — contre 0.3549/0.8098 pour la
production vectorisée sous le même protocole (`cc`). Cause identifiée :
`normalize_volume`/`normalize_volume_fixed` divise chaque volume par son propre
percentile avant que le réseau ne le voie — l'information est détruite avant
d'atteindre le flow. `corr(niveau du foreground, gain oracle) = -0.76 à -0.88`.

## Implémentation

- `common.io.foreground_level(vol, threshold=0.02)` : formule partagée avec le
  témoin (`eval_quantitative_3arch.py::_fg_level` délègue maintenant à cette
  fonction) — même quantité mesurée et injectée, jamais deux variantes qui
  portent le même nom.
- Niveau calculé sur le volume résamplé mais **non normalisé** (avant
  `normalize_volume_fixed`), un scalaire par volume source. Ajouté au cache
  de production (`medvae_finetune_c4d1e200`, 1939 échantillons, in-place,
  sauvegarde `.bak`, `cache_id` inchangé — voir
  `src/cfm/augment_cache_index_with_src_level.py`).
- Injection : rejoint le vecteur `cond` déjà partagé par le temps et la classe
  (`time_cond: film`), via un `nn.Linear(1, 32)` appris — même mécanisme FiLM
  déjà mesuré comme facteur dominant pour le temps, pas de canal séparé.
  `level_cond=False` (défaut) = zéro paramètre ajouté, comportement historique
  bit-à-bit inchangé (vérifié).
- Config : `configs/mmfm/vectorized_trajectory_srclevel.yaml`, copie stricte
  de la production (`vectorized_trajectory.yaml`) + `level_cond: true`,
  `level_embed_dim: 32`, `level_mean: 0.333942`, `level_scale: 0.109826`
  (mesurés sur les 1939 échantillons du cache : 0.1T=0.4895 → 7T=0.2073,
  décroissant avec le champ).
- Entraînement : 25 000 itérations, identique à la production sur tout le
  reste (même cache, même couplage OT chaîné, même loss, mêmes hyperparamètres)
  — 122.7 min GPU (mesuré ; le smoke test à 8 itérations avait sous-estimé le
  temps réel d'un facteur ~6, l'échantillon était trop court pour représenter
  le régime stable).

## Résultat (protocole officiel, géométrie `cc`, 20 paires × 3 sujets × 3 contrastes)

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **avec `level_cond`** | 0.3547 | 0.8101 | 0.1928 |
| production (`cc`, identique sinon) | 0.3549 | 0.8098 | 0.1928 |
| Δ | -0.0003 | +0.0002 | -0.0000 |
| témoin `scaled_identity` (srclevel) | 0.2561 | 0.8707 | — |

**Comparaison appariée, 60 cellules (20 paires × 3 contrastes)** : nRMSE
30/60 victoires (p=1.00, test des signes ; p=0.889 Wilcoxon), SSIM 28/60
(p=0.699 ; p=0.522), LPIPS 31/60 (p=0.897 ; p=0.935). **Aucune métrique n'est
distinguable de la production — c'est un pile ou face.**

Décomposition par contraste (nRMSE) : T1W -0.0018, T2W +0.0017, T2FLAIR
-0.0007 — aucun contraste ne porte un effet.

## Diagnostic : signal appris, mais inutile au score

Contrairement au bug historique « aveugle au temps » (`cos(v(t=0),v(t=1)) =
1.000000` avant `time_scale`), ce n'est PAS un cas de signal ignoré :

- `cond_proj.weight` (bloc résiduel, colonnes correspondant à `level_feat`) :
  magnitude moyenne 0.0016-0.0018 sur les 4 blocs, du même ordre que celle
  du temps (0.0031-0.0033) — le réseau a appris à router ce signal.
- Sensibilité mesurée : à `(z_t, z_src, t, y)` fixés, faire varier `level` de
  0.10 (niveau typique 7T) à 0.49 (niveau typique 0.1T) change le champ de
  vitesse prédit de façon non triviale (cos = 0.949 à 0.9999 selon
  l'échantillon, différence relative de norme 1.7 % à 31.7 % sur 8 tirages) —
  très loin de l'insensibilité totale mesurée pour le bug du temps.

Le réseau utilise donc l'information — mais l'utiliser ne referme pas l'écart
avec le témoin trivial. Cohérent avec deux résultats déjà établis :
[[project-medvae-lpips]] (un meilleur VAE gagne 6.9 % au plafond de
représentation, 0 % au score) et l'audit du 2026-09-04
([[project-audit-20260904-flow-constant]], mécanisme réparé, score inchangé).
**Troisième confirmation indépendante que ce pipeline n'additionne pas les
gains de ses composants** — voir [[project-plateau-huit-leviers]].

## Réserves

- `cc` uniquement (pas de confirmation `8win`, ~5h de calcul non payées —
  écart mesuré entre les deux géométries sur la région notée : jusqu'à 0.0034
  de nRMSE sur d'autres runs, plus petit que rien ici puisque Δ≈0).
- Comparaison à 60 cellules (pair × contraste, moyennées sur 3 sujets) : la
  granularité que ce pipeline d'évaluation expose, pas le niveau par sujet
  (`evaluate.py` n'écrit pas encore le détail par sujet).
- N'exclut pas qu'une capacité (`level_embed_dim`) ou un budget d'itérations
  différents changent la conclusion — non testé, coût jugé disproportionné
  face à l'absence de tout signal d'amélioration à ce budget.
