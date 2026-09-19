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

## Suite — décomposition de l'erreur restante (2026-09-18, soir)

Piste « comprendre pourquoi les gains n'itèrent pas » : décomposition
`eval_quantitative_3arch.py --mode calibration` (brut → correction réalisable
côté source → oracle d'échelle), rejouée sur les prédictions déjà écrites,
production (`cc`) ET `level_cond`, mêmes 3 sujets, mêmes 20 paires.

| | brut | réalisable (subject_scale) | oracle d'échelle | % d'énergie retirée (oracle) |
|---|---|---|---|---|
| T1W (prod / srclevel) | 0.4578 / 0.4569 | 0.5648 / 0.5635 (**pire**) | 0.2535 / 0.2537 | 76.0 % / 76.0 % |
| T2W (prod / srclevel) | 0.3043 / 0.3047 | 0.3435 / 0.3433 (**pire**) | 0.2663 / 0.2676 | 25.0 % / 24.3 % |
| T2FLAIR (prod / srclevel) | 0.3627 / 0.3619 | 0.4845 / 0.4841 (**pire**) | 0.2176 / 0.2157 | 68.1 % / 68.8 % |

**Deux découvertes, indépendantes du chiffre nul de `level_cond` ci-dessus :**

1. **`level_cond` ne change quasiment RIEN au profil de calibration** — les
   facteurs d'échelle oracle par champ cible (`a_or`) sont quasi identiques
   avant/après (ex. T1W→7T : 0.78 vs 0.78 ; T1W→3T : 1.80 vs 1.79). Le réseau
   a beau router le signal (mesuré plus haut), la sortie décodée finale n'en
   change presque pas la calibration globale.

2. **T2W est structurellement différent de T1W/T2FLAIR** — seulement 25 %
   de son erreur est retirable par une correction d'échelle globale, contre
   68-76 % pour les deux autres contrastes. Le facteur oracle par champ y
   reste proche de 1 (0.92-1.23) là où T1W/T2FLAIR vont de 0.76 à 1.8. **Toute
   poursuite de gain structurel (par opposition à un gain d'échelle) a plus de
   chances d'être mesurable sur T2W** : sur T1W/T2FLAIR, un progrès structurel
   serait noyé sous 68-76 % de bruit d'échelle non lié à l'architecture.

3. **La correction "réalisable" (`subject_scale`, information disponible
   côté source seule, sans oracle) est activement NUISIBLE sur les trois
   contrastes** (-62 %, -36 %, -97 % d'énergie *ajoutée*, pas retirée) — la
   confirmation la plus nette à ce jour que ce n'est pas un problème
   d'estimateur : voir [[project-plateau-huit-leviers]] (item de dette #4,
   fermé négatif le 2026-08-30 pour la même raison).

**Piste ouverte, non explorée dans cette session** : `estimate_intensity_recalibration.py`
(constante par PAIRE, pas par sujet, estimée sans appariement sur les 143
sujets d'entraînement par classe) donne un gain SIGNIFICATIF et propre au
vectorisé — `0.3737 → 0.3525`, p=0.014, 40/60 — retiré de la config de
production uniquement parce qu'il dégrade l'UNet et l'INR (`+0.0654`,
`+0.0941`) et que la comparaison à 3 architectures exigeait un réglage
commun. **Ce réglage n'a jamais été réadopté pour le vectorisé seul** — voir
`results/mmfm/recalib_20260829/manifest.md` §8 et l'entrée CHANGELOG.md du
2026-08-30.

## Suite — recalibration d'intensité par paire de champs, réessayée sur le checkpoint actuel (2026-09-18, nuit)

**Verdict : NÉGATIF, plus net que « sans effet » — dégradation significative.**
Le gain historique (0.3737 → 0.3525, p=0.014, entrée du 2026-08-30) avait été
mesuré sur l'ancien checkpoint `vec_rbest` (antérieur aux correctifs de
l'audit du 2026-09-04) et retiré de la config de production pour une raison
de COHÉRENCE inter-architectures, pas d'échec sur le vectorisé. Réessayé ici
sur le checkpoint de production ACTUEL (`vectorized_trajectory`, post-audit).

**Méthode** : régénéré une table de recalibration propre à ce checkpoint
(`estimate_intensity_recalibration.py`, 4 sujets `retro_train` par classe,
240 volumes, `configs/mmfm/intensity_recalibration_vectorized_trajectory_20260918.json`),
appliquée aux prédictions d'évaluation déjà écrites
(`apply_intensity_recalibration.py`, CPU, aucune inférence), réévaluée avec
le protocole officiel.

| protocole officiel `cc`, 60 cellules | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **recalibré** | 0.3893 | 0.8061 | 0.1937 |
| production (identique sinon) | 0.3549 | 0.8098 | 0.1928 |
| Δ | **+0.0344** | **-0.0038** | +0.0008 |

Comparaison appariée : nRMSE 22/60 (p=0.052 signes, **p=0.014 Wilcoxon**),
SSIM 20/60 (p=0.014, p=0.007) — les deux significativement **pires**.
Par contraste (nRMSE) : T1W +0.0351 (7/20), **T2W +0.0676 (7/20, le plus
touché)**, T2FLAIR +0.0005 (8/20, quasi neutre).

**Cause identifiée, pas un bug d'implémentation** : les facteurs de la table
(1.02-1.11, tous proches de 1 — ce checkpoint est déjà mieux calibré que
l'ancien `vec_rbest`, qui allait de 0.74 à 1.56) sont comparés au facteur
ORACLE mesuré par cellule dans la section précédente (`a_or`, avec les
volumes de vérité des 3 sujets d'évaluation). Sur les 15 cellules
(3 contrastes × 5 champs cible), **6 vont dans le sens OPPOSÉ à l'oracle** —
concentrées sur T2W (3/5 : 0.1T, 1.5T, 3T) et sur les champs à plus forte
correction oracle en T1W (5T, 7T). C'est exactement là que les dégâts sont
les plus lourds.

**Ce que ça ajoute à la conclusion du 2026-08-30** (« variance inter-sujets
35 % contre une constante par classe, non réparable par un meilleur
estimateur ») : ce n'est plus seulement que le remède est insuffisant, c'est
que **sa direction même n'est plus stable d'un checkpoint à l'autre**. Un
correctif de mécanisme (l'audit du 2026-09-04) a suffisamment changé le biais
résiduel du modèle pour qu'une table de recalibration, réestimée proprement
sur CE checkpoint, pointe à l'envers de ce que les 3 sujets d'évaluation
demandent sur près de la moitié des cellules. **Piste refermée, plus
fermement qu'avant** : pas de biais de classe stable à corriger sans
appariement, sur aucun checkpoint testé à ce jour.

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
