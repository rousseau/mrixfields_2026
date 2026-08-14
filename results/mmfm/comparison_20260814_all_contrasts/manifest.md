# Task 3 sur les TROIS contrastes — T1W n'était pas représentatif

**Date** : 2026-08-14
**Modèle** : vectorisé de production (`outputs/mmfm/vectorized/weights/model_final.pth`,
run 2026-08-14 avec augmentation par flip).
**Protocole** : identique à T1W — `--norm_mode field_fixed --center_crop_only`,
évaluateur officiel, 20 paires x 3 sujets par contraste, split
`Training_prospective`.

---

## Pourquoi cette évaluation

Tous les chiffres du projet depuis son origine portaient sur T1W seul. Le
challenge en compte TROIS. **Deux tiers du problème n'étaient pas mesurés.**

## Résultat global

| contraste | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| T1W | 0.4353 | **0.8997** | 0.0983 |
| **T2W** | **0.3376** | 0.8980 | **0.0911** |
| T2FLAIR | 0.3654 | 0.8949 | 0.0927 |
| **moyenne 3 contrastes** | **0.3794** | **0.8975** | **0.0941** |

**T1W est le contraste le PLUS DIFFICILE des trois.** Le projet a donc optimisé
sur son pire cas, et le score réel toutes modalités (0.3794) est meilleur de 13 %
que le chiffre affiché jusqu'ici (0.4353). Le SSIM et le LPIPS, eux, sont
remarquablement stables d'un contraste à l'autre (0.895-0.900 / 0.091-0.098) :
c'est le nRMSE qui discrimine.

## nRMSE par champ CIBLE — le « mur du 7T » est propre au contraste

| contraste | 0.1T | 1.5T | 3T | 5T | 7T |
|---|---|---|---|---|---|
| T1W | 0.2709 | 0.3306 | 0.3146 | 0.5061 | **0.7542** |
| T2W | 0.2927 | 0.3505 | 0.3507 | 0.3323 | **0.3620** |
| T2FLAIR | 0.3386 | 0.2524 | 0.3333 | 0.2668 | **0.6359** |

En T2W, la cible 7T coûte à peine plus que les autres. En T1W elle coûte le
TRIPLE des cibles faciles. La difficulté du haut champ — qui a motivé une part
importante du travail de ce projet (l'INR un temps retenu pour ses victoires sur
->7T, le conditionnement AdaGN visant explicitement ces cibles) — n'est donc pas
une propriété du problème mais **du contraste T1W**.

T2FLAIR est intermédiaire : plat jusqu'à 5T, puis 0.6359 vers 7T.

### Paires extrêmes

| contraste | pire paire | meilleure paire |
|---|---|---|
| T1W | `1.5T_to_7T` **0.9569** | `3T_to_0.1T` 0.1959 |
| T2W | `0.1T_to_3T` 0.5053 | `3T_to_1.5T` 0.2419 |
| T2FLAIR | `3T_to_7T` **0.8565** | `3T_to_5T` 0.1454 |

Noter que T2W n'a AUCUNE paire au-dessus de 0.51, là où T1W et T2FLAIR
dépassent 0.85. Et que les pires paires de T2W partent toutes de 0.1T (le champ
le plus éloigné), pas vers 7T.

## Le volume de données n'explique RIEN — prédiction réfutée

Avant de lancer, la rareté de T2W@5T (43 volumes, la plus petite classe du jeu)
avait été annoncée comme cause probable si ce champ se dégradait. **Il ne s'est
pas dégradé** : 0.3323, son deuxième meilleur score.

La corrélation est même INVERSE de celle attendue :

| | volumes d'entraînement | nRMSE ->7T |
|---|---|---|
| T1W@7T | **235** (le plus du jeu) | **0.7542** (le pire) |
| T2FLAIR@7T | 84 | 0.6359 |
| T2W@7T | 81 | **0.3620** (le meilleur) |

Le contraste le mieux doté en 7T est celui qui y échoue le plus. **Augmenter les
données de T1W@7T ne serait donc pas la piste à suivre.**

Hypothèse (physique, NON établie par ces mesures) : à 7 T le temps de relaxation
T1 s'allonge nettement, si bien qu'un T1W à 7 T diffère beaucoup d'un T1W à bas
champ, tandis qu'un T2W y est bien moins affecté. La transition à apprendre
serait alors intrinsèquement plus grande en T1W. À confirmer ou infirmer
autrement que par ces chiffres.

## Distribution d'entraînement (retro_train, 1939 volumes)

|  | 0.1T | 1.5T | 3T | 5T | 7T |
|---|---|---|---|---|---|
| T1W | 100 | 104 | 143 | 121 | 235 |
| T2W | 100 | 221 | 142 | **43** | 81 |
| T2FLAIR | 99 | 221 | 143 | 102 | 84 |

Les 15 classes sont peuplées, donc toutes les transitions ont été vues à
l'entraînement. T2W@5T (43) reste la classe qui contraint la taille de lot :
au-delà, son chargeur ne produirait jamais de lot complet avec `drop_last=True`
— blocage silencieux, pas erreur.

## Conséquences

1. **Le score de référence du projet devient 0.3794** (moyenne 3 contrastes), et
   non 0.4353. Les comparaisons d'architectures passées restent valides entre
   elles — toutes mesurées sur T1W — mais sous-estimaient le niveau réel.
2. **Toute optimisation ciblant le haut champ doit préciser le contraste.**
   Un gain sur T1W@7T ne se transporte pas : T2W n'a pas ce problème.
3. **Les architectures n'ont été comparées que sur T1W.** Rien ne garantit que le
   classement vectorisé > UNet > INR tienne sur T2W/T2FLAIR — non mesuré.

## Fichiers

| chemin | contenu |
|---|---|
| `task3_vectorized_T2W.csv` | 20 paires x 3 sujets, T2W |
| `task3_vectorized_T2FLAIR.csv` | 20 paires x 3 sujets, T2FLAIR |
| `../comparison_20260814_vectorized_flip/task3_vectorized_flip_T1W.csv` | T1W (référence) |
