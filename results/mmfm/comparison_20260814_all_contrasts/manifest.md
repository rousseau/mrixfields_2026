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
3. ~~Les architectures n'ont été comparées que sur T1W.~~ **RÉGLÉ le 2026-08-15**
   (complément en fin de fichier) : les trois architectures ont été évaluées sur
   les trois contrastes, et le classement vectorisé > UNet > INR tient dans les
   NEUF cellules, sur les trois métriques.

## Fichiers

| chemin | contenu |
|---|---|
| `task3_vectorized_T2W.csv` | 20 paires x 3 sujets, T2W |
| `task3_vectorized_T2FLAIR.csv` | 20 paires x 3 sujets, T2FLAIR |
| `../comparison_20260814_vectorized_flip/task3_vectorized_flip_T1W.csv` | T1W (référence) |


---

# Complément (2026-08-15) — les TROIS architectures sur les TROIS contrastes

Le manifest ci-dessus n'évaluait que le vectorisé. Question ouverte qu'il
laissait : le classement vectorisé > UNet > INR, établi sur T1W seul — donc sur
le contraste dont on venait d'apprendre qu'il est atypique — tient-il ailleurs ?

**Checkpoints** : vectorisé de production (2026-08-14, avec flip), UNet de
référence (0.4617, PRE-AdaGN — évalué via une config `use_adagn: false`, le
checkpoint étant incompatible avec la config de production actuelle), INR de
production @1mm. Même protocole que partout ailleurs.

## nRMSE

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| **Vectorisé** | **0.4353** | **0.3376** | **0.3654** | **0.3794** |
| UNet | 0.4617 | 0.3676 | 0.3807 | 0.4033 |
| INR | 0.6223 | 0.6470 | 0.5998 | 0.6231 |

## SSIM

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| **Vectorisé** | **0.8997** | **0.8980** | **0.8949** | **0.8975** |
| UNet | 0.8955 | 0.8963 | 0.8929 | 0.8949 |
| INR | 0.8002 | 0.7576 | 0.8028 | 0.7868 |

## LPIPS

| | T1W | T2W | T2FLAIR | moyenne |
|---|---|---|---|---|
| **Vectorisé** | **0.0983** | **0.0911** | **0.0927** | **0.0941** |
| UNet | 0.1014 | 0.0919 | 0.0927 | 0.0953 |
| INR | 0.2051 | 0.2189 | 0.2034 | 0.2091 |

## Conclusion : le classement tient partout

**Neuf cellules, neuf fois le même ordre** — vectorisé > UNet > INR, en nRMSE
comme en SSIM et en LPIPS. Le choix d'architecture du projet, pris sur T1W seul,
vaut pour l'ensemble du challenge. C'était la principale question laissée
ouverte par l'évaluation multi-contrastes ; elle est tranchée.

Nuance sur l'INR : son écart au vectorisé se CREUSE sur le contraste le plus
facile (T2W +0.309, contre T1W +0.187). Sa limite est représentationnelle, pas
liée à la difficulté de la tâche — cohérent avec les deux fermetures déjà
documentées (plafond `modulation_dim`, puis décodeur à grille).

## Le 7T : porté par les données, pas par l'architecture

nRMSE vers 7T :

| | T1W | T2W | T2FLAIR |
|---|---|---|---|
| Vectorisé | 0.7542 | **0.3620** | 0.6359 |
| UNet | 0.7956 | 0.3954 | 0.6899 |
| INR | 0.8259 | 0.6422 | 0.7174 |

Les trois architectures réussissent le 7T en T2W et échouent en T1W, **dans le
même ordre**. Aucune n'a de force propre sur le haut champ. Cela clôt
définitivement l'idée — née des premiers résultats de l'INR à 2mm, qui gagnait
alors 4/4 des paires ->7T — qu'une architecture pourrait être spécifiquement
adaptée aux champs extrêmes. Cet avantage était un artefact de son flou, pas une
propriété.

## Détail par champ cible

### vectorisé — nRMSE par champ cible
| contraste | 0.1T | 1.5T | 3T | 5T | 7T |
|---|---|---|---|---|---|
| T1W | 0.2709 | 0.3306 | 0.3146 | 0.5061 | 0.7542 |
| T2W | 0.2927 | 0.3505 | 0.3507 | 0.3323 | 0.3620 |
| T2FLAIR | 0.3386 | 0.2524 | 0.3333 | 0.2668 | 0.6359 |

### UNet — nRMSE par champ cible
| contraste | 0.1T | 1.5T | 3T | 5T | 7T |
|---|---|---|---|---|---|
| T1W | 0.3017 | 0.3526 | 0.3299 | 0.5286 | 0.7956 |
| T2W | 0.3806 | 0.3561 | 0.3657 | 0.3404 | 0.3954 |
| T2FLAIR | 0.3452 | 0.2697 | 0.3379 | 0.2606 | 0.6899 |

### INR — nRMSE par champ cible
| contraste | 0.1T | 1.5T | 3T | 5T | 7T |
|---|---|---|---|---|---|
| T1W | 0.5635 | 0.5575 | 0.5396 | 0.6250 | 0.8259 |
| T2W | 0.6193 | 0.6676 | 0.6677 | 0.6382 | 0.6422 |
| T2FLAIR | 0.5868 | 0.5449 | 0.5853 | 0.5647 | 0.7174 |
