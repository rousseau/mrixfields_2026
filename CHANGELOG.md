# Journal des expériences — MRIxFields 2026

Chaque étape et chaque test mesuré, du plus récent au plus ancien. Une entrée par
expérience, **y compris et surtout les résultats négatifs** : ils sont la majorité,
et ce sont eux qu'on refait par inadvertance quand ils ne sont écrits nulle part.

**Convention.** Toute expérience qui produit un chiffre entre ici, avec :
sa date, ce qui était testé, **le chiffre**, le verdict, et le chemin du manifeste
détaillé. Le manifeste porte la méthode et les réserves ; cette page porte
l'historique et permet de retrouver quoi que ce soit en une lecture. Une
expérience non écrite ici est réputée ne pas avoir eu lieu.

---

## État de référence au 2026-08-26

Task 3, 20 paires × 3 sujets, évaluateur officiel, 1 mm, prédictions écrites à
0.5 mm natif.

| nRMSE | T1W | T2W | T2FLAIR | moyenne | SSIM moy. | LPIPS moy. |
|---|---|---|---|---|---|---|
| **INR** (`outputs/mmfm/inr_std`) | 0.4070 | 0.3800 | **0.3376** | **0.3749** | 0.8631 | 0.1557 |
| **Vectorisé** (`outputs/mmfm/vectorized`) | **0.4353** | **0.3376** | 0.3654 | 0.3794 | **0.8975** | **0.0941** |
| UNet (`outputs/mmfm/unet`) | 0.4617 | 0.3676 | 0.3807 | 0.4033 | 0.8949 | 0.0953 |
| *Témoin identité (recopier la source)* | *0.9273* | *0.3859* | *0.5574* | *0.6235* | *0.8882* | — |

**Le classement n'est pas un ordre total** : l'INR est première en nRMSE et
dernière en SSIM et LPIPS. Ne jamais écrire « la meilleure » sans la métrique.

**Planchers connus, à soustraire de toute conclusion** : écrêtage à `hi`
(0.10–0.13 de nRMSE pour un prédicteur parfait), interpolation 1 mm → 0.5 mm
(0.0156). **n = 3 est un plafond imposé par les données**, pas une négligence.

---

## Leçons récurrentes, avec leur compteur

**La loss d'entraînement ne prédit rien sur ce pipeline — 6 occurrences.**
2026-08-13 (AdaGN : loss +2.4, résultat identique) · 2026-08-14 (flip : loss +1.8,
résultat identique) · et la plus nette, 2026-08-26 : la loss du flow INR divisée
par 3.3 et passée sous le seuil du prédicteur constant, **score T1W dégradé**
(0.3787 → 0.4070). Ne jamais conclure sur une courbe de loss.

**Une moyenne ne dit pas ce qu'on croit — 2 occurrences majeures.**
2026-08-14 (T1W n'était pas représentatif : 2/3 du problème non mesuré) ·
2026-08-25 (sans témoin identité, un gain de +5.6 % passait pour un score de
0.3376). Toujours décomposer : par sujet, par paire, et contre un témoin trivial.

**Un test dont le seuil accepte le cassé ne teste rien.**
`test_inr_backbone_smoke.py:200` assère `nrmse_fg < 0.6` quand la production
valait 0.6383, à 2 mm alors que la production est à 1 mm. Il a laissé passer
trois bugs pendant des mois. **Non corrigé à ce jour.**

---

## 2026-08-26 — Audit phase B : correctifs de fond `d8db590`

**Verdict : l'INR passe première en nRMSE (0.3749) et reste dernière en SSIM et
LPIPS.** Détail : `results/mmfm/audit_20260825/manifest.md`.

| correctif | test | résultat |
|---|---|---|
| EMA avec échauffement (`(1+t)/(10+t)`) | même modèle avec/sans EMA | 0.4070 vs 0.4055 — **du bruit** (l'écart valait 0.195 avant) |
| Standardisation de l'entrée du flow (`latent_scale: 2.16e-4`) | loss contre « prédire zéro » (4.02e-4) | 1.10e-3 → **3.37e-4**, soit 16 % sous la meilleure constante |
| Deux clés de config + garde-fou `_check_cache_consistency` | refus de tourner sur cache incohérent | testé dans les deux sens |
| `sample_points` graîné | deux ajustements du même volume | déterminisme vérifié (plancher de 4.4 % éliminé) |
| **Contrôle** : vectorisé avec orientation alignée | 20 paires × 3 sujets | 0.4351 vs 0.4353 — **neutre, les chiffres publiés tiennent** |

**Résultat négatif notable** : le réentraînement dégrade T1W (0.3787 → 0.4070)
tout en améliorant la moyenne (0.3824 → 0.3749). Dans un latent à faible structure
commune, un flow qui bouge davantage prend plus de risques qu'il n'en gagne.

**Plafond restant, mesuré et géométrique** : part de l'énergie portée par la
moyenne commune du nuage de latents — MedVAE **91.3 %** (dispersion 0.416), INR
**25.1 %** (dispersion 1.221, quasi isotrope). Ce n'est plus un bug.

**Non fait** : régénérer le cache INR sous le prétraitement corrigé (~6 h) ;
remplacer le test dont le seuil accepte le cassé.

## 2026-08-25 — Audit phase A : trois bugs d'inférence `d6ade93`

**Verdict : l'INR n'était pas limitée, elle était cassée. 0.6383 → 0.3787 sur
T1W, sans réentraîner.** Détail : `results/mmfm/audit_20260825/manifest.md`.

| test | résultat |
|---|---|
| A1 — INR sans EMA | 0.6383 → **0.4430**, 19/20 paires améliorées |
| A2–A4 — dérive du latent (10 volumes) | 0.691 → 0.305 (orientation) → 0.069 (normalisation) → 0.044 (plancher) |
| A5 — **témoin** : vectorisé sans EMA | 0.4353 → 0.4359, **insensible** |
| A6 — aller-retour de représentation à 1 mm | nRMSE 0.14 / 0.18 / 0.28 à 0.1T / 3T / 7T — **la représentation n'était pas en cause** |
| A7 — les trois correctifs ensemble | **0.3787**, devant le vectorisé sur T1W |

**Les trois bugs** : ① EMA sans correction de biais (`0.9999^25000 = 0.082`, soit
8.2 % du bruit initial servi à l'inférence — 216× le signal INR, négligeable pour
MedVAE) ; ② orientation **LAS→RAS** (chaque volume arrivait mirroré ;
`flip_lr_prob: 0.5` rendait le vectorisé et l'UNet insensibles, `arch_inr.py`
refuse le flip donc l'INR prenait tout) ; ③ normalisation (cache bâti sans
`field_norm_stats`, inférence en `field_fixed`).

**L'augmentation par flip masquait le bug d'orientation pour deux architectures
sur trois.**

## 2026-08-25 — Évaluation qualitative et quantitative des trois approches `45f1cf1`

**Verdict : 80 % de l'énergie de l'erreur est une erreur d'échelle d'intensité,
pas d'anatomie.** Détail : `results/mmfm/qualitative_20260824/manifest.md`.

| mesure | résultat |
|---|---|
| Correction d'échelle oracle (un scalaire par volume) | supprime **82 %** (T1W) et **83 %** (T2FLAIR) de l'énergie de l'erreur ; 39 % en T2W ; 20–24 % pour l'INR |
| Constante de recalage par paire, estimée sans oracle (leave-one-subject-out) | 0.3794 → **0.3231**, −15 %, **sans réentraîner** |
| Témoin identité, gain paire-à-paire | +34.1 % T1W, **+5.6 % T2W, −1.2 % T2FLAIR** (battu sur 11 paires sur 20) |
| Sujet 0009, T1W@7T | norme L2 de 325 contre 845 et 705 → **46 % du « mur du 7T » vient d'un seul volume** |
| Indice de netteté (intérieur du cerveau) | vectorisé 0.36 / UNet 0.36 / INR 0.24, témoin 1.00 — **un tiers de la finesse réelle** |
| Estimateur d'échelle depuis le volume source | **dégrade** (0.4353 → 0.5642) : non biaisé mais trop bruité |

**Découverte structurelle** : les données d'entraînement sont **non appariées** —
aucun des 1056 sujets rétrospectifs n'existe à deux champs, et il n'y a que
**3 sujets appariés** dans tout le jeu. `n = 3` est un plafond, pas un raccourci.

## 2026-08-25 — MedVAE avec perte perceptuelle : la barre est franchie

**Verdict : LPIPS + PatchGAN bat le pré-entraîné sur les deux métriques et les
deux résolutions.** Détail : `results/benchmark_vae/analysis/medvae_lpips_20260825/manifest.md`.

| auto-reconstruction | @1 mm nRMSE / SSIM | @0.5 mm nRMSE / SSIM |
|---|---|---|
| pré-entraîné | 0.0819 / 0.9152 | 0.0488 / 0.9481 |
| fine-tuné L1 (écarté) | **0.0592** / 0.8790 | **0.0350** / 0.8837 |
| **fine-tuné LPIPS** | 0.0682 / **0.9727** | 0.0381 / **0.9825** |

**Le diagnostic du 2026-08-07 (« la recette de loss n'est pas la cause
dominante ») était faux** : même recette, LPIPS à la place de L1, +0.057 de SSIM
là où L1 en perdait 0.036. **Adoption NON décidée** — impose de régénérer les
caches et de réentraîner les deux flows (>1 jour), et n'agit que sur les 20 % de
l'erreur qui ne sont pas d'échelle.

## 2026-08-15 — Les trois architectures sur les trois contrastes `27f1153`

**Verdict de l'époque : le classement vectorisé > UNet > INR tient dans les neuf
cellules.** Détail : `results/mmfm/comparison_20260814_all_contrasts/manifest.md`.

> **⚠ INVALIDÉ le 2026-08-25** : ce classement mesurait trois bugs d'inférence de
> l'INR. La case INR/T1W mélangeait de plus deux modèles (0.6223 pour `z=4096`
> alors que le checkpoint de production est `z=129024`, soit 0.6383).
> L'écart vectorisé/UNet, lui, tient — mais il vaut un sixième de l'écart-type
> inter-sujet et n'est significatif que sur T1W (16/20, p=0.006 ; T2W 12/20 ;
> T2FLAIR 10/20).

## 2026-08-14 — Task 3 sur les trois contrastes : T1W n'était pas représentatif `2f00b5b`

**Verdict : deux tiers du problème n'étaient pas mesurés.** Détail :
`results/mmfm/comparison_20260814_all_contrasts/manifest.md`.

Vectorisé : T1W 0.4353, **T2W 0.3376**, T2FLAIR 0.3654 → moyenne **0.3794**.
**T1W est le contraste le plus difficile** ; le projet avait optimisé sur son pire
cas. Le « mur du 7T » (0.7542 en T1W) **n'existe pas en T2W** (0.3620).

**Prédiction réfutée** : la rareté de T2W@5T (43 volumes, la plus petite classe)
avait été annoncée comme cause probable de dégradation. Score obtenu : 0.3323, son
deuxième meilleur. La corrélation avec le volume de données est **inverse** —
T1W@7T est la classe la mieux dotée (235 volumes) et la plus mauvaise.

## 2026-08-14 — Bug d'équité du flip corrigé ; le classement est confirmé `b6efe83`

**Verdict : le flip est neutre (0.4353 contre 0.4354), le classement publié n'en
dépendait pas.** Détail : `results/mmfm/comparison_20260814_vectorized_flip/manifest.md`.

`flip_lr_prob: 0.5` était déclaré dans les trois configs mais **seul l'UNet le
recevait** — `FlatLatentCacheDataset` n'acceptait pas l'argument et l'avalait sans
erreur. Un latent retourné diffère de **26 %** en L2. Le bug handicapait le
gagnant, la conclusion était donc conservatrice — mais ce n'était pas mesuré.
Ce checkpoint devient la référence : seul reproductible depuis sa config.

## 2026-08-14 — Piège de l'encodeur sans cache `ea6e990`

L'entraînement sans cache encodait le volume entier là où le precompute encode par
tuiles : **1.5 à 3.7 %** d'écart sur le latent. Aligné via `_encode_like_cache`.

> Erreur de méthode commise en vérifiant : j'ai d'abord utilisé une normalisation
> par percentiles là où le cache utilise `field_norm_stats`, ce qui a donné un
> faux 10.6 % et la fausse conclusion que le correctif ne marchait pas.

## 2026-08-13 — UNet + conditionnement AdaGN : NÉGATIF `223e5f4`

**Verdict : aucun effet. 0.4623 contre 0.4617.** Détail :
`results/mmfm/comparison_20260813_unet_adagn/manifest.md`.

3/20 paires gagnées, écarts uniformément entre +0.0003 et +0.0010 y compris sur
5T et 7T que le conditionnement visait. **Les prédictions des deux modèles ne
diffèrent que de 0.2–0.3 % en L2** malgré 3.4 M paramètres d'écart et 2.4 points
de loss. Le conditionnement n'était pas le goulot.

**Témoin identité établi pour la première fois** : 0.9273. Aucun modèle du projet
n'avait jamais été comparé à « ne rien faire ».

> Piège rencontré : MONAI initialise `conv2` et la conv de sortie à zéro, donc le
> réseau sort exactement zéro à l'initialisation et **tout test de conditionnement
> passe trivialement**. Corrigé en réveillant les 140 tenseurs nuls avant mesure.

## 2026-08-12 — Décodeur implicite conditionné par grille : NÉGATIF

**Verdict : l'architecture fonctionne mais n'apporte rien.** Détail :
`results/mmfm/grid_inr_decoder/manifest.md`.

Capacité portée de 1536 à 8.3 M, et pourtant sur 76 volumes tenus à l'écart à
0.5 mm : conv+trilinéaire **0.1346 / 0.9154** contre INR **0.1511 / 0.8836**.
L'écart croît avec le champ. **Deuxième fermeture de la piste INR**, sur un design
très différent du premier.

**Leçon de méthode** : `infer_mmfm_unified.py` interpole 1 mm → 0.5 mm à l'ordre 1,
ce qui coûte 0.0156 de nRMSE. Le décodeur à grille était donc plafonné à ce gain —
**calculable en dix minutes avant d'implémenter quoi que ce soit.**

## 2026-08-10/11 — Capacité du latent INR : 4096 → 129024 : NÉGATIF

**Verdict : légèrement pire. 0.6223 → 0.6383, pour ~20 h de calcul.** Détail :
`results/mmfm/comparison_20260807_1mm/manifest.md`.

> **Explication trouvée le 2026-08-25** : le goulot n'est pas `latent_dim`.
> `hypernet.net.0.weight` est (512, 129024) et `hypernet.net.2.weight` (1536, 512) :
> tout est écrasé en aval sur **1536 modulations** pour 8.26 M voxels. L'expérience
> ne pouvait rien changer par construction. `latent_dim` nomme la taille vue par le
> flow, pas la capacité de la représentation — confusion qui a coûté une expérience
> entière.

## 2026-08-07/10 — Migration 2 mm → 1 mm et comparaison à trois

**Verdict : la résolution était le vrai goulot pour vectorisé et UNet, et l'INR
est la seule que 1 mm dégrade.** Détail : `results/mmfm/comparison_20260807_1mm/manifest.md`.

| @1 mm, T1W | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| Vectorisé | **0.4354** | **0.8995** | **0.0985** |
| UNet | 0.4617 | 0.8955 | 0.1014 |
| INR (`z=4096`) | 0.6223 | 0.8002 | 0.2051 |

L'INR passe de 0.4869 (@2 mm) à 0.6223 (@1 mm) : **+0.135**.

## 2026-08-01 — Comparaison finale à 2 mm, trois architectures

**Verdict : vectorisé 0.4288, UNet 0.4741, INR 0.4869.** Détail :
`results/mmfm/comparison_20260801_final/manifest.md`.

Trois candidats par architecture (50k, prolongation à plateau, prolongation à
décroissance immédiate). La prolongation à **schedule corrigé** gagne pour le
vectorisé (0.4361 → 0.4288) et **dégrade** l'UNet (0.4741 → 0.4766). Section
« la résolution est le vrai goulot » : c'est ce diagnostic qui a motivé la
migration à 1 mm.

## 2026-07-30 — Comparaison harmonisée vectorisé vs UNet

**Verdict : 0.4361 contre 0.4741, à code strictement identique hors architecture.**
Détail : `results/mmfm/analysis/protocol_20260730_harmonized_comparison/`.

Suite à la demande explicite d'harmoniser le code pour comparer équitablement.
L'écart se concentre sur les cibles hautes : →5T 0.4997 contre 0.5435, →7T 0.7220
contre 0.8209.

## 2026-07-29 — Baseline UNet multi-marginal, et le correctif de dénormalisation

**Verdict : la dénormalisation par champ cible divise l'erreur par plus de deux
sur les paires extrêmes.** Détail : `results/mmfm/analysis/protocol_20260729_unet_mm_baseline/`.

`0.1T→7T` : nRMSE **1.506 → 0.666**, SSIM 0.770 → 0.878. Aucune évaluation fiable
n'existait pour cette lignée avant cette porte. Triage de 4 modèles avant
évaluation complète (run2_local retenu, 2.456 contre 3.07–3.63).

## 2026-07-22 — MMFM v2 vectorisé, et clamp de la vitesse `5ef9bca` `37da6f4`

Première version vectorisée du flow multi-marginal. Le clamp `±1e6` ajouté pour
l'instabilité numérique.

> **Vérifié le 2026-08-25 : ce clamp est inerte pour les trois architectures**
> (vitesses de l'ordre de 4e-4 pour l'INR, 15 pour MedVAE — cinq à neuf ordres de
> grandeur sous le seuil). L'epsilon `max(dt, 1e-4)` du même commit a disparu dans
> l'unification, correctement : `|dt| ≥ 0.25` par construction.

## 2026-06 à 2026-07-20 — Fondations

Benchmark des VAE 3D (AEKL, VQ-VAE, RHVAE, MedVAE, NV-Generate), réorganisation de
`results/`, MMFM-UNet v2 à attention factorisée, scripts Jean Zay (DDP 4×H100),
inférence pleine résolution par patches, intégration de l'évaluateur officiel.
Détail dans l'historique git — aucun manifeste dédié pour cette période, et c'est
une lacune que ce journal existe pour ne plus reproduire.

---

## Dette connue, non traitée

| # | sujet | coût estimé |
|---|---|---|
| 1 | `test_inr_backbone_smoke.py` : seuil `nrmse_fg < 0.6` qui accepte le cassé, à 2 mm | 1 h |
| 2 | Aucun test qui compare la loss finale à « prédire zéro » — trois lignes, aurait tout arrêté | 15 min |
| 3 | Régénérer le cache INR sous le prétraitement corrigé | ~6 h GPU |
| 4 | Constante de recalage d'intensité par paire (−15 % mesuré, sans réentraîner) : pas de données appariées pour l'ajuster hors des 3 sujets d'évaluation ; voie non testée = comparer les distributions d'intensité prédites et réelles, sans appariement | à instruire |
| 5 | Adoption du MedVAE perceptuel : régénérer les caches + réentraîner les deux flows | >1 jour |
| 6 | Géométrie du latent INR (25 % de structure commune contre 91 %) : canoniser l'ajustement | ~6 h |
