# Fine-tuning LOO — budget étendu (5000 itérations), point d'arrêt par contraste

**Date** : 2026-09-19/20
**Verdict : MIXTE.** Gain net et significatif en nRMSE ; mais SIGNIFICATIVEMENT
pire en SSIM et LPIPS sous la géométrie officielle (`8win`) — un compromis
distorsion/perception concentré sur T1W, pas un gain propre.

## Motivation

Le point d'arrêt du fine-tuning LOO original (1500 itérations, entrées du
2026-09-16/17) avait été choisi sur un PROXY — le nRMSE latent agrégé sur les
3 contrastes — jamais sur le score Task 3 réel ni décomposé par contraste.
Comme T2W utilise de toute façon la production à l'inférence (sélection par
contraste, entrée du 2026-09-17), rallonger le budget spécifiquement pour
T1W/T2FLAIR et choisir le point d'arrêt sur LEUR propre trajectoire ne pouvait
pas risquer T2W.

## Méthode

- 3 replis réentraînés à 5000 itérations (au lieu de 1500), tout le reste
  identique (`configs/mmfm/vectorized_finetune_loo_excl{0006,0007,0009}_5k.yaml`
  — copie stricte + `total_iters`/`save_every`, sortie dans un dossier `_5k`
  dédié pour ne pas écraser la référence à 1500 itérations).
- Diagnostic bon marché (`diagnose_finetune_checkpoints.py --config ..._5k.yaml`,
  nRMSE latent contre la VRAIE cible du sujet tenu à l'écart, décomposé par
  contraste via les 12 valeurs par checkpoint) pour choisir, PAR (repli,
  contraste), le meilleur checkpoint T1W et T2FLAIR sans payer l'inférence
  Task 3 sur les ~21 checkpoints × 3 replis :

| repli | T1W (iter, nRMSE latent) | T2FLAIR (iter, nRMSE latent) |
|---|---|---|
| 0006 | 3750, 0.2136 (repère 1500-budget : 0.2021) | 2750, 0.1844 (repère : 0.2046) |
| 0007 | 4750, 0.1861 (repère : 0.2062) | 2000, 0.2595 (repère : 0.2692) |
| 0009 | 4750, 0.2057 (repère : 0.2493) | 1000, 0.1929 (repère : 0.1945) |

5 des 6 cellules améliorent le proxy latent par rapport au meilleur atteignable
dans le budget original (1500 itérations) ; seule 0006/T1W régresse légèrement
(0.2021 → 0.2136).

- Assemblage : prédictions Task 3 générées avec le checkpoint choisi par
  (repli, contraste) pour T1W et T2FLAIR (un sujet par repli, celui qu'il n'a
  pas vu), T2W repris directement des prédictions production déjà écrites
  (mêmes fichiers, aucune inférence). Évalué en `cc` puis confirmé en `8win`.

## Résultat

### `cc` (rapide)

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **nouveau (5k, par contraste)** | 0.2902 | 0.8218 | 0.1826 |
| best-of-both (référence, repli seul) | 0.3207 | 0.8238 | 0.1802 |
| Δ | **-0.0305** (p=0.0034 Wilcoxon) | -0.0020 (NS) | +0.0024 (NS, limite) |

Signal prometteur, net sur nRMSE, neutre sur SSIM/LPIPS — a motivé la
confirmation `8win`.

### `8win` (officielle) — le tableau se complique

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **nouveau (5k, par contraste)** | 0.2854 | 0.8387 | 0.2014 |
| best-of-both (référence, repli seul) | 0.3092 | 0.8421 | 0.1958 |
| Δ | **-0.0237** (p=0.0161 Wilcoxon, p=0.268 signes NS) | **-0.0035** (p=0.029, PIRE) | **+0.0055** (p=0.0015, PIRE) |

**SSIM et LPIPS sont significativement PIRES sous la géométrie officielle**,
alors qu'ils étaient neutres en `cc`. Décomposition par contraste :

| | ΔnRMSE | ΔSSIM | ΔLPIPS |
|---|---|---|---|
| T1W | -0.0322 | **-0.0112** | **+0.0129** |
| T2W | 0.0000 (identique) | 0.0000 | 0.0000 |
| T2FLAIR | -0.0390 | +0.0008 (neutre) | +0.0038 (neutre) |

**Le compromis est concentré sur T1W.** Le checkpoint T2FLAIR étendu (iter
2000/2750/1000 selon le repli) est un gain propre sur les trois métriques.
Le checkpoint T1W étendu (iter 3750/4750) échange de la justesse d'échelle
(nRMSE) contre de la fidélité structurelle/perceptuelle (SSIM/LPIPS) — un
sur-apprentissage probable sur les 2 sujets d'entraînement de ce contraste au
-delà d'un certain budget, visible seulement sous la géométrie qui moyenne 8
fenêtres (plus sensible à la structure fine que le simple crop centré).

## Verdict et recommandation

**Ne pas adopter tel quel comme nouvelle référence.** Le gain nRMSE existe et
est statistiquement soutenu (Wilcoxon), mais la dégradation SSIM/LPIPS sur
T1W est elle aussi significative — remplacer la référence reviendrait à
répéter l'erreur déjà écartée pour la recalibration d'intensité (« garder un
réglage qui marche sur 1 contraste sans explication du coût sur les autres
n'est pas justifié »).

**Ce qui EST actionnable sans coût supplémentaire** : adopter l'extension
UNIQUEMENT pour T2FLAIR (gain propre, trois métriques) tout en gardant le
checkpoint T1W du best-of-both original (repère 1500 itérations).

## Suite — combinaison "mixte" : T1W original + T2FLAIR étendu + T2W production (2026-09-20)

Assemblée SANS coût GPU supplémentaire (recombinaison de prédictions déjà
écrites : T1W depuis `outputs/mmfm/finetune_loo_aggregate_8win` — best-of-both
original, 1500 itérations — T2FLAIR depuis le run étendu ci-dessus, T2W depuis
la production). C'est le candidat suggéré par la section précédente.

| `8win`, 60 cellules | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **mixte** | **0.2962** | 0.8424 | 0.1971 |
| best-of-both (référence) | 0.3092 | 0.8421 | 0.1958 |
| production | 0.3394 | 0.8313 | 0.2060 |

Seul T2FLAIR diffère du best-of-both (T1W et T2W identiques, 40/60 cellules
strictement égales) — comparaison isolée sur les 20 cellules T2FLAIR :
nRMSE **-0.0390 (Wilcoxon p=0.0107, significatif)**, SSIM +0.0008 (neutre,
p=0.67), LPIPS +0.0038 (pas significatif, p=0.15, mais dans le mauvais sens —
à surveiller, pas à ignorer).

**Verdict : gain net et propre sur nRMSE, sans dégradation significative de
SSIM/LPIPS.** Meilleur candidat de référence à ce jour pour le fine-tuning
LOO — remplace le best-of-both par repli.

## Réserves

- La tendance LPIPS légèrement défavorable sur T2FLAIR (+0.0038, NS à n=20)
  mérite d'être surveillée si ce contraste est retouché à nouveau — à ce
  budget de preuve elle n'est pas distinguable du bruit, mais elle va dans
  le même sens que le compromis observé sur T1W à budget plus long.

- Le proxy (nRMSE latent) qui a guidé le choix du checkpoint T1W ne voit pas
  le compromis SSIM/LPIPS révélé par l'évaluation Task 3 complète — troisième
  cas dans ce projet où un proxy latent améliore sans que l'image décodée
  suive de la même façon.
- Non testé : un point d'arrêt T1W intermédiaire (entre 1500 et 4750) pourrait
  offrir un meilleur compromis nRMSE/SSIM — nécessiterait de payer l'inférence
  Task 3 sur 2-3 checkpoints candidats supplémentaires plutôt que de se fier
  au seul proxy latent.
- La combinaison « T1W best-of-both original + T2FLAIR étendu + T2W
  production » (le vrai candidat à confirmer comme prochaine référence) n'a
  pas encore été assemblée ni évaluée.
