# Validation de la couche de représentation — MedVAE, décomposée source par source

**Date** : 2026-09-06
**Script** : `src/cfm/bench_representation.py`
**Données** : `Training_prospective`, sujet 0006, 3 contrastes × 5 champs = 15 volumes par variante
**CSV** : `cells.csv` (une ligne par volume × variante), `summary.csv` (moyennes)

---

## Ce que cette campagne mesure, et pourquoi elle était nécessaire

Le « plafond de représentation » du projet (nRMSE 0.1048, mesuré le 2026-09-01) est
produit par une inférence identité qui traverse **toute** la chaîne : rééchantillonnage
0.5 → 1 mm, normalisation avec écrêtage aux percentiles, crop, encodage par tuiles en
bfloat16, décodage, dénormalisation, retour en 0.5 mm. Ce nombre agrège donc au moins
six termes d'erreur, et **rien dans le dépôt ne disait lequel domine**. Toutes les
décisions prises sur la foi de « le VAE plafonne à 0.10 » — dont l'adoption du MedVAE
perceptuel — reposaient sur une quantité qui n'est pas une propriété du VAE.

Le banc mesure trois familles sur les mêmes volumes, avec le même code :

| famille | colonnes | ce qu'elle isole |
|---|---|---|
| **VAE seul** | `*_norm` | `encode → decode` comparé à l'entrée normalisée, dans l'espace de travail. Aucun rééchantillonnage, aucun écrêtage, aucune dénormalisation. |
| **chaîne complète** | `*_slab`, `*_full` | le chiffre comparable au classement, même chemin géométrique que `infer_mmfm_unified.process_volume_unified` avec le flow retiré. |
| **témoins sans VAE** | variantes `noop`, `protocol*` | le plancher imposé par le protocole seul. |

`nrmse_slab` porte sur la tranche axiale [150, 180) réellement notée
(`common/io.py::Z_CLIP_RANGE`) ; `nrmse_full` sur les 364 coupes, seule grandeur
comparable aux chiffres du CHANGELOG antérieurs au 2026-09-04.

---

## Résultat principal : le « plafond du VAE » est majoritairement le protocole

| variante | VAE seul nRMSE | SSIM | PSNR | chaîne [150,180) | SSIM | volume entier |
|---|---|---|---|---|---|---|
| `noop` (contrôle) | 0.0000 | 1.0000 | 99.00 | **0.0000** | 1.0000 | 0.0000 |
| **`protocol`** (chaîne SANS VAE) | 0.0000 | 1.0000 | 99.00 | **0.0685** | 0.9717 | 0.0705 |
| `protocol_noclip` (sans écrêtage) | 0.0000 | 1.0000 | 99.00 | **0.0326** | 0.9907 | 0.0364 |
| `protocol_nocrop` (sans crop) | 0.0000 | 1.0000 | 99.00 | 0.0685 | 0.9717 | 0.0705 |
| **`prod`** (production exacte) | **0.0597** | 0.9412 | 31.42 | **0.1140** | 0.8984 | 0.1184 |

**Le témoin `noop` vaut exactement 0** : la chaîne de mesure est vérifiée, elle ne
fabrique pas d'erreur.

**60 % de l'erreur mesurée par le plafond ne vient pas du VAE.** La chaîne sans aucun
VAE coûte déjà 0.0685 sur la tranche notée, contre 0.1140 pour la chaîne complète. Et
cette part se décompose encore :

- **écrêtage au percentile 99.5 : 0.0359** (0.0685 − 0.0326). C'est le terme dominant du
  protocole. Un prédicteur parfait ne peut pas restituer les voxels au-dessus de `hi`.
- **rééchantillonnage 0.5 → 1 → 0.5 mm : 0.0326**.
- **crop 192×224×192 : 0.0000** — mesuré, pas supposé (`protocol` et `protocol_nocrop`
  sont identiques à la quatrième décimale). Le crop ne coûte rien.

Le chiffre « 0.10–0.13 de nRMSE pour un prédicteur parfait », cité depuis le 2026-08-26
comme un plancher connu, est donc **confirmé dans son ordre de grandeur mais mal
attribué** : il valait 0.0685 sur la tranche notée pour ce sujet, et il est constitué à
52 % d'écrêtage et à 48 % de rééchantillonnage.

### La part du protocole varie énormément d'une cellule à l'autre

| cellule | protocole seul | production | part réellement due au VAE |
|---|---|---|---|
| T2FLAIR@3T | 0.1247 | 0.1397 | **11 %** |
| T1W@3T | 0.1121 | 0.1351 | 17 % |
| T1W@7T | 0.0981 | 0.1475 | 34 % |
| T2FLAIR@0.1T | 0.0135 | 0.0464 | **71 %** |
| T1W@5T | 0.0453 | 0.1202 | 62 % |

De 11 % à 71 % selon la cellule. **Comparer deux cellules du plafond entre elles n'a
donc pas de sens** : on compare surtout deux quantités d'écrêtage.

---

## Les écarts à la recette officielle MedVAE sont neutres ou favorables

La recette officielle (`medvae/utils/loaders.py::load_mri_3d`) est : orientation RAS,
`ScaleIntensity(0,1)` **min-max global**, `Normalize(0.5, 0.5)`, `CropForeground(k=16)`,
puis fenêtre glissante gaussienne (`medvae_main.py::MVAE.encode`). Le dépôt s'en écarte
sur cinq points. Tous ont été mesurés, un par un :

| écart au standard | VAE seul nRMSE | PSNR | chaîne [150,180) | verdict |
|---|---|---|---|---|
| **production** | 0.0597 | 31.42 | 0.1140 | référence |
| échantillon → **mode** de la postérieure | 0.0597 | 31.42 | 0.1139 | **neutre** |
| bfloat16 → **float32** | 0.0597 | 31.43 | 0.1140 | **neutre** |
| les deux | 0.0597 | 31.43 | 0.1140 | **neutre** |
| marge de contexte 16 → **32** | 0.0595 | 31.46 | 0.1137 | **ne mesure PAS la marge**, voir ci-dessous |
| tuilage maison → **fenêtre glissante officielle** | 0.0599 | 31.35 | 0.1168 | **le tuilage maison est légèrement MEILLEUR**, et 1.7× plus rapide |
| **recette officielle complète** | 0.0737 | 31.30 | 0.1509 | **nettement pire** |
| percentile + CropForeground + fenêtre glissante | 0.1167 | 26.63 | 0.1215 | **CropForeground coûte 4.7 dB** |

**Aucun des écarts au standard n'est un défaut de performance.** Le tuilage par tuiles
non recouvrantes avec marge de contexte, développé ici pour contourner l'OOM à 1 mm,
fait aussi bien que la fenêtre glissante gaussienne officielle. La normalisation par
percentiles bat le min-max officiel de bout en bout (0.1140 contre 0.1509). Et
`CropForeground`, qui est dans la recette officielle, **dégrade** ici de 4.7 dB —
vraisemblablement parce qu'un crop serré met le cerveau au contact du bord, où les
convolutions complètent par des zéros.

### Le bras « marge 32 » ne mesurait pas la marge — corrigé

`MVAE.encode()` re-découpe silencieusement son entrée dès qu'une dimension dépasse
`gpu_dim = 160` (`medvae/utils/extras.py::roi_size_calc`). Vérifié :

| marge | tuile élargie | fenêtre effective de MedVAE |
|---|---|---|
| 16 | 128×144×128 | 128×144×128 — **une seule fenêtre, mesure propre** |
| 32 | 160×176×160 | 160×**88**×160 — **re-découpé en fenêtres glissantes** |

Le chiffre 0.1137 est donc celui de « marge 32 **+ sous-tuilage en W** », pas celui de
la marge. La conclusion « augmenter la marge ne sert à rien » n'était pas établie.

**Reprise propre** (`gpu_dim = 256`, une seule fenêtre dans les deux cas, 15 volumes,
`results/mmfm/representation_20260906_d/`) :

| | VAE seul nRMSE | SSIM | PSNR | chaîne notée |
|---|---|---|---|---|
| marge 16 | 0.0597 | 0.9414 | 31.43 | 0.1140 |
| marge 32 | 0.0594 | 0.9419 | 31.47 | 0.1138 |
| **écart** | **−0.00030** | +0.0005 | **+0.043 dB** | −0.0002 |

**La conclusion tient : la marge de 16 suffit.** Elle est maintenant établie sur une
expérience où la marge est la seule variable qui change. Contrôle de cohérence :
`prod_margin16_clean` (avec `gpu_dim = 256`) donne exactement les mêmes chiffres que
`prod` — ce qui était attendu, puisqu'à marge 16 la tuile élargie passe déjà sous 160.

La production est donc bien dans le régime propre.

### Un défaut réel, mais sans effet sur la fidélité

`MVAE.encode()` en 3D passe par `AutoencoderKL_3D.forward(sample_posterior=True)` et
renvoie donc `posterior.sample()`, **pas le mode** — contrairement au commentaire
« returns mode directly » de `src/models/maisi_vae.py:130`. Mesuré : deux appels
successifs sur la même entrée diffèrent, rapport bruit/signal du latent **2.3e-3**.

C'est un vrai défaut de **reproductibilité** (le cache de latents n'est pas
déterministe), et il doit être corrigé pour cette raison. Mais son effet sur la
reconstruction est **nul à la quatrième décimale** (`prod` 0.0597 contre `prod_mode`
0.0597). Ne pas lui attribuer d'espoir de gain.

---

## Le seul levier réel : combien la dynamique est remplie, et combien on écrête

L'erreur ABSOLUE du VAE est à peu près constante (PSNR 31.3–31.5 dB dans toutes les
variantes de normalisation). L'erreur RELATIVE, elle, dépend entièrement de la place
que le signal occupe dans [−1, 1]. D'où un compromis à un seul paramètre :

| variante | occupation (écart-type fg) | VAE seul nRMSE | SSIM | chaîne [150,180) |
|---|---|---|---|---|
| `prod_hi050` (`hi` ÷2) | 0.375 | 0.0589 | **0.9617** | **0.3900** |
| `prod_hi075` (`hi` ×0.75) | 0.424 | 0.0626 | 0.9498 | 0.2028 |
| **`prod`** (`hi` de population) | 0.406 | 0.0597 | 0.9412 | 0.1140 |
| `prod_volpct` (percentile par volume) | 0.402 | 0.0603 | 0.9392 | **0.1009** |
| **`prod_p999`** (percentile 99.9 par volume) | 0.368 | **0.0551** | 0.9423 | **0.0986** |

Serrer `hi` remplit la dynamique et améliore franchement le SSIM du VAE seul
(0.9412 → 0.9617), **mais détruit le score de bout en bout** (0.1140 → 0.3900) parce que
l'écrêtage devient irréversible. Desserrer l'écrêtage (percentile 99.9) gagne des deux
côtés : **−0.0154 de nRMSE sur la chaîne complète, soit 13.5 % relatif.**

**Réserve décisive sur `prod_volpct` et `prod_p999`** : les percentiles PAR VOLUME sont
exactement inversibles pour une reconstruction identité, mais **indisponibles en
traduction** — on ne connaît pas les percentiles du volume cible. Le gain mesuré avec
eux est un plafond, pas un résultat transposable tel quel. La forme transposable est un
`hi` FIXE par (contraste, champ), simplement desserré : c'est le balayage `hi_scale`.

### Le balayage `hi_scale` — la forme transposable, et son optimum

`hi_scale` multiplie le `hi` de `field_norm_stats.json`. C'est toujours une borne fixe
par (contraste, champ), donc **utilisable en traduction**, contrairement aux percentiles
par volume. 15 volumes, tranche notée :

| `hi` × | 0.5 | 0.75 | **1.0 (prod)** | **1.25** | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| nRMSE | 0.3900 | 0.2028 | **0.1140** | **0.0989** | 0.1001 | 0.1053 |
| SSIM | 0.7901 | 0.8569 | 0.8984 | **0.9077** | 0.9089 | 0.9085 |

**Optimum plat entre ×1.25 et ×1.5, à −13.2 %.** Le mécanisme est mesuré : la table
sature 25–50 % des voxels de cerveau selon la cellule (T2FLAIR@3T 50.0 %, T1W@3T 37.1 %,
T1W@7T 25.1 %), et l'écrêtage n'est pas inversible.

### Les deux leviers se composent

| variante | nRMSE tranche notée | SSIM | vs production | part propre au VAE¹ |
|---|---|---|---|---|
| *protocole seul (plancher)* | *0.0685* | *0.9717* | — | *0.0000* |
| **production** | 0.1140 | 0.8984 | — | 0.0911 |
| MedVAE LPIPS | 0.1063 | 0.9129 | −6.7 % | 0.0813 |
| production + `hi` ×1.25 | 0.0989 | 0.9077 | −13.2 % | 0.0714 |
| **LPIPS + `hi` ×1.25** | **0.0901** | **0.9227** | **−21.0 %** | **0.0585** |
| LPIPS + `hi` ×1.5 | 0.0904 | 0.9221 | −20.7 % | 0.0589 |

¹ `sqrt(total² − plancher²)`, les deux termes étant approximativement orthogonaux.

**−21.0 % sur la chaîne complète et −36 % sur la part propre au VAE, sans réentraîner
quoi que ce soit** : un checkpoint déjà présent dans le dépôt et un scalaire par cellule
dans un JSON. Le SSIM monte aussi (0.8984 → 0.9227), donc ce n'est pas un arbitrage.

---

## Reproduction du plafond publié

Le banc reproduit le plafond publié le 2026-09-01 (volume entier, MedVAE pré-entraîné)
avec un décalage systématique de **+0.0135** (0.1184 contre 0.1048), de même signe sur
13 cellules sur 15.

| | publié (3 sujets) | banc (sujet 0006) |
|---|---|---|
| moyenne, volume entier | 0.1048 | 0.1184 |

Deux causes possibles, non départagées ici : le banc ne porte que sur le sujet 0006
alors que le chiffre publié moyenne 3 sujets, et le chemin publié passe par
`infer_mmfm_unified` (recomposition par patches pondérés possible) là où le banc
utilise le crop centré. **L'écart ne change aucune conclusion de cette page** : toutes
les variantes empruntent le même chemin, donc les différences entre variantes sont
valides même si le niveau absolu diffère de 13 % du chiffre publié.

---

## Le banc VAE historique (`results/benchmark_vae/`) est invalide — deux causes vérifiées

C'est ce banc qui a servi à choisir MedVAE parmi AEKL / Pythae / VQ-VAE / RHVAE / MAISI.
Deux défauts, chacun vérifié directement dans le code :

1. **Il alimente MedVAE en [0, 1]** (`src/vae3d/benchmark_vae.py:77-79`, commentaire
   « Percentile normalization → [0, 1] ») alors que la recette amont ET la production
   utilisent [−1, 1] (`MonaiNormalize(mean=0.5, std=0.5)`). Le modèle est donc évalué
   hors de sa distribution d'entraînement.
2. **`PatchedVAE` laisse 14.89 % des voxels sans aucun patch.** `_get_patches`
   (`src/utils/patched_vae.py:106`) itère `range(0, h - ph + 1, sh)` et ne couvre donc
   jamais la queue de chaque axe ; le bloc dit « de bord » n'ajoute qu'un seul patch de
   coin. Simulé exactement sur la géométrie du banc (364×436×364, patch 112×128×80,
   recouvrement 0.25 → pas 84×96×60) : **81 patches, 8 601 152 voxels (14.89 %) jamais
   écrits**, qui sortent à zéro et entrent quand même dans les métriques.

Aucun des deux ne touche la production (qui passe par `models/tiled_vae.py`), mais
**les CSV de `results/benchmark_vae/` ne peuvent pas départager deux architectures** —
ils donnent d'ailleurs l'inverse de la décision de production sur le choix du
checkpoint. Le présent banc les remplace pour cette question.

## Réserves

- **Un seul sujet** (0006) sur les 3 disponibles. Les écarts entre variantes sont
  mesurés sur 15 volumes appariés, ce qui les rend fiables ; le NIVEAU absolu ne l'est
  qu'à ±13 % (voir ci-dessus).
- Le banc `native05` (encodage à 0.5 mm, sans rééchantillonnage) n'est pas dans cette
  campagne. Il bornerait le terme de 0.0326.
- Aucune de ces mesures ne dit ce que le score de bout en bout deviendrait : le
  2026-09-02 a établi qu'un gain de représentation de −0.0072 s'était transmis en
  **+0.0004** sur le score. Un gain de plafond n'est pas un gain de score.
