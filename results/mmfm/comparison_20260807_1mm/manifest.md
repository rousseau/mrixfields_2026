# Comparaison des trois architectures MMFM à 1mm — 2026-08-07/10

Première comparaison des trois architectures **à résolution égale et à budget égal**, après la
migration du pipeline de 2mm vers 1mm (motivée par
`results/mmfm/comparison_20260801_final/manifest.md`, section « LA RÉSOLUTION EST LE VRAI GOULOT »).

## Protocole — ce qui est strictement identique

- **Résolution** : 192×224×192 @ 1mm (contre 96×112×96 @ 2mm auparavant).
- **Représentation** : MedVAE **pré-entraîné** (notre ancien fine-tuning L1 est abandonné :
  il plafonnait le SSIM d'auto-reconstruction à ~0.88 à toute résolution contre 0.915 @1mm et
  0.948 @0.5mm pour le pré-entraîné). Encodage par tuiles 2×2×2 de 96×112×96 + marge de contexte
  de 16 voxels (`src/models/tiled_vae.py`) — obligatoire car MedVAE ne peut pas encoder un volume
  1mm en une passe (attention quadratique au goulot : OOM >121 GB).
- **Latents** : vectorisé et UNet partagent **exactement les mêmes latents** — le cache aplati du
  vectorisé est une conversion (reshape) du cache spatial de l'UNet
  (`src/cfm/convert_unet_cache_to_flat.py`), pas un ré-encodage. L'INR a son propre latent `z`
  (4096-d) par construction (il n'utilise pas MedVAE).
- **Budget** : 25000 itérations, batch_size=1, mêmes lr/warmup/EMA/grad_clip pour les trois.
- **Évaluation** : évaluateur officiel, 20 paires × 3 sujets, T1W, prédictions ré-interpolées vers
  la grille native 364×436×364 @0.5mm.

## Résultats

| Architecture | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **Vectorisé @1mm** | **0.4354** | **0.8995** | **0.0985** |
| UNet @1mm | 0.4617 | 0.8955 | 0.1014 |
| INR @1mm | 0.6223 | 0.8002 | 0.2051 |

CSV : `task3_{vectorized,unet,inr}_1mm_T1W.csv`. Figures : `figures/`.

### Évolution 2mm → 1mm (mêmes architectures)

| | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| Vectorisé | 0.4288 → 0.4354 (+0.007) | 0.8730 → 0.8995 (**+0.027**) | 0.1407 → 0.0985 (**−30%**) |
| UNet | 0.4741 → 0.4617 (−0.012) | 0.8721 → 0.8955 (**+0.023**) | 0.1392 → 0.1014 (**−27%**) |
| INR | 0.4869 → 0.6223 (**+0.135**) | 0.8369 → 0.8002 (**−0.037**) | 0.1751 → 0.2051 (**+17%**) |

### nRMSE par champ cible @1mm

| Cible | Vectorisé | UNet | INR |
|---|---|---|---|
| →0.1T | 0.2697 | 0.3017 | 0.5635 |
| →1.5T | 0.3293 | 0.3526 | 0.5575 |
| →3T | 0.3140 | 0.3299 | 0.5396 |
| →5T | 0.5072 | 0.5286 | 0.6250 |
| →7T | 0.7568 | 0.7956 | 0.8259 |

## Trois conclusions

1. **Le vectorisé reste la meilleure architecture, y compris à 1mm** — il gagne sur les trois
   métriques et sur 16/20 paires en nRMSE face à l'UNet. La prédiction selon laquelle un MLP sur
   latent aplati de 129k dimensions s'effondrerait à cette résolution était **fausse** : il est en
   outre 8x plus rapide (3.5 vs 0.42 it/s) et 5x plus léger (8.6 vs 46.5 GB) que l'UNet. La
   recommandation antérieure de basculer sur l'UNet est donc annulée.

2. **La migration à 1mm est un gain net pour les deux architectures à base de MedVAE** : SSIM
   +0.023/+0.027 et LPIPS −27%/−30%, pour un nRMSE quasi inchangé. Le nRMSE est précisément la
   métrique qui récompense le lissage (cf. le diagnostic de la perte L1) : il est attendu qu'une
   représentation plus nette n'y gagne pas, pendant que SSIM et LPIPS progressent nettement. À noter
   que le run 1mm n'a eu que 25000 itérations contre 50000 pour l'ancienne référence 2mm.

3. **L'INR est la seule architecture que le passage à 1mm DÉGRADE**, et fortement (nRMSE +0.135,
   SSIM −0.037, LPIPS +17%). C'est la conséquence directe et prévue de sa capacité fixe : son `z`
   reste à **4096 dimensions quelle que soit la résolution**, alors que le latent MedVAE passe de
   16128 (@2mm) à 129024 (@1mm). L'écart de capacité passe donc de **3.9× à 31.5×**. La propriété
   d'invariance à la résolution de NOIR/MedFuncta permet d'**évaluer** à n'importe quelle
   résolution ; elle n'apporte aucune capacité pour **représenter** davantage. Le backbone lui-même
   s'est pourtant bien entraîné (loss finale 0.0395, meilleure que les 0.040 du backbone 2mm) — ce
   n'est pas un défaut d'exécution, c'est une limite structurelle.

   Visuellement (voir `figures/`), les prédictions INR @1mm présentent des artefacts massifs
   (plages saturées, structures anatomiques perdues) absents à 2mm : le flow doit interpoler dans
   un espace latent devenu très insuffisant pour le signal à restituer.

   **Note** : le commentaire de `configs/mmfm/inr_backbone.yaml` justifiant `latent_dim: 4096` par
   « la taille du latent MedVAE » repose sur une valeur erronée — 4096 provient d'un encodage
   factice (1,16,16,16), alors que le vrai latent MedVAE @2mm fait 16128. L'INR était donc déjà
   3.9× sous-dimensionné à 2mm, et non « à budget égal » comme annoncé.

## Configuration de référence recommandée

**Vectorisé, 1mm, MedVAE pré-entraîné, encodage tuilé** — `configs/mmfm/vectorized.yaml`.
Pour aller plus loin sur l'INR il faudrait augmenter substantiellement `latent_dim` (ordre de
grandeur : 129024 pour égaler MedVAE @1mm), ou passer à une modulation directe par couche façon
MedFuncta — pas un simple ré-entraînement.

---

# Expérience `latent_dim` de l'INR : 4096 → 129024 (2026-08-10/11)

Objectif : tester si la **capacité du latent `z`** est le facteur limitant de l'INR, en portant
`latent_dim` à 129024 — exactement la taille du latent MedVAE @1mm (48×56×48).

## Analyse préalable : le vrai goulot n'est pas `latent_dim`

Un SIREN à modulation par décalage reçoit, pour décrire un volume, exactement
`modulation_dim = num_hidden_layers × hidden_dim` = 6 × 256 = **1536 nombres**. Cette quantité est
**indépendante de `latent_dim` ET de la résolution**. Le hypernetwork comprime donc
`latent_dim → hyper_hidden_dim → modulation_dim`, soit avec latent_dim=129024 :
**129024 → 512 → 1536**, un goulot de **252:1** (le projet avait déjà observé qu'un goulot de 64:1
DÉGRADAIT la fidélité — cf. `../comparison_20260801_final/inr_reconstruction_capacity.csv`).

### Coût mesuré des alternatives qui lèvent réellement le goulot

Sonde directe (3 pas de `meta_train_step` réels, volumes 192×224×192, batch=2, 16384 points) :

| Config | params | `modulation_dim` | mémoire | s/iter | 50k itérations |
|---|---|---|---|---|---|
| **A — littérale** (6×256, hyper 512) | 67.2M | **1536** (inchangé) | 11.4 GB | 0.98 | **13.7 h** |
| B — modulation ×4 (12×512, hyper 2048) | 279.7M | 6144 | 46.0 GB | 5.98 | 83 h |
| C — modulation ×8 (12×1024, hyper 4096) | 590.4M | 12288 | 92.2 GB | 9.68 | 134 h |

Même la config C, à 134 h de calcul, plafonnerait **10× en dessous** des 129024 de MedVAE.
**Conclusion structurelle** : on ne peut pas égaler la capacité d'un latent spatial avec un latent
global modulant par décalage, quel que soit le budget. Il faudrait une modulation *spatialement
variable* (champ de modulation par voxel produit par un réseau convolutif) — ce qui revient à
réintroduire un VAE convolutif. **Config A retenue** (choix utilisateur), avec l'avertissement que
c'est très probablement un no-op sur la capacité utile.

## Résultats d'entraînement (config A, latent_dim=129024)

| | latent_dim=4096 | latent_dim=129024 | Δ |
|---|---|---|---|
| Backbone : params | 3.22M | **67.18M** (×21) | — |
| Backbone : loss finale (50k) | 0.0395 | **0.0374** | −5 % |
| Flow : params | ~47M | **430.5M** (×9) | — |
| Flow : loss finale (25k) | 0.0024 | **0.0012** | −50 % |

Lecture : **32× plus de dimensions de latent n'achètent que 5 % de loss de backbone.** C'est
exactement le profil attendu d'un goulot en aval : `z` encode mieux, mais ne peut transmettre que
1536 nombres au SIREN. La loss de flow chute de moitié, mais elle est calculée dans un espace
latent 32× plus grand — elle n'est donc pas comparable d'une variante à l'autre (leçon déjà apprise
sur ce projet : la loss d'entraînement est un proxy trompeur du résultat Task 3).

## Résultat Task 3 — l'hypothèse est RÉFUTÉE

| Architecture @1mm | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **Vectorisé** | **0.4354** | **0.8995** | **0.0985** |
| UNet | 0.4617 | 0.8955 | 0.1014 |
| INR (z=4096) | 0.6223 | 0.8002 | 0.2051 |
| **INR (z=129024)** | **0.6383** | **0.7966** | **0.2061** |

**Multiplier `latent_dim` par 32 dégrade légèrement le résultat** (nRMSE +0.016, SSIM −0.004,
LPIPS +0.001) — au prix de 21× plus de paramètres de backbone, 9× plus de paramètres de flow, et
~20 h de calcul. La capacité de `z` n'est donc **pas** le facteur limitant de l'INR : c'est bien
`modulation_dim` (1536), comme l'analyse préalable le prévoyait.

Par champ cible (nRMSE) :

| Cible | Vectorisé | UNet | INR z=4096 | INR z=129024 |
|---|---|---|---|---|
| →0.1T | 0.2697 | 0.3017 | 0.5635 | 0.5926 |
| →1.5T | 0.3293 | 0.3526 | 0.5575 | 0.5901 |
| →3T | 0.3140 | 0.3299 | 0.5396 | 0.5835 |
| →5T | 0.5072 | 0.5286 | 0.6250 | 0.6486 |
| →7T | 0.7568 | 0.7956 | 0.8259 | **0.7765** |

Le z=129024 est pire sur 4 des 5 champs cibles ; il n'améliore que →7T (0.8259 → 0.7765), sans que
cela compense la dégradation ailleurs.

**Note méthodologique importante** : la loss d'entraînement du flow a été **divisée par deux**
(0.0024 → 0.0012) alors que le résultat Task 3 se dégrade. C'est la troisième fois sur ce projet
que la loss d'entraînement induit en erreur (après l'extension à 375k itérations et le passage à
batch_size=16). **Ne jamais conclure d'une loss d'entraînement sur ce pipeline** — ici la loss n'est
en outre pas comparable entre variantes, puisqu'elle est calculée dans un espace latent 32× plus
grand.

## Conclusion sur l'INR

L'INR est structurellement inadapté à cette tâche à haute résolution, pour une raison précise et
désormais mesurée : un SIREN à modulation par décalage transmet `num_couches × hidden_dim` nombres
par volume, indépendamment de `latent_dim` et de la résolution. Les trois leviers testés ont tous
échoué ou plafonné :

1. **Corriger l'implémentation** (3 bugs vs le code NOIR) : SSIM d'auto-reconstruction 0.68 → 0.83,
   mais Task 3 dégradé.
2. **Régulariser le latent** (pénalité de Hutchinson) : aucune plage de poids utilisable entre
   « sans effet » et « collapse ».
3. **Augmenter `latent_dim`** (×32) : dégradation légère, confirmant que le goulot est ailleurs.

La seule voie restante serait une **modulation spatialement variable** (champ de modulation par
voxel produit par un réseau convolutif, façon 3D MTransINR) — c'est-à-dire réintroduire un encodeur
convolutif, donc renoncer à ce qui distinguait l'approche INR. **Piste close** sauf décision
contraire explicite.
