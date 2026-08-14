# MMFM 3D — Architecture unifiée

Multi-Marginal Flow Matching pour la translation de champ magnétique (Task 3,
any-to-any, ex. 0.1T↔7T). Le projet compare **trois variantes du modèle de
flow**, toutes entraînées/évaluées par le **même code** — seule
l'architecture du réseau (et, pour l'INR, la représentation latente
elle-même) change. Vectorisé et UNet opèrent sur un latent MedVAE gelé ;
l'INR opère directement sur le volume brut (voir ligne INR ci-dessous).

| Variante | Statut | Description |
|---|---|---|
| **vectorisée** (`mmfm3d_vectorized`) | active | MLP résiduel opérant sur le latent MedVAE aplati |
| **UNet** (`mmfm3d_unet`) | active | UNet 3D spatial (MONAI `DiffusionModelUNet`) opérant sur le latent MedVAE natif |
| **INR** (`mmfm3d_inr`) | active | SIREN+hypernetwork méta-appris, opère sur le volume brut (pas de MedVAE) — voir `docs/MMFM_INR_STATE_OF_THE_ART.md` |

Le champ magnétique est l'axe temporel continu du flow (`0.1T→1.5T→3T→5T→7T`
mappé sur `[0,1]`), le contraste (T1W/T2W/T2FLAIR) est la classe
conditionnante. Couplage OT-CFM réel entre marginales adjacentes
(`torchcfm.ExactOptimalTransportConditionalFlowMatcher`).

## Code : un seul point de variation légitime

```
src/cfm/mmfm_core.py       # boucle d'entraînement + inférence partagée : sampler
                            #   multi-marginal, couplage OT-CFM, pertes (L1) +
                            #   régularisateurs (cycle/edge-consistency), EMA,
                            #   optimizer/scheduler, checkpointing, logging
src/cfm/arch_vector.py     # ArchAdapter pour la variante vectorisée
src/cfm/arch_unet.py       # ArchAdapter pour la variante UNet
src/cfm/arch_inr.py        # ArchAdapter pour la variante INR (réutilise VectorMMFM comme flow)
src/cfm/train_mmfm_unified.py   # CLI unique : --method mmfm3d_vectorized|mmfm3d_unet|mmfm3d_inr
```

Chaque architecture fournit un `ArchAdapter` (dataclass dans `mmfm_core.py`) :

```python
@dataclass
class ArchAdapter:
    name: str
    build_model: Callable[[], nn.Module]
    make_model_fn: Callable[[nn.Module], Callable[[z_t, z_src, t, y], v_t]]
    prep_latent: Callable[[vae, z], Tuple[Tensor, Any]]      # VAE-latent -> espace modèle
    restore_latent: Callable[[Tensor, Any], Tensor]           # espace modèle -> VAE-latent
    checkpoint_key_remap: Callable[[dict], dict]              # shim de compat state_dict
    arch_meta_dict: Callable[[], dict]                        # métadonnées stockées au checkpoint
    cache_prebakes_prep: bool
    build_cache_dataset: Callable[[Path, Path, dict], Any]
    validate_cache_shape: Callable[[Any, Tuple[int, ...]], None]
    default_cache_root: str
```

`prep_latent`/`restore_latent` s'appliquent une seule fois par échantillon
(juste après `vae.encode()` / juste avant `vae.decode()`) — le reste (sampler,
couplage OT-CFM, intégration Euler, `cycle_rollout_vector`,
`Sobel3D`/`edge_consistency_loss`) reste en espace modèle du début à la fin et
est **entièrement partagé**, sans aucune connaissance de l'architecture.

### Ajouter une nouvelle variante (4e architecture)

Marche à suivre générale, telle qu'appliquée pour l'INR :

1. Créer `src/cfm/arch_<nom>.py` avec une fonction `make_adapter(cfg, latent_shape,
   n_classes) -> ArchAdapter`, sur le modèle de `arch_vector.py`/`arch_unet.py`/`arch_inr.py`.
   Si le latent produit est un vecteur plat, `arch_vector.build_vector_mmfm` (donc `VectorMMFM`)
   est directement réutilisable comme modèle de flow — aucune nouvelle architecture de flow à
   écrire (c'est ce que fait l'INR).
2. Ajouter `"mmfm3d_<nom>"` aux choix `--method` dans `train_mmfm_unified.py` et
   au dispatch `_resolve_arch_module()` dans `mmfm_core.py`.
3. Créer `configs/mmfm/<nom>.yaml` (mêmes blocs `data:`/`train:`/`inference:` que
   `vectorized.yaml`/`unet.yaml`, pour rester comparable).
4. Si la variante n'utilise pas MedVAE (comme l'INR — voir
   `docs/MMFM_INR_STATE_OF_THE_ART.md` section (C)) : `vae: {vae_type: identity}`
   (`models/vae_wrappers.py::IdentityVAEWrapper`) laisse `mmfm_core.py` totalement inchangé, le
   vrai travail par volume se fait dans `prep_latent`/`restore_latent` de l'adapter.

L'essentiel (`mmfm_core.py`, sampler, couplage OT-CFM, EMA, checkpointing) reste partagé sans
connaissance de l'architecture — mais `src/cfm/infer_mmfm_unified.py` (pipeline d'inférence pleine
résolution avec patches/blending, utilisé pour l'évaluation Task 3) a lui aussi besoin d'une petite
branche de dispatch dans `_build_model()` (comme pour `arch_vector`/`arch_unet`) : le contrat
adapter/vae y est déjà générique, seule la construction du modèle est câblée en dur par méthode.
`src/evaluation/evaluate.py` a de même besoin que le nom de la méthode soit ajouté à
`SUPPORTED_METHODS` (liste `choices` du CLI) si on veut l'évaluer via l'évaluateur officiel.

## Configs et sorties

```
configs/mmfm/
  vectorized.yaml   unet.yaml   inr.yaml   inr_backbone.yaml   field_norm_stats.json

outputs/mmfm/
  vectorized/{weights,train_metrics.jsonl,predictions}/
  unet/{weights,train_metrics.jsonl,predictions}/
  inr/{weights,train_metrics.jsonl,predictions}/          # flow — consomme le backbone ci-dessous
  inr_backbone/{weights,train_metrics.jsonl}/              # backbone INR gelé, pas de predictions/
  latent_cache/{vectorized,unet,inr}/
```

`inr_backbone/` est la seule exception à la règle "un dossier = une architecture de flow" : c'est la
sortie d'un entraînement séparé (analogue à MedVAE), consommé par `arch_inr.py` via
`cfg['inr_backbone']['checkpoint']`, pas un modèle de flow lui-même — pas de sous-dossier
`predictions/`.

Convention : `configs/mmfm/{vectorized,unet,inr,inr_backbone}.yaml` sont **édités sur place**
pour chaque nouvelle phase d'entraînement (pas de copie `_v2`/`_ext`/`_test`).
`outputs/mmfm/<arch>/weights/` contient **un seul checkpoint courant**,
écrasé à chaque nouvelle phase — pas d'accumulation. L'historique des
expériences reste dans `git log` (configs) et
`results/mmfm/comparison_<date>_<slug>/` (résultats mesurés — voir plus bas).

## Usage

```bash
# Entraînement
PYTHONPATH=src python src/cfm/train_mmfm_unified.py \
    --method mmfm3d_vectorized --config configs/mmfm/vectorized.yaml --env local
PYTHONPATH=src python src/cfm/train_mmfm_unified.py \
    --method mmfm3d_unet --config configs/mmfm/unet.yaml --env local \
    --resume outputs/mmfm/unet/weights/model_final.pth --resume_weights_only

# Inférence pleine résolution (patches + blending), batch Task 3
PYTHONPATH=src python src/cfm/infer_mmfm_unified.py \
    --config configs/mmfm/vectorized.yaml \
    --checkpoint outputs/mmfm/vectorized/weights/model_final.pth \
    --output_dir outputs/mmfm/vectorized/predictions \
    --split Training_prospective --modalities T1W \
    --norm_mode field_fixed --center_crop_only \
    --field_norm_stats configs/mmfm/field_norm_stats.json

# Évaluation
PYTHONPATH=src python src/evaluation/evaluate.py \
    --method mmfm_v2 --task task3 --modality T1W \
    --pred-dir outputs/mmfm/vectorized/predictions/task3/T1W \
    --output-csv results/mmfm/comparison_<date>/task3_vectorized_T1W.csv
```

## Résultats de référence

La comparaison à trois architectures, à code strictement identique (même dataloader, sampler,
couplage OT-CFM, loss, pipeline d'inférence — seule l'architecture diffère) est documentée dans
`results/mmfm/comparison_20260801_final/manifest.md` (checkpoints de production, 20 paires × 3
sujets, T1W, évaluateur officiel) :

> ⚠️ **Ces chiffres sont ceux de l'ancien pipeline 2mm.** La référence actuelle est la comparaison
> **à 1mm** (`results/mmfm/comparison_20260807_1mm/manifest.md`), reproduite plus bas.

| Modèle (ancien, @2mm) | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **Vectorisé** | **0.4288** | **0.8730** | **0.1407** |
| UNet | 0.4741 | 0.8721 | 0.1392 |
| INR (backbone corrigé, 2026-08-06) | 0.4869 | 0.8369 | 0.1751 |

### Référence actuelle — comparaison à 1mm (2026-08-10)

Mêmes latents (vectorisé/UNet), même budget (25000 itérations, batch=1), MedVAE **pré-entraîné**,
encodage par tuiles :

| Architecture @1mm | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| **Vectorisé** | **0.4353** | **0.8997** | **0.0983** |
| UNet | 0.4617 | 0.8955 | 0.1014 |
| INR (z=4096) | 0.6223 | 0.8002 | 0.2051 |

**⚠ Ces chiffres portent sur T1W SEUL.** Évalué le 2026-08-14 sur les trois
contrastes (vectorisé de production, même protocole) :

| contraste | nRMSE | SSIM | LPIPS |
|---|---|---|---|
| T1W | 0.4353 | 0.8997 | 0.0983 |
| **T2W** | **0.3376** | 0.8980 | **0.0911** |
| T2FLAIR | 0.3654 | 0.8949 | 0.0927 |
| **moyenne 3 contrastes** | **0.3794** | **0.8975** | **0.0941** |

**T1W est le contraste le plus DIFFICILE** : le projet a optimisé sur son pire
cas et le score réel toutes modalités est meilleur de 13 %. Surtout, le « mur du
7T » (nRMSE 0.7542 en T1W) **n'existe pas en T2W** (0.3620) — la difficulté du
haut champ est propre au contraste, pas au problème. Le volume de données ne
l'explique pas : T1W@7T est la classe la MIEUX dotée (235 volumes) et la plus
mauvaise. Détails : `results/mmfm/comparison_20260814_all_contrasts/manifest.md`.

Les comparaisons d'architectures ci-dessus n'ont été faites que sur T1W ; rien
ne garantit que le classement tienne sur les deux autres contrastes.

> **Mise à jour 2026-08-14** — ces chiffres du vectorisé proviennent du run
> réentraîné AVEC l'augmentation par flip. Jusqu'au 2026-08-13, `flip_lr_prob: 0.5`
> était déclaré dans sa config mais **silencieusement ignoré** (`FlatLatentCacheDataset`
> n'acceptait pas l'argument) : seul l'UNet en bénéficiait, ce qui invalidait
> l'affirmation « configs identiques hors du bloc `model:` ». Après correction et
> réentraînement, le flip s'avère **neutre** (0.4354 → 0.4353) et l'écart avec l'UNet
> est **inchangé** (-0.0264, 16/20 paires). Le classement ne dépendait pas du bug —
> c'est désormais mesuré. Voir `results/mmfm/comparison_20260814_vectorized_flip/`.
> L'INR, lui, ne peut PAS recevoir cette augmentation : son latent est un vecteur de
> modulation global sans structure spatiale (`arch_inr.py` lève une erreur).
| INR (z=129024) | 0.6383 | 0.7966 | 0.2061 |

- Le **vectorisé reste le meilleur**, y compris à 1mm (16/20 paires devant l'UNet en nRMSE), tout en
  étant 8x plus rapide et 5x plus léger. La recommandation antérieure de basculer sur l'UNet est
  **annulée**.
- La migration 2mm→1mm apporte **+0.023/+0.027 de SSIM et −27%/−30% de LPIPS** aux deux
  architectures MedVAE, à nRMSE quasi constant.
- L'**INR est la seule architecture dégradée** par le passage à 1mm (nRMSE +0.135). Porter `latent_dim`
  à 129024 (= taille du latent MedVAE @1mm, ×32) **ne corrige rien** : nRMSE 0.6223 → 0.6383, pour 21×
  plus de paramètres et ~20h de calcul. Le goulot n'est pas `latent_dim` mais
  `modulation_dim = num_couches × hidden_dim` = 1536 — le nombre de valeurs qu'un SIREN à modulation
  par décalage reçoit par volume, **indépendamment de `latent_dim` et de la résolution**. Y remédier
  demanderait une modulation spatialement variable (réseau convolutif), donc renoncer à l'approche INR.
  **Piste close** — voir `results/mmfm/comparison_20260807_1mm/manifest.md`.

Le vectorisé reste la référence globale. L'INR a une histoire en trois temps, documentée en détail
dans le manifest et la mémoire projet : un premier backbone (bugué vs la référence NOIR — voir
`docs/MMFM_INR_STATE_OF_THE_ART.md` section 6) donnait un nRMSE plus bas (0.4566) mais une
représentation nettement moins fidèle (SSIM auto-reconstruction ~0.68 vs MedVAE ~0.91) et gagnait sur
les cibles →5T/→7T ; après correction de trois écarts d'implémentation réels (identifiés en comparant
le code à la référence officielle, pas seulement au papier), la fidélité de représentation a nettement
progressé (SSIM auto-reconstruction ~0.83) mais le nRMSE Task 3 s'est dégradé et l'avantage sur
→5T/→7T a disparu. L'hypothèse initiale (flow sous-dimensionné pour un latent "8x plus grand") s'est
révélée fausse à vérification : le latent INR (4096-d) est en réalité 4x plus PETIT que le latent
MedVAE aplati du vectorisé (16128-d), et doubler la capacité du flow (`hidden_dim` 1024→2048) a
dégradé les résultats plutôt que de les améliorer (voir manifest, section "l'hypothèse testée et
réfutée"). Le facteur limitant n'est donc probablement ni la représentation ni la capacité du flow,
mais la structure/régularité de l'espace latent INR lui-même (chaque `z` fitté par quelques pas de
gradient par volume, sans régularisateur de continuité, contrairement au latent MedVAE façonné par
KL+perceptuel). Cette hypothèse a été directement vérifiée (2026-08-06) : la rugosité d'interpolation
croisée-champ de l'INR est ~2.9x sa propre rugosité même-champ, contre ~1.3x pour MedVAE — diagnostic
confirmé (voir manifest, section "régularisation de lissage sur z"). Un correctif (pénalité de
Hutchinson sur le Jacobien de `z`) a été implémenté et testé en balayage, mais aucune valeur de poids
testée n'améliore la rugosité mesurée sans faire s'effondrer la dépendance de `decode(z)` à `z` — le
retrain complet n'a pas été lancé (porte de smoke-test non franchie). Le résultat de production INR
reste 0.4869/0.8369/0.1751 ; pistes de correctif non testées documentées dans le manifest/la mémoire
projet.

Historique antérieur (avant la comparaison finale et l'ajout de l'INR) :
`results/mmfm/analysis/protocol_20260729_unet_mm_baseline/` et
`results/mmfm/analysis/protocol_20260730_harmonized_comparison/`.

## ⚠️ Résolution de travail — goulot identifié (2026-08-07)

Les résultats ci-dessus (vectorisé 0.4288 / UNet 0.4741 / INR 0.4869) sont tous obtenus à
**96×112×96 @ 2mm**, alors que la grille native du challenge est **364×436×364 @ 0.5mm**. Une étude de
capacité de représentation multi-résolution (voir `results/mmfm/comparison_20260801_final/manifest.md`,
section « LA RÉSOLUTION EST LE VRAI GOULOT ») a établi que **cette résolution de travail est le
facteur limitant principal** du projet, devant l'architecture du flow :

| Représentation (SSIM / nRMSE) | 2mm | 1mm | 0.5mm (natif) |
|---|---|---|---|
| MedVAE pré-entraîné (poids bruts) | 0.800 / 0.187 | 0.915 / 0.082 | **0.948 / 0.049** |
| MedVAE fine-tuné L1 (production actuelle) | 0.838 / 0.146 | 0.879 / 0.059 | 0.884 / 0.035 |
| INR (SIREN, z=4096) | 0.834 / 0.205 | 0.808 / 0.209 | — |

Trois implications directes pour cette architecture :

1. **Le checkpoint MedVAE de production (fine-tuné L1) n'est pas le meilleur disponible** : le
   pré-entraîné non modifié le dépasse nettement dès 1mm. Notre fine-tuning L1 plafonne le SSIM à
   ~0.89 à toute résolution. Piège : à 2mm la hiérarchie est **inversée**, donc toute comparaison de
   représentations faite à 2mm est trompeuse.
2. **MedVAE ne peut pas encoder un volume 1mm en une passe** (attention quadratique au goulot : OOM
   >121GB à 192×224×192). Le traitement par patches 64³ est obligatoire au-delà de 2mm — c'est aussi
   la recette officielle MedVAE.
3. **Le flow vectorisé ne survit pas à la montée en résolution** : le latent MedVAE aplati passe de
   16k (2mm) à 129k (1mm) à 1.03M (0.5mm) valeurs. `VectorMMFM` devient inopérant ; **l'UNet spatial
   devient l'architecture naturelle**, et à 0.5mm son latent (96×112×96) a exactement la taille des
   volumes qu'il traite déjà. L'INR, dont le `z` est figé à 4096 quelle que soit la résolution, ne
   profite pas de ce changement (il régresse même légèrement).

**La hiérarchie des trois méthodes documentée plus haut est donc conditionnée à une résolution
aujourd'hui identifiée comme sous-optimale** et devra être réétablie à la résolution cible retenue.

## Résultats à 1mm (2026-08-09) — nouvelle référence

Après la migration à 1mm (voir section précédente), les deux architectures à latent MedVAE ont été
réentraînées et comparées **à équité stricte** : mêmes latents (le cache vectorisé est le cache UNet
aplati, cf. `src/cfm/convert_unet_cache_to_flat.py`), mêmes 25000 itérations, même batch=1 — seule
l'architecture du flow diffère. Détail complet :
`results/mmfm/comparison_20260807_1mm/manifest.md`.

| Modèle | nRMSE | SSIM | LPIPS | Mémoire | Vitesse |
|---|---|---|---|---|---|
| **Vectorisé @1mm** | **0.4353** | **0.8997** | **0.0983** | 8.6 GB | 3.48 it/s |
| UNet @1mm | 0.4617 | 0.8955 | 0.1014 | 46.5 GB | 0.42 it/s |
| *Vectorisé @2mm (ancienne réf., 50k iters)* | *0.4288* | *0.8730* | *0.1407* | — | — |

**Le vectorisé reste la meilleure architecture**, y compris à 1mm : il gagne sur les trois métriques,
sur 16/20 paires en nRMSE, et il est 8× plus rapide et 5× plus léger que l'UNet.

⚠️ **Correction** : la section précédente annonçait que « l'UNet spatial devient l'architecture
naturelle » à 1mm, le flow vectorisé devant devenir inopérant avec un latent aplati de 129k
dimensions. **C'est faux** — mesuré, le vectorisé y est meilleur ET moins coûteux. Cette prédiction
n'a pas résisté à l'expérience.

Apport réel de la résolution (vectorisé, 2mm→1mm) : **SSIM +0.027, LPIPS −30 %**, nRMSE quasi stable
(0.4288 → 0.4354) — avec deux fois moins d'itérations que l'ancienne référence. Le gain est
perceptuel et structurel, cohérent avec la capacité de représentation mesurée (SSIM
d'auto-reconstruction 0.800 @2mm → 0.915 @1mm).

**L'INR n'a pas été réentraîné à 1mm** (voir les réserves du manifest) : son `z` figé à 4096
dimensions ne profite pas de la résolution — sa capacité de représentation s'y dégrade même
(0.834 → 0.808).
