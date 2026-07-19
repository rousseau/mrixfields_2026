# MMFM-UNet 3D multi-marginal

Cette méthode implémente une **vraie formulation Multi-Marginal Flow Matching
(MMFM)** pour la Task 3 du challenge MRIxFields 2026 : traduction *any-to-any*
de champs magnétiques sur des données rétrospectives non appariées.

## Principe

Contrairement aux versions précédentes qui traitaient les 15 combinaisons
`(modalité, champ)` comme des classes discrètes indépendantes, cette version
sépare explicitement :

- **Classe conditionnante** = la **modalité / contraste** (T1W, T2W, T2FLAIR)  
  → embedding de classe sur `3` modalités.
- **Axe temporel du flow** = le **champ magnétique** ordonné  
  `0.1T < 1.5T < 3T < 5T < 7T` mappé uniformément sur `[0, 1]`.

L'intégration ODE se fait alors sur la trajectoire continue
`t_source → t_target`, ce qui préserve la structure ordonnée des champs et
garantit un modèle unique pour toutes les traductions.

## Formulation

Pour un couple source/cible de **mêmes modalité et sujet** (données non
appariées entre champs) :

1. Tirage uniforme d'une modalité `c ∈ {0,1,2}`.
2. Tirage d'un couple de champs **adjacents** `(i, i+1)` avec probabilité
   `1 - identity_prob`, sinon couple identique `(i, i)`.
3. Tirage d'un échantillon source `x_0` et d'un échantillon cible `x_1`
   dans les deux marginales correspondantes.
4. Couplage OT minibatch entre ces deux marginales adjacentes.
5. Interpolation conditionnée au temps `t` global du champ :
   `z_t = (1 - t) z_0 + t z_1 + σ ε`.
6. L'UNet prédit la vitesse `v_t(z_t, t, c, z_0)` où `z_0` est concaténé à
   `z_t` pour ancrer l'anatomie.

L'objectif est la perte CFM habituelle :

```text
L = E_{t, c, (z_0,z_1)} || v_θ(z_t, t, c, z_0) - (z_1 - z_0) ||²
```

## Architecture

- **Encodeur / décodeur** : MedVAE fine-tuné sur les 3 modalités, **frozen**.
- **Latent spatial** : `(1, 44, 54, 44)` pour un volume entier de
  `182 × 218 × 182` voxels à 1 mm.
- **UNet 3D** : `DiffusionModelUNet` MONAI avec
  - `channel_mult: [1, 2, 4]` (3 niveaux de down-sampling)
  - `model_channels: 128`
  - attention aux niveaux 2 et 3
  - `num_class_embeds: 3` (contraste)
  - concaténation de `z_source` en entrée (2 canaux : `z_t` + `z_0`)
- **Padding réversible** : le latent a `W = 54`, non divisible par 4 (contrainte
  UNet à 3 niveaux). On padde à 56 par zéros avant l'UNet et on recroppe à 54
  avant décodage. Aucune information anatomique n'est perdue.

## Cache de latents

Pour éviter de ré-encoder le VAE à chaque itération, les volumes entiers sont
pré-encodés une fois :

```text
python src/cfm/precompute_latents.py \
    --config configs/mmfm3d_multimarginal_medvae_run1.yaml \
    --env local
```

Le cache est invalidé automatiquement si le checkpoint VAE change
(`vae_id = <vae_type>_<hash8(checkpoint)>`).

L'entraînement lit directement les latents fp16 depuis le disque (précharge
optionnel en RAM).

## Fichiers clés

- [src/cfm/train_mmfm_unet_3d.py](../src/cfm/train_mmfm_unet_3d.py) —
  entraînement multi-marginal, padding/crop, intégration continue.
- [src/cfm/precompute_latents.py](../src/cfm/precompute_latents.py) —
  construction du cache de latents.
- [src/common/dataset.py](../src/common/dataset.py) — `LatentCacheDataset`.
- [src/cfm/infer_mmfm_unified.py](../src/cfm/infer_mmfm_unified.py) —
  inférence any-to-any (contrast + t_source → t_target).
- [configs/mmfm3d_multimarginal_medvae_run1.yaml](../configs/mmfm3d_multimarginal_medvae_run1.yaml) —
  configuration du run actuel.

## Lancement local (DGX GB10)

### 1. Pré-encoder les latents

```bash
PYTHONPATH=src python src/cfm/precompute_latents.py \
    --config configs/mmfm3d_multimarginal_medvae_run1.yaml \
    --env local
```

### 2. Entraîner le flow

```bash
PYTHONPATH=src python src/cfm/train_mmfm_unet_3d.py \
    --config configs/mmfm3d_multimarginal_medvae_run1.yaml \
    --env local
```

### 3. Inférer sur un sujet prospectif

```bash
PYTHONPATH=src python src/cfm/infer_mmfm_unified.py \
    --config configs/mmfm3d_multimarginal_medvae_run1.yaml \
    --env local \
    --checkpoint outputs/cfm3d/runs/mmfm3d_multimarginal_medvae_run1/weights/checkpoint_12000.pth \
    --source-subject ~/Data/MRIxFields_20260414/Training_prospective/P_T1W_1.5T_0006.nii.gz \
    --target-modality T1W \
    --target-field 7T \
    --output outputs/predictions/mmfm3d_multimarginal_medvae_run1/P_T1W_7T_0006.nii.gz
```

> Le script supporte aussi l'inférence sur les 3 sujets prospectifs avec toutes
> les paires source↔cible.

## Différences avec MMFM v1 vectorisé

| Aspect | MMFM v1 vectorisé | MMFM-UNet multi-marginal |
|--------|-------------------|--------------------------|
| Espace latent | aplati en vecteur | spatial 3D |
| Modèle | MLP résiduel | UNet 3D MONAI |
| Classes | 15 domaines `(mod, field)` | 3 contrastes |
| Axe temporel | discret (15 domaines) | continu (champ B0 ∈ [0,1]) |
| Couplage | OT entre 15 marginales | OT entre marginales adjacentes |
| Identité | non ancré | cas identité `identity_prob` + concat `z_0` |
| Résolution | crop 128×128×80 | volume entier 182×218×182 |

## Notes de conception

- **Pourquoi adjacent_only ?** Sur des données non appariées, apprendre
  directement des couplages entre champs éloignés (ex. 0.1T ↔ 7T) est mal
  posé. Le couplage entre marginales adjacentes fournit des trajectoires
  courtes et stables.
- **Pourquoi le cas identité ?** Il force le modèle à préserver l'anatomie
  quand `t_source = t_target`.
- **Pourquoi le cache ?** L'encodage MedVAE du volume entier (~9s/volume) serait
  le goulot de l'entraînement sans cache.
- **Pourquoi batch 1 ?** Le latent entier `(1, 44, 56, 44)` avec l'UNet V1
  (178M params) occupe ~35 GB en forward+backward. Batch 8 est impossible ;
  batch 2 dépasse la mémoire disponible.

## État actuel

- Reformulation multi-marginal implémentée.
- Cache latents en cours de construction sur `Training_retrospective`.
- Run 1 : 12 000 itères sur ~12h (DGX GB10).
