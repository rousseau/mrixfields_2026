# ⚠️ CES RÉSULTATS SONT INVALIDES — ne pas s'en servir pour départager deux VAE

**Marqué le 2026-09-07.** Deux défauts, chacun suffisant, vérifiés dans le code.

## 1. Le banc alimentait les VAE en [0, 1] au lieu de [−1, 1]

`src/vae3d/benchmark_vae.py::load_prospective_volume` renvoyait un volume dans [0, 1]
(commentaire d'origine : « Percentile normalization → [0, 1] »), alors que :

- le contrat de la maison est [−1, 1] — `src/models/vae_base.py:74`, « single-channel
  3D volume, range [-1, 1] » ;
- `src/common/io.py::normalize_volume`, qu'utilise tout le reste du dépôt, applique
  bien `*2 - 1` ;
- la recette amont de MedVAE fait `ScaleIntensity(0,1)` **puis**
  `Normalize(mean=0.5, std=0.5)` (`medvae/utils/loaders.py::load_mri_3d`).

Les modèles étaient donc évalués **hors de leur distribution d'entraînement**. Mesuré
sur un patch 64³ centré de T1W@3T sujet 0006 à 1 mm, MedVAE 4_1_3d pré-entraîné :
entrée [0,1] → nRMSE **0.1357** ; entrée [−1,1] → nRMSE **0.1120**. Soit **+21 %**.

## 2. `PatchedVAE` laissait 14.89 % des voxels sans aucun patch

`src/utils/patched_vae.py::_get_patches` itérait `range(0, L - P + 1, S)` et n'atteignait
jamais la queue d'un axe ; le bloc dit « boundary » qui suivait ne rattrapait qu'un seul
patch de coin (sa condition était toujours vraie et ses indices étaient des constantes).

Simulé exactement sur la géométrie du banc (364×436×364, patch 112×128×80,
recouvrement 0.25 → pas 84×96×60) : **81 patches, 8 601 152 voxels (14.89 %) jamais
écrits**, sortant à exactement zéro et entrant quand même dans MAE / MSE / SSIM /
nRMSE / LPIPS. Et `benchmark_vae.py:244` fait `use_patched = not is_rhvae`, donc
**tous** les VAE à latent spatial étaient concernés.

Les deux défauts sont corrigés dans le code au 2026-09-07 (couverture désormais
vérifiée à 0.00 % de voxels manquants, 120 patches), mais **les fichiers ci-dessous ont
été produits avant** et n'ont pas été régénérés.

## Fichiers invalides

- `metrics/benchmark_results.csv`
- `metrics/benchmark_results_legacy.csv`
- `visuals/comparison_*.png` (échelle d'affichage [0,1] en plus)
- `benchmark_visuals/`
- et, par conséquence, le classement des 7 VAE de `docs/VAE_PERFORMANCE.md`, dont la
  recommandation « Pythae_VAE is recommended baseline for CFM » n'a **pas** d'autre
  source que ce banc.

## Fichiers NON concernés (à ne pas marquer)

Produits par `eval_representation_multires.py` et `compare_medvae_checkpoints.py`, qui
passent par `load_nifti_volume(normalize=True)` — donc [−1,1] — et n'utilisent pas
`PatchedVAE` :

- `metrics/medvae_representation_multires.csv`
- `metrics/medvae_lpips_vs_pretrained.csv`
- `metrics/medvae_pretrained_vs_finetuned.csv`

## Ce qui les remplace

Pour la question « MedVAE est-il bien paramétré, et quelle variante prendre »,
la référence est désormais `src/cfm/bench_representation.py` et
`results/mmfm/representation_20260906/manifest.md` : 23 variantes mesurées sous la
géométrie de production, avec un témoin sans VAE qui isole le coût du protocole.
