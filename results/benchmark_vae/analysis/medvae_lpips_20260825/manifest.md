# Perte perceptuelle sur MedVAE — la barre est franchie, largement

**Date** : 2026-08-25 (run terminé le 2026-08-24 à 16h30, 6000 pas, ~6h20)
**Verdict** : **le fine-tuning LPIPS bat le pré-entraîné sur les deux métriques,
aux deux résolutions, et sur les trois champs testés.** La porte posée avant le
lancement — « si le résultat ne dépasse pas 0.915 de SSIM à 1 mm, l'adoption
s'arrête là » — est franchie avec 0.9727.

---

## Le résultat

Auto-reconstruction par patches 64³ (recouvrement 0.5, fusion Hann), 3 sujets
`Training_prospective` x 3 champs, exactement le même protocole et les mêmes
volumes pour les trois variantes — une seule passe, trois modèles chargés
ensemble (`eval_representation_multires.py --checkpoints`).

| @1mm (192x224x192) | nRMSE | SSIM |
|---|---|---|
| pré-entraîné (référence de production) | 0.0819 | 0.9152 |
| fine-tuné L1 (2026-08-07, écarté) | **0.0592** | 0.8790 |
| **fine-tuné LPIPS** | 0.0682 | **0.9727** |

| @0.5mm (384x448x384) | nRMSE | SSIM |
|---|---|---|
| pré-entraîné | 0.0488 | 0.9481 |
| fine-tuné L1 | **0.0350** | 0.8837 |
| **fine-tuné LPIPS** | 0.0381 | **0.9825** |

Par champ, à 1 mm (SSIM) : 0.1T 0.9311 → **0.9874**, 3T 0.8992 → **0.9667**,
7T 0.9153 → **0.9640**. Le gain est le plus fort là où le pré-entraîné était le
plus faible (3T), et il ne se dégrade nulle part.

## Ce que la comparaison à trois montre, et que la comparaison à deux cachait

Le fine-tuning L1 de 2026-08-07 avait été écarté sur son SSIM (0.879 contre
0.915) et la conclusion tirée à l'époque était que « la recette de loss n'est pas
la cause dominante ». **Ce diagnostic était faux, et le contre-exemple est ici** :
la même recette de fine-tuning, sur les mêmes données, avec LPIPS + PatchGAN à la
place du L1, gagne 0.057 de SSIM sur le pré-entraîné là où le L1 en perdait 0.036.

Noter aussi que le L1 garde le MEILLEUR nRMSE des trois (0.0592) tout en ayant le
PIRE SSIM. C'est le compromis perception/distorsion dans sa forme la plus nue :
minimiser l'erreur quadratique produit l'espérance conditionnelle, donc du flou,
donc un bon nRMSE et une mauvaise structure. **Le LPIPS, lui, gagne sur les DEUX**
— ce qui n'était pas acquis et qui est le vrai résultat de ce run.

## Absence de contamination — vérifiée

Le fine-tuning tourne sur `split: retro_train` = `Training_retrospective`
(préfixe `R_`, 1056 sujets) ; l'évaluation porte sur `Training_prospective`
(préfixe `P_`, sujets 0006/0007/0009). Les identifiants numériques se recoupent
entre les deux cohortes mais le préfixe les distingue : `R_..._0006` et
`P_..._0006` ne sont pas le même sujet. Aucun volume d'évaluation n'a été vu à
l'entraînement.

## Ce que cela ne dit PAS

Ce run mesure la **capacité de représentation** (encoder puis décoder le même
volume), pas la performance Task 3. Adopter ce checkpoint impose de **régénérer
les deux caches de latents et de réentraîner les deux flows** (vectorisé et
UNet), soit plus d'une journée de calcul, et invalide tous les résultats en place
tant que ce n'est pas fait. C'est une décision à prendre explicitement.

Un élément la nuance sérieusement : l'évaluation des trois architectures du même
jour (`results/mmfm/qualitative_20260824/manifest.md`) montre que **80 % de
l'énergie de l'erreur Task 3 en T1W et T2FLAIR est une erreur d'échelle
d'intensité**, pas un défaut de représentation. Améliorer le VAE agit sur les
20 % restants. Le gain de netteté est réel — les modèles ne restituent qu'un
tiers de la finesse de la vérité — mais il ne faut pas en attendre un
effondrement du nRMSE.

## Fichiers

| chemin | contenu |
|---|---|
| `../../metrics/medvae_lpips_vs_pretrained.csv` | 54 lignes : 3 variantes x 3 sujets x 3 champs x 2 résolutions |
| `outputs/medvae/runs/medvae_finetune_lpips/train.log` | journal d'entraînement (jalons tous les 1000 pas) |
| `outputs/medvae/runs/medvae_finetune_lpips/eval.log` | sortie de la chaîne d'évaluation |
| `outputs/medvae/runs/medvae_finetune_lpips/weights/model_best.pth` | checkpoint retenu (pas 5000) |
| `configs/medvae_finetune_lpips.yaml` | recette : L1 + 1.0·LPIPS(2.5D) + 1e-6·KL + PatchGAN(w=0.5, start=1000) |
