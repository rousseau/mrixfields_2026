# Script de soumission Task 3

Ce document décrit le script dédié pour préparer une soumission MRIxFields 2026 Task 3.

Fichier script:
- src/submission/build_task3_submission.py

## Objectif

Le script convertit des prédictions Task 3 vers le format attendu par le challenge:

- structure: task3/<modalite>/<pair>/pred/*.nii.gz
- nommage: P_<MOD>_<TARGET_FIELD>_<ID>.nii.gz
- clip axial: [150:180] par défaut (shape finale attendue 364x436x30)
- option de création directe de task3.zip

Le script est compatible avec:

1. Fichiers au format officiel (ex: P_T1W_7T_0001.nii.gz)
2. Fichiers legacy (ex: P_T1W_0.1T_0001_T1W_7T_mmfm_unet.nii.gz)

## Entrées attendues

Le script accepte deux types de racines d'entrée:

1. Racine contenant task3:
- <pred_root>/task3/T1W/...

2. Racine directement sur task3:
- <pred_root>/T1W/...

Dans tous les cas, l'organisation interne attendue est:
- <task3_root>/<MOD>/<SRC>_to_<TGT>/*.nii.gz

## IDs validation Task 3

Le script valide les IDs par champ source selon la spec challenge:

- 0.1T -> 0001, 0002, 0003
- 1.5T -> 0004, 0005, 0008
- 3T -> 0010, 0011, 0012
- 5T -> 0013, 0014, 0015
- 7T -> 0016, 0017, 0018

## Utilisation rapide

Depuis la racine du projet:

```bash
source /home/rousseau/miniforge3/etc/profile.d/conda.sh
conda activate mrixfields2026
cd /home/rousseau/Exp/mrixfields_2026

PYTHONPATH=src python src/submission/build_task3_submission.py \
  --pred-root outputs/submission_candidates/mmfm_unet_v2_val \
  --output-task3-dir outputs/submission_ready/task3 \
  --accept-preclipped \
  --strict \
  --zip-output ~/task3.zip
```

## Exemple complet recommandé

### 1) Générer les prédictions Task 3 (split validation)

```bash
PYTHONPATH=src python src/cfm/infer_mmfm_unified.py \
  --config configs/mmfm3d_unet_v2_medvae_multimodal.yaml \
  --checkpoint outputs/cfm3d/runs/mmfm3d_unet_v2_medvae_multimodal/weights/checkpoint_115000.pth \
  --output_dir outputs/submission_candidates/mmfm_unet_v2_val \
  --split Validating_prospective \
  --modalities T1W T2W T2FLAIR \
  --env local \
  --skip_existing
```

### 2) Construire l'arborescence de soumission + zip

```bash
PYTHONPATH=src python src/submission/build_task3_submission.py \
  --pred-root outputs/submission_candidates/mmfm_unet_v2_val \
  --output-task3-dir outputs/submission_ready/task3 \
  --clean \
  --accept-preclipped \
  --strict \
  --zip-output ~/task3.zip
```

### 3) Vérifier le contenu du zip

```bash
unzip -l ~/task3.zip | head -n 30
```

## Arguments CLI

- --pred-root PATH
  - racine des prédictions (obligatoire)
- --output-task3-dir PATH
  - dossier task3 de sortie (défaut: outputs/submission_ready/task3)
- --modalities T1W T2W T2FLAIR
  - modalités à traiter (défaut: les 3)
- --z-start INT
  - début clip axial inclusif (défaut: 150)
- --z-end INT
  - fin clip axial exclusif (défaut: 180)
- --accept-preclipped
  - accepte les volumes déjà clippés à 30 slices
- --strict
  - échoue si un couple modalité/pair/ID est incomplet
- --clean
  - supprime le dossier de sortie avant reconstruction
- --dry-run
  - affiche les opérations sans écrire de fichiers
- --zip-output PATH
  - crée directement un zip task3

## Comportement et validations

Le script:

1. Parcourt 3 modalités x 20 paires directed
2. Retient uniquement les 3 IDs attendus par champ source
3. Préfère les fichiers au format officiel si doublons
4. Convertit chaque volume en float32 et clippe les intensités en [0, 1]
5. Applique le clip axial [z_start:z_end] (par défaut [150:180])
6. Sauvegarde au nom officiel dans pred/

En mode --strict, toute anomalie rend le code retour non nul.

## Sortie attendue

Après exécution réussie:

- arborescence:
  - outputs/submission_ready/task3/T1W/.../pred/*.nii.gz
  - outputs/submission_ready/task3/T2W/.../pred/*.nii.gz
  - outputs/submission_ready/task3/T2FLAIR/.../pred/*.nii.gz
- total attendu:
  - 180 fichiers NIfTI (3 modalités x 20 paires x 3 sujets)
- zip optionnel:
  - ~/task3.zip

## Limites et remarques

- Ce script est dédié Task 3 uniquement (pas de seg).
- Il suppose que vos prédictions couvrent bien la cohorte validation publique.
- La vérification finale officielle reste recommandée avec les outils Submission/evaluation-2026 du repo challenge.
