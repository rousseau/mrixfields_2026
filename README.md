# MRIxFields 2026

Challenge MICCAI 2026 — Cross-Field MRI Translation and Harmonization  
https://mrixfields.chihucloud.com/2026

## Où regarder en premier

**[`CHANGELOG.md`](CHANGELOG.md) — le journal des expériences.** Chaque étape,
chaque test mesuré, du plus récent au plus ancien, avec le chiffre obtenu et le
chemin du manifeste détaillé. Il porte aussi l'état de référence courant, les
planchers connus, les leçons récurrentes et la dette non traitée.

**Convention** : toute expérience qui produit un chiffre y entre. Une expérience
non écrite dans le journal est réputée ne pas avoir eu lieu.

## Structure

```
mrixfields_2026/
├── CHANGELOG.md   # journal des expériences — À TENIR À JOUR
├── src/           # code Python (modèles, pipelines, utilitaires)
├── configs/       # fichiers de configuration des expériences (YAML)
├── results/       # résultats légers : métriques CSV, manifestes, courbes PNG
├── outputs/       # résultats lourds : images, prédictions (gitignore)
└── paper/         # sources LaTeX de l'article
```

## Données

- Challenge : `~/Data/MRIxFields_20260414/`
- Code officiel du challenge : `~/Code/MRIxFields2026/` (référence, non modifié)

## Dépendances

```bash
conda activate mrixfields2026   # environnement dédié
```

## Tâches

| Task | Description |
|------|-------------|
| Task 1 | Synthèse ultra-haut champ (→ 7T) depuis 0.1T/1.5T/3T/5T |
| Task 2 | Enhancement bas champ (0.1T → 1.5T/3T/5T/7T) |
| Task 3 | Traduction any-to-any (modèle unifié) |

## Méthodes principales

| Méthode | Description | Doc |
|---------|-------------|-----|
| StarGAN 2D | Baseline any-to-any du challenge | `AGENTS.md` |
| **MMFM** (vectorisé / UNet / INR) | Multi-Marginal Flow Matching, 3 architectures comparables à code identique | `docs/MMFM_ARCHITECTURE.md` |

## Évaluation

Script d'évaluation unifié (nRMSE, SSIM, LPIPS, Dice, VolumeConsistency) :

```bash
python src/evaluation/evaluate.py --help
```

Documentation complète : `docs/EVALUATION_SCRIPT.md`

### Pipeline d'évaluation

1. **Entraînement** (VAE → CFM)
2. **Inférence** (générer les prédictions)
3. **Segmentation** (SynthSeg pour Dice/Volume — Task 1/2 seulement)
4. **Évaluation** (métriques officielles)

Voir `AGENTS.md` pour le workflow complet.
