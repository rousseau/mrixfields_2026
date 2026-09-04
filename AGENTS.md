# AGENTS.md — Projet MRIxFields 2026

## Objectif

Ce projet a pour but de **développer et comparer des méthodes de translation cross-field d'images IRM** dans le cadre du **challenge MRIxFields 2026** (MICCAI 2026).

- Site du challenge : https://mrixfields.chihucloud.com/2026/
- Repo GitHub : https://github.com/rousseau/mrixfields_2026
- Code officiel du challenge (baseline) : `~/Code/MRIxFields2026/` *(ne pas modifier)*
- Données : `~/Data/MRIxFields_20260414/`

Le challenge évalue trois tâches complémentaires sur 5 champs magnétiques (0.1T, 1.5T, 3T, 5T, 7T) et 3 modalités (T1W, T2W, T2FLAIR) :

| Task | Description |
|------|-------------|
| **Task 1** | Synthèse ultra-haut champ (→ 7T) depuis 0.1T / 1.5T / 3T / 5T |
| **Task 2** | Enhancement bas champ (0.1T → 1.5T / 3T / 5T / 7T) |
| **Task 3** | Traduction any-to-any (modèle unifié conditionné par le champ cible) |

---

## Environnement

### Installation

Un environnement conda **dédié à ce projet** est requis pour garantir la reproductibilité :

```bash
# Création de l'environnement
conda create -n mrixfields2026 python=3.11 -y
conda activate mrixfields2026

# Dépendances principales
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install monai[all]>=1.3.0 torchcfm pot torchdiffeq
pip install nibabel scipy einops matplotlib pandas

# Package officiel du challenge (baseline StarGAN)
pip install --no-deps ~/Code/MRIxFields2026/Baseline

# Vérification
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

> **Note** : L'environnement s'appelle `mrixfields2026` (et non `mf`).

### Matériel supporté

Les codes sont conçus pour fonctionner sur deux environnements :

| Environnement | Matériel | Config | Lancement |
|---|---|---|---|
| **Local — DGX GB10 Lenovo ThinkStation** | GPU GB10 (128GB VRAM), multi-GPU local | `configs/env/local.yaml` | `src/slurm/launch_cfm3d_dgx.sh` ou `torchrun` |
| **Remote — HPC** | 4×H100 SXM5 80GB | `configs/env/remote.yaml` | Scripts `src/slurm/*.slurm` |

La bascule entre environnements se fait via `--env local` ou `--env remote` :

```bash
# Local
python src/vae3d/train_vae_3d.py --config configs/vae3d_T1W.yaml --env local

# Jean Zay (via SLURM)
sbatch src/slurm/train_vae_jeanzay.slurm aekl vae3d_T1W
```

Les fichiers `configs/env/*.yaml` définissent les chemins spécifiques à chaque machine :
- `data_root` — données brutes NIfTI
- `output_root` — sorties lourdes (checkpoints, prédictions)
- `project_root` — racine du projet

---

## Étapes de développement

Le projet est structuré en **4 étapes séquentielles**. Chaque étape produit des résultats évalués avec le **même script d'évaluation unifié**, permettant une comparaison directe entre méthodes.

### Étape 1 — Baseline StarGAN 2D *(référence)*

**Objectif** : Établir une référence de performance solide avant d'explorer les approches latentes 3D.

- Méthode : **StarGAN v2 2D** — code baseline du challenge MICCAI
- Entraînement sur slices 2D extraites des volumes NIfTI (512×512)
- Couvre directement la **Task 3** (any-to-any)
- Sert de **baseline de comparaison** pour toutes les méthodes suivantes

| Fichier | Rôle |
|---------|------|
| `src/stargan/train_stargan2d.sh` | Wrapper d'entraînement (local et Jean Zay) |
| `src/stargan/run_inference_stargan2d.sh` | Inférence sur les sujets de test |
| `src/stargan/launch_training_screen.sh` | Lancement en session screen (local) |
| `src/stargan/watch_and_viz.sh` | Surveillance des checkpoints et QC |
| `src/slurm/stargan_jeanzay.slurm` | Job SLURM Jean Zay |

```bash
# Entraînement local
bash src/stargan/train_stargan2d.sh T1W retro_scratch

# Jean Zay
sbatch src/slurm/stargan_jeanzay.slurm T1W retro_scratch
```

**État** : ✅ Runs `task3_any_to_any_T1W` terminés.

---

### Étape 2 — Comparaison VAE 3D *(espace latent)*

**Objectif** : Identifier la meilleure architecture VAE 3D pour encoder les volumes IRM dans un espace latent adapté à la translation de champs.

Quatre architectures sont comparées :

| VAE | Référence | Particularité | Latent |
|-----|-----------|---------------|--------|
| **AEKL (MONAI)** | https://github.com/Project-MONAI/GenerativeModels | AutoencoderKL 3D — 4 canaux latents | Spatial `(C,H',W',D')` |
| **Pythae VAE 3D** | https://github.com/clementchadebec/benchmark_VAE | VAE propre 3D (remplace VQ-VAE NeuroQuant) | Spatial `(C,H',W',D')` |
| **Pythae VQ-VAE 3D** | https://github.com/clementchadebec/benchmark_VAE | VQ-VAE 3D avec quantizer 5D custom | Spatial `(C,H',W',D')` |
| **RHVAE 3D** | https://github.com/clementchadebec/benchmark_VAE | Riemannian Hamiltonian VAE — vecteur plat | Vectoriel `(D_lat)` |
| **MedVAE** | https://github.com/StanfordMIMI/MedVAE | Pré-entraîné sur 1M images médicales — frozen / fine-tuné | Spatial `(C,H',W',D')` |
| **MAISI** | https://github.com/NVIDIA-Medtech/NV-Generate-CTMR / MONAI bundle | Pré-entraîné sur 55k CT+MRI volumes | Spatial `(C,H',W',D')` |

> **Règle challenge Task 3** : un seul modèle unifié est autorisé. Par conséquent, tous les VAE sont entraînés sur **T1W + T2W + T2FLAIR** (pas de modèles mono-modaux). Voir `docs/VAE_IMPLEMENTATION_PLAN.md`.

> **Patch standard** : `(128, 128, 128)` pour tous les VAE (inférence patch-based via `PatchedVAE`).

**Critères de sélection du VAE final** :
- Qualité de reconstruction (MAE, MSE, SSIM sur volumes 3D complets)
- Disentanglement anatomie / modalité / champ dans l'espace latent
- Compatibilité CFM : shape des latents, nombre de tokens, mémoire requise

| Fichier | Rôle |
|---------|------|
| `src/models/vae_base.py` | Classe abstraite `MRIxFieldsVAE` — API commune |
| `src/models/vae_wrappers.py` | Wrappers AEKL/MedVAE/VQ-VAE héritant de `MRIxFieldsVAE` |
| `src/models/vae_loader.py` | `load_vae(cfg, device)` — point d'entrée unifié |
| `src/common/dataset_vae.py` | Dataset **multimodal** (T1W+T2W+T2FLAIR × tous champs) |
| `src/vae3d/train_vae_3d.py` | Entraînement AEKL 3D (refactorisé pour dataset multimodal) |
| `src/vae3d/train_vqvae.py` | Entraînement NeuroQuant adapté (VQ-VAE 3D) — **deprecated** |
| `src/vae3d/train_medvae_disentangle_v1.py` | Entraînement MedVAE Disentanglement v1 — legacy |
| `src/vae3d/benchmark_vae.py` | **Script de benchmark unifié** — compare tous VAE |
| `src/vae3d/benchmark_disentanglement_v1.py` | Benchmark disentanglement v1 (legacy) |
| `src/vae3d/qc_vae_3d.py` | QC visuel des reconstructions |
| `src/vae3d/visualize_latent_umap.py` | UMAP 2D coloré (modalité + champ) — à créer |
| `src/utils/patched_vae.py` | Wrapper patch-based (inférence full-res) |
| `src/slurm/train_vae_jeanzay.slurm` | Job SLURM unifié (aekl / vqvae / medvae) |

```bash
# AEKL multimodal (128³)
python src/vae3d/train_vae_3d.py --config configs/vae3d_multimodal.yaml --env local

# NeuroQuant (VQ-VAE) — legacy, remplacé par Pythae VQ-VAE dans Phase B
python src/vae3d/train_vqvae.py --config configs/vqvae3d_T1W.yaml --env local

# MedVAE Disentanglement v1 (legacy)
python src/vae3d/train_medvae_disentangle_v1.py \
    --data-root /home/rousseau/Data/MRIxFields_20260414 \
    --output-dir outputs/medvae_disentangle_v1/runs/dev_run \
    --splits retro_train \
    --modalities T1W T2W T2FLAIR \
    --fields 0.1T 1.5T 3T 5T 7T

# Benchmark automatique des architectures (étape 2)
python src/vae3d/benchmark_vae.py --modality T1W --field 0.1T

# Jean Zay — AEKL multimodal
sbatch src/slurm/train_vae_jeanzay.slurm aekl vae3d_multimodal
```

**État** : AEKL ✅ (T1W only, à ré-entraîner multimodal), Pythae VAE/VQ-VAE/RHVAE ⏳ (Phase A–C), MedVAE ⏳ (fine-tuning à lancer).

---

### Étape 3 — OT-CFM dans l’espace latent *(translation, meilleur VAE)*

**Objectif** : Comparer différentes approches de Conditional Flow Matching (OT-CFM / CFM) dans l’espace latent du meilleur VAE sélectionné à l’étape 2. **À ce stade, le meilleur VAE est MedVAE**.

| Approche | Config | VAE latent |
|----------|--------|-----------|
| **OT-CFM 3D + AEKL** | `cfm3d_T1W_aekl.yaml` | 4 canaux (MONAI AutoencoderKL) |
| **OT-CFM 3D + MedVAE** | `cfm3d_T1W_medvae.yaml` | VAE retenu à ce stade (frozen ou fine-tuné) |
| **OT-CFM 3D + VQ-VAE** | `cfm3d_T1W_vqvae.yaml` | 64 canaux anatomiques z_q (NeuroQuant) |
| **Variants** | options YAML | `exact` vs `sinkhorn` OT, Euler vs DoPri5 |

Le script `train_cfm_3d.py` sélectionne le VAE via `vae.vae_type` dans le YAML.
Les `latent_channels` sont déduits automatiquement du wrapper — sans configuration manuelle.

**Fichiers clés** :
- `src/cfm/train_cfm_3d.py` — OT-CFM 3D multi-VAE (script unique)
- `configs/cfm3d_T1W_aekl.yaml`, `cfm3d_T1W_medvae.yaml`, `cfm3d_T1W_vqvae.yaml`
- `src/slurm/cfm_3d_jeanzay.slurm` — Job SLURM CFM 3D (4×H100)
- `src/slurm/launch_cfm3d_dgx.sh` — Lancement multi-GPU local (DGX)

```bash
# OT-CFM 3D — choisir le VAE via la config
python src/cfm/train_cfm_3d.py --config configs/cfm3d_T1W_aekl.yaml   --env local
python src/cfm/train_cfm_3d.py --config configs/cfm3d_T1W_medvae.yaml --env local
python src/cfm/train_cfm_3d.py --config configs/cfm3d_T1W_vqvae.yaml  --env local

# Jean Zay
sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_aekl.yaml
sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_medvae.yaml
sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_vqvae.yaml
```

**État** : CFM 3D ⏳ — la baseline cible est MedVAE, les autres variantes restent comparatives.

---

### Étape 4 — MMFM (vectorisé / UNet / INR)

**Objectif** : Multi-Marginal Flow Matching, comparant plusieurs architectures de modèle de flow à
code strictement identique (seule l'architecture diffère). Vectorisé/UNet opèrent sur un latent
MedVAE gelé ; l'INR opère directement sur le volume brut via un backbone SIREN+hypernetwork
méta-appris séparément (pas de MedVAE pour cette variante).

**Principe** : voir `docs/MMFM_ARCHITECTURE.md` pour le design complet (sampler multi-marginal,
couplage OT-CFM, `ArchAdapter`) et `docs/MMFM_INR_STATE_OF_THE_ART.md` pour le design/état de l'art
spécifique à la variante INR.

| Fichier | Rôle |
|---------|------|
| `src/cfm/mmfm_core.py` | Boucle d'entraînement/inférence partagée (sampler, OT-CFM, pertes, EMA, checkpointing) |
| `src/cfm/arch_vector.py` | `ArchAdapter` — variante vectorisée (MLP résiduel) |
| `src/cfm/arch_unet.py` | `ArchAdapter` — variante UNet 3D spatial (MONAI) |
| `src/cfm/arch_inr.py` | `ArchAdapter` — variante INR (réutilise `VectorMMFM` comme modèle de flow) |
| `src/cfm/mmfm_vectorized.py` | Briques vectorielles: flatten/unflatten, embeddings temps/classe, MLP résiduel |
| `src/cfm/inr_backbone.py` | Backbone INR : SIREN + hypernetwork + meta-learning (NOIR Algorithme 1/2) |
| `src/cfm/train_inr_backbone.py` | Entraînement séparé du backbone INR (analogue au fine-tuning MedVAE) |
| `src/cfm/precompute_inr_latents.py` | Cache des latents INR fittés (Algorithme 2, coûteux à refaire à chaque step) |
| `src/cfm/train_mmfm_unified.py` | CLI unique train+infer : `--method mmfm3d_vectorized\|mmfm3d_unet\|mmfm3d_inr` |
| `configs/mmfm/vectorized.yaml`, `configs/mmfm/unet.yaml`, `configs/mmfm/inr.yaml` | Configs d'entraînement canoniques (flow) |
| `configs/mmfm/inr_backbone.yaml` | Config d'entraînement du backbone INR |
| `docs/MMFM_ARCHITECTURE.md` | Documentation de l'architecture MMFM unifiée |
| `src/slurm/cfm_3d_jeanzay.slurm` | Job SLURM Jean Zay (phases `mmfm`/`mmfm_unet`) |
| `src/slurm/launch_cfm3d_dgx.sh` | Lancement multi-GPU local (phases `mmfm`/`mmfm_unet`) |

```bash
# Local
PYTHONPATH=src python src/cfm/train_mmfm_unified.py --method mmfm3d_vectorized \
    --config configs/mmfm/vectorized.yaml --env local

# Jean Zay
sbatch src/slurm/cfm_3d_jeanzay.slurm mmfm T1W configs/mmfm/vectorized.yaml
```

**État** : ✅ Les trois variantes sont codées, entraînées et évaluées à code strictement identique
(voir `results/mmfm/comparison_20260807_1mm/` et `results/mmfm/comparison_20260814_all_contrasts/`).
État de référence courant (Task 3, 20 paires × 3 sujets, évaluateur officiel, 1 mm, prédictions
écrites à 0.5 mm natif) :

| Architecture | nRMSE ↓ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| **INR** (`outputs/mmfm/inr_std`) | **0.3749** | 0.8631 | 0.1557 |
| **Vectorisé** (`outputs/mmfm/vectorized`) | 0.3794 | **0.8975** | **0.0941** |
| **UNet** (`outputs/mmfm/unet`) | 0.4033 | 0.8949 | 0.0953 |

L'INR est **première en nRMSE** (0.3749) mais **derrière en SSIM et LPIPS** ; les deux scores de
nRMSE sont **statistiquement indiscernables** (29 victoires/60, test des signes p = 0.65, Wilcoxon
p = 0.38 — ne pas écrire « la meilleure » sans la métrique). Le meilleur connu en **géométrie de
flow** est R-best (`configs/mmfm/vectorized_rbest.yaml`, nRMSE **0.3737**), non basé en canonique
parce que son gain est sous la résolution statistique (n = 3). L'historique complet, les planchers
connus et les délibérations sont dans **`CHANGELOG.md`** (source de vérité).
---

### Évaluation unifiée

**Toutes les méthodes sont évaluées avec le même script**, garantissant la cohérence des comparaisons.

#### Usage

```bash
# Évaluation quantitative (nRMSE, SSIM, LPIPS)
python src/evaluation/evaluate.py --method aekl_cfm3d \
    --vae-checkpoint outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth \
    --cfm-checkpoint outputs/cfm3d/runs/cfm3d_T1W/weights/model_final.pth \
    --subjects prospective_5fields \
    --output-csv results/evaluation_table.csv

# Évaluation complète avec Dice/Volume (nécessite SynthSeg)
python src/evaluation/evaluate.py --method stargan2d \
    --checkpoint outputs/stargan2d/runs/task3_any_to_any_T1W/weights/model_final.pth \
    --pred-dir outputs/predictions/ \
    --target-dir ~/Data/MRIxFields_20260414/Training_prospective/ \
    --pred-seg-dir outputs/predictions_seg/ \
    --target-seg-dir ~/Data/MRIxFields_20260414/target_seg/ \
    --metrics nrmse,ssim,lpips,dice,volume

# Aide
python src/evaluation/evaluate.py --help
```

Les figures sont sauvegardées dans `results/{stargan,cfm,mmfm}/visuals/<methode>_<sujet>_<modalite>.png` et montrent la traduction entre les 5 champs pour les 3 vues (axiale, coronale, sagittale).

### Évaluation quantitative

Un tableau de métriques est maintenu dans `results/evaluation_table.csv` et **enrichi progressivement** au fil des expériences. Ce fichier est versionné dans git.

| Méthode | nRMSE ↓ (moy.) | SSIM ↑ | LPIPS ↓ | Résultat |
|---------|---------|--------|---------|-------|
| **INR** (Task 3, 3 contrastes, 1 mm) | **0.3749** | 0.8631 | 0.1557 | ✅ référence |
| **Vectorisé** (Task 3, 3 contrastes, 1 mm) | 0.3794 | **0.8975** | **0.0941** | ✅ référence |
| **UNet** (Task 3, 3 contrastes, 1 mm) | 0.4033 | 0.8949 | 0.0953 | ✅ référence |
| *R-best (géométrie de flow, vectorisé)* | *0.3737* | — | — | non canonique (gain < bruit) |

> **Note** : les métriques Dice/Volume (Task 1/2) et les variantes « star-gan/aek-l/vqvae »
> sont hors périmètre de l'état de référence courant — voir `CHANGELOG.md` et
> `results/mmfm/comparison_20260814_all_contrasts/` pour l'historique complet des 11 entrées de
> ce tableau (lequel est la source de vérité).

**Métriques** (identiques à celles du challenge) :
- `nRMSE` — normalized Root Mean Square Error
- `SSIM` — Structural Similarity Index
- `LPIPS` — Learned Perceptual Image Patch Similarity
- `Dice` — sur segmentations automatiques (FSL/FreeSurfer)
- `VolumeConsistency` — consistance volumique normalisée

```bash
# Calcul des métriques quantitatives (sujets paired prospectifs)
python src/evaluation/evaluate.py --mode quantitative \
    --method aekl_cfm3d \
    --output results/evaluation_table.csv
```

---

## Structure du projet

```
mrixfields_2026/
├── README.md
├── AGENTS.md                           # Ce fichier
├── .gitignore
│
├── configs/                            # ─── CONFIGURATIONS ───
│   ├── env/
│   │   ├── local.yaml                  #   DGX GB10 Lenovo ThinkStation
│   │   └── jeanzay.yaml                #   IDRIS 4×H100 SXM5
│   ├── vae3d_multimodal.yaml           #   AEKL 3D — multimodal
│   ├── vae3d_T1W.yaml                  #   AEKL 3D — T1W (legacy)
│   ├── vqvae3d_T1W.yaml                #   VQ-VAE 3D (NeuroQuant) — T1W (legacy)
│   ├── cfm3d_T1W_aekl.yaml             #   OT-CFM 3D + AEKL — T1W
│   ├── cfm3d_T1W_medvae.yaml           #   OT-CFM 3D + MedVAE — T1W
│   ├── cfm3d_T1W_vqvae.yaml            #   OT-CFM 3D + VQ-VAE — T1W
│   ├── stargan2d_T1W.yaml              #   StarGAN 2D — T1W
│   └── benchmark.yaml                  #   Benchmark VAE (à créer)
│
├── src/                                # ─── CODE SOURCE ───
│   ├── models/
│   │   ├── vae_base.py                 #   Classe abstraite MRIxFieldsVAE — API commune
│   │   ├── vae_wrappers.py             #   Wrappers AEKL/MedVAE/VQ-VAE héritant de MRIxFieldsVAE
│   │   └── vae_loader.py               #   load_vae(cfg, device) — point d'entrée unifié
│   │
│   ├── utils/
│   │   └── patched_vae.py              #   Wrapper patch-based (inference full-res)
│   │
│   ├── vae2d/                          #   VAE 2D (comparison architectures 2D)
│   │   ├── train_vae.py
│   │   └── test_train_vae.py
│   │
│   ├── vae3d/                          #   Étape 2 : VAE 3D
│   │   ├── train_vae_3d.py             #     AEKL 3D
│   │   ├── train_vqvae.py              #     NeuroQuant adapté
│   │   ├── benchmark_vae.py            #     Benchmark unifié (AEKL / VQ-VAE / MedVAE)
│   │   ├── qc_vae_3d.py                #     QC reconstructions
│   │   └── test_patched_vae.py         #     Smoke test wrapper
│   │
│   ├── cfm/                            #   Étape 3 : CFM
│   │   ├── train_cfm_3d.py             #     OT-CFM 3D latent
│   │   ├── mmfm_core.py                #     MMFM : boucle partagée (sampler, OT-CFM, EMA...)
│   │   ├── arch_vector.py              #     MMFM : ArchAdapter vectorisé
│   │   ├── arch_unet.py                #     MMFM : ArchAdapter UNet 3D (multi-marginal)
│   │   ├── train_mmfm_unified.py       #     MMFM : CLI unique train+infer
│   │   ├── precompute_unet_latents.py  #     Cache de latents pour MMFM-UNet
│   │   ├── precompute_mmfm_latents.py  #     Cache de latents pour MMFM vectorisé
│   │   ├── infer_mmfm_unified.py       #     Inférence any-to-any MMFM
│   │   └── mmfm_vectorized.py          #     Briques vectorielles
│   │
│   ├── stargan/                        #   Étape 1 : Baseline StarGAN 2D
│   │   ├── train_stargan2d.sh          #     Entraînement
│   │   ├── run_inference_stargan2d.sh  #     Inférence
│   │   ├── launch_training_screen.sh   #     Lancement screen local
│   │   └── watch_and_viz.sh            #     Surveillance checkpoints
│   │
│   ├── evaluation/                     #   Script d'évaluation UNIFIÉ
│   │   └── evaluate.py                 #     Visual + quantitatif (toutes méthodes)
│   │
│   ├── visualization/                  #   Figures et QC
│   │   ├── visualize.py
│   │   ├── visualize_stargan2d.py
│   │   └── viz_normalization_effect.py
│   │
│   ├── common/                         #   Infrastructure commune
│   │   ├── io.py                       #     NIfTI I/O, preprocessing
│   │   ├── metrics.py                  #     Métriques (nRMSE, SSIM, LPIPS)
│   │   └── dataset_vae.py              #     Dataset multimodal VAE
│   │
│   ├── analysis/                       #   Analyse du dataset
│   │   └── dataset_stats.py
│   │
│   └── slurm/                          #   Jobs SLURM (Jean Zay uniquement)
│       ├── setup_jeanzay.sh
│       ├── sync_to_jeanzay.sh         #     Sync code + poids + cache vers Jean Zay
│       ├── train_vae_jeanzay.slurm
│       ├── train_vqvae_jeanzay.slurm
│       ├── cfm_3d_jeanzay.slurm
│       ├── train_mmfm_multimarginal_jeanzay.slurm  #   Run 2 multi-marginal (DDP 4×H100)
│       ├── infer_mmfm_jeanzay.slurm   #     Inférence multi-GPU Task 3
│       ├── stargan_jeanzay.slurm
│       ├── benchmark_vae_jeanzay.slurm
│       ├── submit_train_vae_jeanzay.sh
│       ├── launch_cfm3d_dgx.sh         #     Lancement multi-GPU local (DGX)
│       └── run_mmfm_multimarginal_night.sh  #     Pré-encode + entraîne MMFM-UNet
│
├── outputs/                            # ── SORTIES LOURDES (gitignore) ──
│   ├── vae3d/runs/vae3d_T1W/weights/   #   AEKL — 20 ckpts + model_best + model_final ✅
│   ├── vqvae3d/runs/                   #   VQ-VAE — smoke + stability tests ✅
│   ├── cfm3d/runs/cfm3d_T1W/weights/   #   CFM 3D — ⚠️ vide (à entraîner)
│   ├── stargan2d/runs/                 #   StarGAN — task3_any_to_any_T1W ✅
│   ├── medvae/                         #   MedVAE QC images
│   ├── aekl/                           #   AEKL QC images
│   └── benchmark_test/                 #   CSV smoke tests
│
├── results/                            # ─── RÉSULTATS LÉGERS (versionnés) ───
│   ├── qc/                             #   Figures QC visuelles (PNG)
│   ├── cfm/                            #   Figures CFM (PNG)
│   ├── benchmark_comparison/           #   Benchmark VAE — 9 CSV (3 mod × 3 champs)
│   ├── stats/                          #   Statistiques dataset — 5 CSV
│   └── evaluation_table.csv            #   Tableau de métriques cumulatif ← À MAINTENIR
│
├── docs/                               # ─── DOCUMENTATION ───
│   ├── BENCHMARK_IMPLEMENTATION.md
│   ├── BENCHMARK_PLAN.md
│   ├── EVALUATION_SCRIPT.md
│   ├── VAE_IMPLEMENTATION_PLAN.md      #   Plan VAE détaillé (Phases A–F)
│   ├── MMFM_ARCHITECTURE.md
│   └── ARCHITECTURE.md                 #   (à créer)
│
├── paper/                              # ─── PAPIER (vide) ───
│   └── .gitkeep
│
└── logs/                               # ─── LOGS (gitignore) ───
    └── [méthode]_[job_id].{out,err}
```

---

## Conventions

### Nommage des runs

`<méthode>_<modalité>[_<variante>]`

| Exemple | Signification |
|---------|---------------|
| `vae3d_T1W` | AEKL 3D, T1W |
| `cfm2d_T1W_H100_B16` | OT-CFM 2D, T1W, H100, batch 16 |
| `task3_any_to_any_T1W` | StarGAN Task 3, T1W |

### Checkpoints

| Méthode | Pattern | Exemple |
|---------|---------|---------|
| VAE (epochs) | `epoch_XXXX.pth` | `epoch_0200.pth` |
| CFM (iters) | `checkpoint_XXXX.pth` | `checkpoint_150000.pth` |
| Spéciaux | `model_best.pth`, `model_final.pth` | AEKL VAE |

### Configs YAML

`<méthode>_<modalité>[_<variante>].yaml`

| Fichier | Usage |
|---------|-------|
| `vae3d_T1W.yaml` | AEKL 3D, T1W |
| `cfm3d_T1W.yaml` | OT-CFM 3D latent, T1W |
| `cfm2d_T1W_H100_B16.yaml` | OT-CFM 2D, T1W, H100, batch 16 |
| `mmfm3d_multimarginal_medvae.yaml` | MMFM-UNet multi-marginal, config de base |
| `mmfm3d_multimarginal_medvae_run1.yaml` | MMFM-UNet multi-marginal, run 1 (12h) |

### Logs

`<méthode>_<job_id>.{out,err}` → dans `logs/`

---

## État d'avancement

> Les numéros d'étape historisent l'ordre de développement. L'état de référence **courant** est
> celui de l'Étape 4 (MMFM vectorisé / UNet / INR), à code strictement identique — voir le tableau
> précédent et `CHANGELOG.md`. Les étapes 3 (OT-CFM 3D) et 4 (MMFM v1) sont des jalons dépassés.

| Étape | Méthode | État | Checkpoint |
|-----|---------|------|------------|
| 1 | StarGAN 2D (T1W) | ✅ Terminé | `task3_any_to_any_T1W/` |
| 2 | MedVAE (pré-entraîné, frozen) | ✅ en usage | poids HuggingFace `medvae_4_1_3d` |
| 2 | MedVAE fine-tuné | ✅ (abandonné, voir `vectorized.yaml:34`) | `outputs/medvae/runs/medvae_finetune_all/weights/model_best.pth` |
| 2 | AEKL / VQ-VAE / RHVAE / Pythae | ✅ Smoke/legacy | voir `results/benchmark_vae/` |
| 3 | OT-CFM 3D + VAE | ⚫ jalons historiques | `cfm3d_T1W_*/weights/` |
| 4 | **MMFM vectorisé (MedVAE gelé)** | ✅ **référence 0.3794 nRMSE** | `outputs/mmfm/vectorized/weights/model_final.pth` |
| 4 | **MMFM UNet (MedVAE gelé)** | ✅ **référence 0.4033 nRMSE** | `outputs/mmfm/unet/weights/model_final.pth` |
| 4 | **MMFM INR (backbone SIREN)** | ✅ **référence 0.3749 nRMSE** | `outputs/mmfm/inr_std/weights/model_final.pth` |
| 4 | R-best (vectorisé, temps/FiLM corrigés) | ✅ meilleur géométrie 0.3737 | `outputs/mmfm/vec_rbest/weights/model_final.pth` |
| — | Script évaluation unifié | ✅ Terminé | `src/evaluation/evaluate.py` (5 méthodes) |
| — | Journal des expériences (source de vérité) | ✅ À MAINTENIR | `CHANGELOG.md` |
| — | Paper | ⬜ Vide | `paper/` |

---

## Données

| Split | Description | Volumes |
|-------|-------------|---------|
| `Training_retrospective` | Données non-appariées multi-centre | 1900+ |
| `Training_prospective` | Sujets voyageurs — **5 champs × 3 modalités** | 45 (3 sujets train) |
| `Validation` | 17 sujets voyageurs (éval intermédiaire) | 255 |
| `Test` | 20 sujets voyageurs (éval finale challenge) | 300 |

Les **3 sujets prospectifs d'entraînement** (acquis avec les 5 champs magnétiques) servent de base pour l'évaluation visuelle qualitative.

---

## Notes importantes

- **`outputs/`** et **`logs/`** sont exclus de Git (`.gitignore`) — poids et prédictions non versionnés
- **`results/`** est versionné — CSV légers et figures de QC
- **`results/evaluation_table.csv`** doit être mis à jour à chaque nouvelle expérience
- L'environnement conda est **`mrixfields2026`** (distinct de `mf` utilisé précédemment)
- Ne pas modifier `~/Code/MRIxFields2026/` (code officiel challenge)
