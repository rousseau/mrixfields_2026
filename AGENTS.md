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

Les codes sont conçus pour fonctionner sur les deux environnements sans modification :

| Environnement | Matériel | Config | Lancement |
|---------------|----------|--------|-----------|
| **Local — DGX Spark GB10** | GPU GB10, multi-GPU local | `configs/env/local.yaml` | Scripts `src/*.sh` ou `torchrun` |
| **Jean Zay — HPC IDRIS** | 4×H100 SXM5 80GB | `configs/env/jeanzay.yaml` | Scripts `src/slurm/*.slurm` |

La bascule entre environnements se fait via `--env local` ou `--env jeanzay` :

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

Le projet est structuré en **3 étapes séquentielles**. Chaque étape produit des résultats évalués avec le **même script d'évaluation unifié**, permettant une comparaison directe entre méthodes.

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

Trois architectures sont comparées :

| VAE | Référence | Particularité |
|-----|-----------|---------------|
| **MedVAE** | https://github.com/StanfordMIMI/MedVAE | Pré-entraîné sur 1M images médicales — testé en version *frozen* (poids originaux) et *fine-tuné* sur MRIxFields |
| **AEKL (MONAI)** | https://github.com/Project-MONAI/GenerativeModels | AutoencoderKL 3D — 4 canaux latents, 200 epochs entraînés |
| **NeuroQuant (adapté)** | https://arxiv.org/html/2604.05171v1 | Adapté pour utiliser donn ées **paired et unpaired** — FiLM conditioning multi-champ, adversary d'invariance de modalité |

**Critères de sélection du VAE final** :
- Qualité de reconstruction (MAE, MSE, SSIM sur volumes 3D complets)
- Disentanglement anatomie / modalité / champ dans l'espace latent
- Compatibilité CFM : shape des latents, nombre de tokens, mémoire requise

| Fichier | Rôle |
|---------|------|
| `src/vae3d/train_vae_3d.py` | Entraînement AEKL 3D |
| `src/vae3d/train_vqvae.py` | Entraînement NeuroQuant adapté (VQ-VAE 3D) |
| `src/vae3d/benchmark_vae.py` | **Script de benchmark unifié** — compare les 3 VAE |
| `src/vae3d/qc_vae_3d.py` | QC visuel des reconstructions |
| `src/utils/patched_vae.py` | Wrapper patch-based pour volumes full-res |
| `src/slurm/train_vae_jeanzay.slurm` | Job SLURM unifié (aekl / vqvae / medvae) |
| `src/slurm/train_vqvae_jeanzay.slurm` | Job SLURM VQ-VAE 3D (4×H100) |

```bash
# AEKL
python src/vae3d/train_vae_3d.py --config configs/vae3d_T1W.yaml --env local

# NeuroQuant (VQ-VAE)
python src/vae3d/train_vqvae.py --config configs/vqvae3d_T1W.yaml --env local

# Benchmark des 3 architectures
python src/vae3d/benchmark_vae.py --modality T1W --field 0.1T

# Jean Zay
sbatch src/slurm/train_vae_jeanzay.slurm aekl vae3d_T1W
sbatch src/slurm/train_vqvae_jeanzay.slurm vqvae3d_T1W
sbatch src/slurm/benchmark_vae_jeanzay.slurm T1W 0.1T
```

**État** : AEKL ✅ (200 epochs), VQ-VAE ✅ (smoke tests), MedVAE ⏳ (fine-tuning à lancer).

---

### Étape 3 — Comparaison CFM dans l’espace latent *(translation)*

**Objectif** : Comparer différentes approches de Conditional Flow Matching (CFM) dans l’espace latent du VAE sélectionné à l’étape 2.

| Approche | Config | VAE latent |
|----------|--------|-----------|
| **OT-CFM 3D + AEKL** | `cfm3d_T1W_aekl.yaml` | 4 canaux (MONAI AutoencoderKL) |
| **OT-CFM 3D + MedVAE** | `cfm3d_T1W_medvae.yaml` | 4 canaux (StanfordMIMI, frozen ou fine-tuné) |
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

**État** : CFM 3D ⏳ — les 3 variantes VAE sont prêtes à lancer.
---

## Évaluation unifiée

**Toutes les méthodes sont évaluées avec le même script**, garantissant la cohérence des comparaisons.

### Évaluation visuelle

Réalisée sur les **3 sujets prospectifs** acquis avec les 5 champs magnétiques (0.1T, 1.5T, 3T, 5T, 7T) :

```bash
# QC visuel — sujets 5 champs (1 figure par méthode et par sujet)
python src/evaluation/evaluate.py \
    --method stargan2d \
    --checkpoint outputs/stargan2d/runs/task3_any_to_any_T1W/weights/model_final.pth \
    --subjects prospective_5fields \
    --output results/qc/

python src/evaluation/evaluate.py \
    --method aekl_cfm3d \
    --vae-checkpoint outputs/vae3d/runs/vae3d_T1W/weights/model_best.pth \
    --cfm-checkpoint outputs/cfm3d/runs/cfm3d_T1W/weights/model_final.pth \
    --subjects prospective_5fields \
    --output results/qc/
```

Les figures sont sauvegardées dans `results/qc/<methode>_<sujet>_<modalite>.png` et montrent la traduction entre les 5 champs pour les 3 vues (axiale, coronale, sagittale).

### Évaluation quantitative

Un tableau de métriques est maintenu dans `results/evaluation_table.csv` et **enrichi progressivement** au fil des expériences. Ce fichier est versionné dans git.

| Méthode | Modalité | nRMSE ↓ | SSIM ↑ | LPIPS ↓ | Dice ↑ | Notes |
|---------|----------|---------|--------|---------|--------|-------|
| StarGAN 2D (baseline) | T1W | — | — | — | — | checkpoint 150k |
| AEKL + OT-CFM 3D | T1W | — | — | — | — | à compléter |
| VQ-VAE + OT-CFM 3D | T1W | — | — | — | — | à lancer |
| MedVAE (frozen) + CFM | T1W | — | — | — | — | à lancer |
| MedVAE (fine-tuné) + CFM | T1W | — | — | — | — | à lancer |

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
│   │   ├── local.yaml                  #   DGX Spark GB10
│   │   ├── jeanzay.yaml                #   IDRIS 4×H100 SXM5
│   │   └── dgx.yaml                    #   DGX Station multi-GPU
│   ├── vae3d_T1W.yaml                  #   AEKL 3D — T1W
│   ├── vqvae3d_T1W.yaml                #   VQ-VAE 3D (NeuroQuant) — T1W
│   ├── cfm3d_T1W_aekl.yaml              #   OT-CFM 3D + AEKL — T1W
│   ├── cfm3d_T1W_medvae.yaml            #   OT-CFM 3D + MedVAE — T1W
│   ├── cfm3d_T1W_vqvae.yaml             #   OT-CFM 3D + VQ-VAE — T1W
│   ├── stargan2d_T1W.yaml              #   StarGAN 2D — T1W
│   └── benchmark.yaml                  #   Benchmark VAE (à créer)
│
├── src/                                # ─── CODE SOURCE ───
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
│   │   ├── train_cfm2d.py              #     OT-CFM 2D
│   │   ├── train_cfm_3d.py             #     OT-CFM 3D latent
│   │   ├── launch_cfm_screen.sh        #     Lancement screen local
│   │   ├── watch_and_viz_cfm2d.sh      #     Surveillance checkpoints
│   │   └── test_benchmark_smoke.py     #     Smoke test
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
│   │   ├── visualize_cfm2d.py
│   │   └── visualize_stargan2d.py
│   │
│   ├── analysis/                       #   Analyse du dataset
│   │   └── dataset_stats.py
│   │
│   └── slurm/                          #   Jobs SLURM (Jean Zay uniquement)
│       ├── setup_jeanzay.sh
│       ├── train_vae_jeanzay.slurm
│       ├── train_vqvae_jeanzay.slurm
│       ├── cfm_jeanzay.slurm
│       ├── cfm_3d_jeanzay.slurm
│       ├── stargan_jeanzay.slurm
│       ├── benchmark_vae_jeanzay.slurm
│       ├── submit_train_vae_jeanzay.sh
│       └── launch_cfm3d_dgx.sh         #     Lancement multi-GPU local (DGX)
│
├── outputs/                            # ─── SORTIES LOURDES (gitignore) ───
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

### Logs

`<méthode>_<job_id>.{out,err}` → dans `logs/`

---

## État d'avancement

| Étape | Méthode | État | Checkpoint |
|-------|---------|------|------------|
| 1 | StarGAN 2D (T1W) | ✅ Terminé | `task3_any_to_any_T1W/` |
| 2 | AEKL 3D (T1W) | ✅ Terminé | `vae3d_T1W/weights/model_best.pth` |
| 2 | VQ-VAE NeuroQuant (T1W) | ✅ Smoke tests | `vqvae3d/runs/smoke_*` |
| 2 | MedVAE frozen | ⏳ À évaluer | poids HuggingFace |
| 2 | MedVAE fine-tuné | ⏳ À lancer | — |
| 2 | Benchmark VAE | ✅ Partiel | `results/benchmark_comparison/` (3T max) |
| 3 | OT-CFM 3D + AEKL (T1W) | ⏳ À lancer | `cfm3d_T1W_aekl/weights/` |
| 3 | OT-CFM 3D + MedVAE (T1W) | ⏳ À lancer | `cfm3d_T1W_medvae/weights/` |
| 3 | OT-CFM 3D + VQ-VAE (T1W) | ⏳ À lancer | `cfm3d_T1W_vqvae/weights/` |
| — | Script évaluation unifié | ⏳ À créer | `src/evaluation/evaluate.py` |
| — | Tableau métriques | ⏳ À initialiser | `results/evaluation_table.csv` |
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
