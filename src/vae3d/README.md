# 🧠 VAE 3D — Représentations Latentes Multimodales

Ce module implémente et gère les Auto-encodeurs Variationnels (VAE) 3D utilisés pour compresser et représenter des volumes IRM multimodaux (T1W, T2W, T2FLAIR) dans le cadre du challenge MRIxFields 2026.

## 📌 État des Lieux & Hiérarchie des Modèles

L'approche adoptée suit une progression allant de la baseline standard vers des modèles intégrant des contraintes géométriques ou profitant de pré-entraînements massifs.

### 1. La Baseline : AEKL (MONAI)
Le modèle **AEKL** (AutoEncoderKL) de MONAI Generative sert de point de référence. Il s'agit d'un VAE 3D classique avec un espace latent spatial.
- **Fichier d'entraînement** : `train_vae_3d.py`
- **Config** : `configs/vae3d_multimodal.yaml`
- **Rôle** : Établir la performance de reconstruction de base.

### 2. Le Framework Pythae
Le framework Pythae est utilisé pour explorer des variantes de l'espace latent.

| Modèle | Relation avec Baseline | Particularité | État / Note Technique |
| :--- | :--- | :--- | :--- |
| **Pythae VAE** | Comparaison directe | Architecture similaire à AEKL | Implémenté via `train_pythae_vae.py` |
| **Pythae VQ-VAE** | Extension | Quantification du latent (Codebook) | Implémenté via `train_pythae_vqvae.py` |
| **Pythae RHVAE** | Évolution géométrique | Contraintes riemanniennes (Hamiltonian VAE) | **Problème d'échelle** : Difficultés à passer sur des patches $128^3$ |

> **Note sur le RHVAE** : Le RHVAE permet d'introduire des contraintes géométriques dans l'espace latent (latent vectoriel), mais sa consommation mémoire et sa stabilité (instabilité en FP16/AMP) limitent son usage sur des volumes haute résolution.

### 3. Modèles Pré-entraînés & Fine-tuning
Pour pallier le manque de données ou accélérer la convergence, des modèles pré-entraînés sur de larges cohortes sont utilisés.

- **MedVAE** : Modèle spécialisé en imagerie médicale.
    - *Usage* : Fine-tuning sur les données du challenge ou entraînement pour le désenchevêtrement (disentanglement).
    - *Fichiers* : `train_medvae_disentangle_v1.py`, `finetune_medvae.py`.
- **MAISIv2** : Modèle de pointe pré-entraîné.
    - *Usage* : Fine-tuning pour adapter la représentation aux spécificités du challenge.
    - *Fichier* : `finetune_medvae.py`.

---

## 🛠️ Synthèse Technique

| Modèle | Type de Latent | Taille Latent (Patch $128^3$) | Stabilité AMP | Objectif Principal |
| :--- | :--- | :--- | :--- | :--- |
| **AEKL** | Spatial (Tenseur) | $4 \times 16^3$ | ✅ Oui | Baseline Reconstruction |
| **Pythae VAE** | Spatial (Tenseur) | $8 \times 16^3$ | ✅ Oui | Comparaison Framework |
| **Pythae VQ-VAE** | Spatial (Quantisé) | $8 \times 16^3$ | ✅ Oui | Compression Discrète |
| **Pythae RHVAE** | Vectoriel ($\mathbb{R}^D$) | $32$ ou $256$ | ❌ Non | Géométrie Latente |
| **MedVAE** | Spatial / Disent. | Variable | ✅ Oui | Transfer Learning |
| **MAISIv2** | Spatial | Variable | ✅ Oui | SOTA Pre-trained |

## 🚀 Guide d'utilisation rapide

### Entraînement
Chaque modèle possède son propre script de lancement. Exemple pour le RHVAE :
```bash
python src/vae3d/train_pythae_rhvae.py --config configs/pythae_rhvae_multimodal.yaml --env local
```

### Validation & QC
Pour vérifier la qualité des reconstructions et la distribution des latents :
- `qc_vae_3d.py` : Contrôle qualité d'un modèle spécifique.
- `qc_all_vaes.py` : Comparaison globale des modèles entraînés.
- `visualize_latent_umap.py` : Analyse de la structure du latent via UMAP.

## ⚠️ Limitations Connues
- **RHVAE & Résolution** : Le passage à des patches de $128^3$ est problématique en raison de la taille de la métrique riemannienne stockée en GPU.
- **Précision Numérique** : Le RHVAE nécessite l'utilisation de `float32` (AMP désactivé) pour éviter les divergences lors des étapes de Leapfrog.
