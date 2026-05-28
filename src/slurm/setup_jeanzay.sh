#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Setup Jean Zay — première installation (nœud de login)
#
# À exécuter UNE SEULE FOIS depuis le nœud de login Jean Zay :
#   bash src/slurm/setup_jeanzay.sh
#
# Ce script :
#   1. Crée la structure de répertoires sous $WORK/MRIX/
#   2. Installe les dépendances Python dans $WORK/.local
#   3. Affiche les commandes sbatch pour les entraînements
#
# ──── Passage SSH ─────────────────────────────────────────────
# Jean Zay est accessible via la passerelle SSH de Telecom Bretagne.
# Pour les transferts (identifiants hors du repo public) :
#
#   bash scripts/jz_sync.sh upload    # local → Jean Zay
#   bash scripts/jz_sync.sh download  # Jean Zay → local
#
# Pour configurer scripts/jz_sync.sh : voir scripts/jz_sync.template
#
# ──── Compte SLURM (SBATCH_ACCOUNT) ──────────────────────────
# Les scripts SLURM ne hardcodent PAS le compte projet.
# Définir la variable avant tout sbatch :
#
#   export SBATCH_ACCOUNT=<projet>@h100
#
# Pour le rendre permanent, ajouter dans ~/.bashrc sur Jean Zay :
#   echo 'export SBATCH_ACCOUNT=<projet>@h100' >> ~/.bashrc
#
# ──── Commandes d'entraînement sur Jean Zay ───────────────────
#
# Étape 1 -- StarGAN 2D (baseline MICCAI)
#
#   sbatch src/slurm/stargan_jeanzay.slurm T1W    retro_scratch
#   sbatch src/slurm/stargan_jeanzay.slurm T2W    retro_scratch
#   sbatch src/slurm/stargan_jeanzay.slurm T2FLAIR retro_scratch
#
# Étape 2 -- VAE 3D (espace latent)
#
#   # AEKL 3D (AutoencoderKL MONAI)
#   sbatch src/slurm/train_vae_jeanzay.slurm aekl vae3d_T1W
#
#   # VQ-VAE 3D (NeuroQuantHybrid, paired+unpaired)
#   sbatch src/slurm/train_vqvae_jeanzay.slurm vqvae3d_T1W "T1W T2W T2FLAIR" "0.1T 1.5T 3T 5T 7T"
#
#   # MedVAE fine-tuné (reprend depuis le checkpoint local s'il existe)
#   sbatch src/slurm/train_vae_jeanzay.slurm medvae medvae_T1W
#
# Étape 3 -- CFM 3D latent (une fois un VAE entraîné)
#
#   # Avec AEKL (4 canaux latents, UNet 3D 128ch)
#   sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_aekl.yaml
#
#   # Avec VQ-VAE (64 canaux latents, UNet 3D 64ch)
#   sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_vqvae.yaml
#
#   # Avec MedVAE frozen (4 canaux HuggingFace) ou fine-tuné (checkpoint local)
#   sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_medvae.yaml
#
# ────────────────────────────────────────────────────────────
#
set -e

MRIX_ROOT="$WORK/MRIX"
PROJECT_DIR="$MRIX_ROOT/mrixfields_2026"
DATA_DIR="$MRIX_ROOT/data"
CHALLENGE_DIR="$MRIX_ROOT/challenge"
# Le package mrixfields est dans le sous-répertoire Baseline/
CHALLENGE_BASELINE="$CHALLENGE_DIR/Baseline"

echo "============================================================"
echo " Setup Jean Zay - MRIxFields 2026"
echo " MRIX_ROOT : $MRIX_ROOT"
echo "============================================================"

# ─── 1. Répertoires ──────────────────────────────────────────────
mkdir -p "$DATA_DIR"
mkdir -p "$PROJECT_DIR/outputs/stargan2d/runs"
mkdir -p "$PROJECT_DIR/outputs/cfm3d/runs"
mkdir -p "$PROJECT_DIR/logs"
echo "[1/4] Répertoires créés"

# ─── 2. Cloner le repo du challenge (pour le package mrixfields) ──
if [ ! -d "$CHALLENGE_DIR" ]; then
    echo ""
    echo "[2/4] Clonage du code officiel du challenge..."
    echo "      [ACTION REQUISE] Lancez manuellement :"
    echo "      git clone <URL_DU_CHALLENGE> $CHALLENGE_DIR"
elif [ ! -f "$CHALLENGE_BASELINE/setup.py" ]; then
    echo "[2/4] Challenge repo présent mais Baseline/ introuvable : $CHALLENGE_BASELINE"
    echo "      Vérifiez la structure du repo."
else
    echo "[2/4] Challenge repo OK : $CHALLENGE_BASELINE"
fi

# ─── 3. Installer les dépendances Python ─────────────────────────
echo ""
echo "[3/4] Installation des dépendances Python.."
echo "      (nibabel, scipy, einops, torchcfm, pot, torchdiffeq, monai, pythae, medvae)"

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"

pip install --user --no-cache-dir nibabel scipy einops
pip install --user --no-cache-dir torchcfm pot torchdiffeq
pip install --user --no-cache-dir "monai[all]>=1.3.0"
pip install --user --no-cache-dir pythae

# MedVAE (StanfordMIMI — nécessite HuggingFace hub)
pip install --user --no-cache-dir "huggingface_hub>=0.20" medvae

# Package mrixfields
if [ -f "$CHALLENGE_BASELINE/setup.py" ]; then
    # --no-deps : evite tensorflow<2.16 incompatible avec Python 3.12
    # On n'utilise que mrixfields.data.transforms (aucune dépendance TF)
    pip install --user --no-cache-dir --no-deps "$CHALLENGE_BASELINE"
    echo "  -> mrixfields installé (sans dépendances) depuis $CHALLENGE_BASELINE"
elif [ -d "$CHALLENGE_DIR" ]; then
    echo "  [ATTENTION] setup.py introuvable dans $CHALLENGE_BASELINE"
    echo "  Vérifiez que le repo challenge est bien structuré avec un Baseline/."
else
    echo "  [ATTENTION] mrixfields non installé (challenge repo manquant)"
    echo "  Relancez ce script après avoir cloné le challenge."
fi

echo "[3/4] Dépendances installées dans \$WORK/.local"

# ─── 3b. Pré-téléchargement des poids MedVAE (cache HuggingFace) ─────────────
echo ""
echo "[3b/4] Pré-téléchargement des poids MedVAE en cache HuggingFace..."
echo "       (nécessaire car les nœuds de calcul peuvent ne pas avoir accès à HF)"
CUDA_VISIBLE_DEVICES="" IDR_DEBUG=WARN python3 - <<'PY'
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
try:
    from medvae import MVAE
    print("  Téléchargement medvae_4_1_3d ...")
    MVAE(model_name="medvae_4_1_3d", modality="mri")
    print("  Téléchargement medvae_8_1_3d ...")
    MVAE(model_name="medvae_8_1_3d", modality="mri")
    print("  [OK] Poids MedVAE en cache.")
except Exception as e:
    print(f"  [WARN] Impossible de pré-télécharger les poids MedVAE : {e}")
    print("         Relancez manuellement depuis le nœud de login :")
    print("         CUDA_VISIBLE_DEVICES='' python3 -c \"from medvae import MVAE; MVAE(model_name='medvae_4_1_3d', modality='mri')\"")
PY

# ─── 4. Commandes d'entraînement ──────────────────────────────────
echo ""
echo "============================================================"
echo " Commandes d'entraînement sur Jean Zay :"
echo "============================================================"
echo ""
echo "  # ----------------------------------------"
echo "  #  Etape 1 - StarGAN v2 (baseline MICCAI)  "
echo "  # ----------------------------------------"
echo "  sbatch src/slurm/stargan_jeanzay.slurm T1W    retro_scratch"
echo "  sbatch src/slurm/stargan_jeanzay.slurm T2W    retro_scratch"
echo "  sbatch src/slurm/stargan_jeanzay.slurm T2FLAIR retro_scratch"
echo ""
echo "  # ----------------------------------------"
echo "  #  Etape 2 - VAE 3D (espace latent)          "
echo "  # ----------------------------------------"
echo "  # AEKL 3D (AutoencoderKL MONAI, 4 canaux latents)"
echo "  sbatch src/slurm/train_vae_jeanzay.slurm aekl vae3d_T1W"
echo ""
echo "  # VQ-VAE 3D (NeuroQuantHybrid, paired+unpaired, 1 GPU)"
echo "  sbatch src/slurm/train_vqvae_jeanzay.slurm vqvae3d_T1W \"T1W T2W T2FLAIR\" \"0.1T 1.5T 3T 5T 7T\""
echo ""
echo "  # MedVAE fine-tuné (reprend depuis le checkpoint local s'il existe)"
echo "  sbatch src/slurm/train_vae_jeanzay.slurm medvae medvae_T1W"
echo ""
echo "  # ----------------------------------------"
echo "  #  Etape 3 - CFM 3D latent                   "
echo "  #   (un VAE finalisé requis avant lancement)   "
echo "  # ----------------------------------------"
echo "  # Avec AEKL (4 canaux latents, UNet 3D 128ch)"
echo "  sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_aekl.yaml"
echo ""
echo "  # Avec VQ-VAE (64 canaux latents, UNet 3D 64ch)"
echo "  sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_vqvae.yaml"
echo ""
echo "  # Avec MedVAE frozen (4 canaux HuggingFace) ou fine-tuné (checkpoint local)"
echo "  sbatch src/slurm/cfm_3d_jeanzay.slurm cfm T1W configs/cfm3d_T1W_medvae.yaml"
echo ""
echo "  # --- Transferts local <-> Jean Zay ---"
echo "  # Voir scripts/jz_sync.sh (hors repo public, contient les identifiants)"
echo "  bash scripts/jz_sync.sh help"
echo ""
echo "============================================================"
echo " Setup terminé."
echo "============================================================"
