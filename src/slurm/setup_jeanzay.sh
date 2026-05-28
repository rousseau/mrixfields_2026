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

# Cache HuggingFace dans $WORK pour éviter de saturer le quota HOME (3 GiB)
export HF_HOME="$WORK/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME"

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

# ─── 3b. Pré-téléchargement des poids MedVAE (cache HuggingFace dans $WORK) ──
echo ""
echo "[3b/4] Vérification/téléchargement des poids MedVAE..."
echo "       Cache : $HF_HOME  (quota WORK, pas HOME)"
CUDA_VISIBLE_DEVICES="" IDR_DEBUG=WARN python3 - <<'PY'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# S'assurer que le cache HF pointe vers $WORK, pas $HOME
hf_home = os.environ.get("HF_HOME", "")
if not hf_home or ".cache/huggingface" not in hf_home.replace("\\", "/"):
    print("  [WARN] HF_HOME ne pointe pas vers $WORK — vérifiez setup_jeanzay.sh")

try:
    from medvae import MVAE

    def _check_or_download(model_name):
        hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))
        # Chercher un fichier .ckpt déjà présent dans le cache
        import glob
        pattern = os.path.join(hub_cache, "**", "*.ckpt")
        ckpts = glob.glob(pattern, recursive=True)
        already = [c for c in ckpts if model_name.replace("_", "") in c.replace("_", "").replace("-", "")]
        if already:
            print(f"  [OK] {model_name} déjà en cache : {already[0]}")
            return
        print(f"  Téléchargement {model_name} ...")
        MVAE(model_name=model_name, modality="mri")
        print(f"  [OK] {model_name} téléchargé.")

    _check_or_download("medvae_4_1_3d")
    _check_or_download("medvae_8_1_3d")
    print("  [OK] Poids MedVAE prêts.")
except Exception as e:
    print(f"  [WARN] Impossible de vérifier/télécharger les poids MedVAE : {e}")
    print("         Relancez manuellement depuis le nœud de login :")
    print("         CUDA_VISIBLE_DEVICES='' HF_HOME=$WORK/.cache/huggingface \\")
    print("           python3 -c \"from medvae import MVAE; MVAE(model_name='medvae_4_1_3d', modality='mri')\"")
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
