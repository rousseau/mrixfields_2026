#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# SLURM — Fine-tuning MedVAE 3D sur Jean Zay (1×H100 SXM5 80GB)
#
# Fine-tune medvae_4_1_3d sur Training_retrospective complet
# (T1W + T2W + T2FLAIR × 0.1T + 1.5T + 3T + 5T + 7T, ~1939 volumes).
#
# Lancement :
#   sbatch src/slurm/finetune_medvae_jeanzay.slurm
#
#   # Override du nombre de steps ou du LR :
#   STEPS=30000 LR=5e-6 sbatch src/slurm/finetune_medvae_jeanzay.slurm
#
# Prérequis sur le nœud de login (une seule fois) :
#   module purge && module load arch/h100 && module load pytorch-gpu/py3/2.5.0
#   export PYTHONUSERBASE=$WORK/.local && export PATH=$WORK/.local/bin:$PATH
#   pip install --user medvae
#   # Pré-télécharger les poids (réseau disponible sur login node, pas sur compute)
#   python3 -c "from medvae import MVAE; MVAE(model_name='medvae_4_1_3d', modality='mri')"
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=medvae_ft
# SBATCH_ACCOUNT doit être défini avant sbatch :
#   export SBATCH_ACCOUNT=<projet>@h100   (ex: crp@h100)
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=48:00:00
#SBATCH --output=logs/medvae_ft_%x_%j.out
#SBATCH --error=logs/medvae_ft_%x_%j.err
# Signal USR1 envoyé 120s avant la fin → auto-requeue propre
#SBATCH --signal=USR1@120

set -euo pipefail

# ─── Chemins ─────────────────────────────────────────────────────────────────
CODE_DIR="$WORK/MRIX/mrixfields_2026"
DATA_ROOT="${DATA_ROOT:-$WORK/MRIX/data}"
CONFIG="configs/medvae_finetune_all.yaml"

mkdir -p "$CODE_DIR/logs"

echo "======================================================================="
echo " Job ID    : ${SLURM_JOB_ID:-local}"
echo " Nœud      : ${SLURM_JOB_NODELIST:-localhost}"
echo " Config    : $CONFIG"
echo " Data root : $DATA_ROOT"
echo " Code dir  : $CODE_DIR"
echo "======================================================================="

# ─── Modules Jean Zay ────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ─── Dépendances utilisateur ─────────────────────────────────────────────────
export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"
export TORCH_HOME="$WORK/.cache/torch"
# Évite la fragmentation mémoire CUDA sur les longs trainings
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# HuggingFace offline : pas de réseau sur les nœuds compute
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"

_require() {
    python3 -c "import $1" 2>/dev/null || {
        echo "[ERREUR] Module Python manquant : $1"
        echo "Installer sur le nœud de login :"
        echo "  module purge && module load arch/h100 && module load pytorch-gpu/py3/2.5.0"
        echo "  export PYTHONUSERBASE=$WORK/.local && export PATH=$WORK/.local/bin:\$PATH"
        echo "  pip install --user $2"
        exit 1
    }
}

_install_if_missing() {
    python3 -c "import $1" 2>/dev/null || pip install --user --quiet "$2"
}

_require medvae medvae
_install_if_missing nibabel nibabel
_install_if_missing scipy  scipy

# ─── Vérification des poids MedVAE en cache ──────────────────────────────────
python3 - <<'EOF'
import os
os.environ["HF_HUB_OFFLINE"] = "1"
try:
    from medvae import MVAE
    MVAE(model_name="medvae_4_1_3d", modality="mri")
    print("[OK] Poids MedVAE disponibles dans le cache HuggingFace")
except Exception as e:
    print(f"[ERREUR] Poids MedVAE introuvables en cache : {e}")
    print("Sur le nœud de login, exécuter :")
    print("  python3 -c \"from medvae import MVAE; MVAE(model_name='medvae_4_1_3d', modality='mri')\"")
    exit(1)
EOF

# ─── Vérification des données ─────────────────────────────────────────────────
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERREUR] DATA_ROOT introuvable : $DATA_ROOT"
    exit 1
fi

N_VOLS=$(find "$DATA_ROOT/Training_retrospective" -name "*.nii.gz" 2>/dev/null | wc -l)
echo "Volumes Training_retrospective : $N_VOLS"
echo "GPU : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ─── Détection du dernier checkpoint pour auto-reprise ───────────────────────
WEIGHTS_DIR="$CODE_DIR/outputs/medvae/runs/medvae_finetune_all/weights"
RESUME_FLAG=""

if [ -d "$WEIGHTS_DIR" ]; then
    # Priorité : model_final.pth (requeue) > dernier medvae_step_XXXXXX.pth
    if [ -f "$WEIGHTS_DIR/model_final.pth" ]; then
        RESUME_FLAG="--resume $WEIGHTS_DIR/model_final.pth"
        echo "Reprise depuis   : model_final.pth"
    else
        LAST_STEP=$(ls "$WEIGHTS_DIR"/medvae_step_*.pth 2>/dev/null \
                    | sort | tail -1)
        if [ -n "$LAST_STEP" ]; then
            RESUME_FLAG="--resume $LAST_STEP"
            echo "Reprise depuis   : $(basename "$LAST_STEP")"
        fi
    fi
fi

if [ -z "$RESUME_FLAG" ]; then
    echo "Aucun checkpoint — démarrage à zéro."
fi

# ─── Lancement (single GPU) ──────────────────────────────────────────────────
cd "$CODE_DIR"

echo "======================================================================="
echo " Démarrage fine-tuning MedVAE (1×H100)"
echo "======================================================================="

python3 src/vae3d/finetune_medvae.py \
    --config "$CONFIG" \
    --env    jeanzay \
    $RESUME_FLAG

echo "======================================================================="
echo " Job terminé (ou sauvegarde d'urgence + requeue)"
echo "======================================================================="
