#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Entraînement VQ-VAE 3D multimodal sur Jean Zay (H100 SXM5 80GB)
#
# Partition gpu_p6 : H100 SXM5 80GB — 4 GPUs (DDP via torchrun)
# QOS : qos_gpu_h100-t4 (100h max)
#
# Lancement :
#   # Entraînement complet (toutes modalités + champs)
#   sbatch src/slurm/train_vqvae_jeanzay.slurm
#
#   # Override du run name / modalités
#   sbatch src/slurm/train_vqvae_jeanzay.slurm vqvae3d_t1w_only "T1W" "0.1T 1.5T 3T"
#
#   # Override du nombre de GPUs (défaut 4)
#   sbatch src/slurm/train_vqvae_jeanzay.slurm vqvae3d_full "T1W T2W T2FLAIR" "0.1T 1.5T 3T 5T 7T" 0 4
#
# Arguments positionnels :
#   $1 : RUN_NAME  (défaut : vqvae3d_hybrid)
#   $2 : MODALITIES espace-séparées entre guillemets  (défaut : "T1W T2W T2FLAIR")
#   $3 : FIELDS     espace-séparées entre guillemets  (défaut : "0.1T 1.5T 3T 5T 7T")
#   $4 : FORCE_RESTART (0|1, défaut 0)
#   $5 : NPROC_PER_NODE (défaut 4)
#
# Prérequis (via setup_jeanzay.sh) :
#   $WORK/MRIX/mrixfields_2026/   ← ce repo
#   $WORK/MRIX/data/              ← données NIfTI (format R_{MOD}_{FIELD}_{ID}.nii.gz)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=vqvae3d
# SBATCH_ACCOUNT doit être défini dans l'environnement avant sbatch :
#   export SBATCH_ACCOUNT=<projet>@h100   (ex: crp@h100)
# SLURM lit automatiquement cette variable — ne pas hardcoder ici.
#SBATCH -A crp@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64       # 16 CPUs par GPU (4 GPUs)

#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=100:00:00

#SBATCH --output=logs/vqvae3d_%x_%j.out
#SBATCH --error=logs/vqvae3d_%x_%j.err

# Signal 120s avant la fin pour auto-requeue
#SBATCH --signal=USR1@120

set -e

RUN_NAME="${1:-vqvae3d_hybrid}"
MODALITIES="${2:-T1W T2W T2FLAIR}"
FIELDS="${3:-0.1T 1.5T 3T 5T 7T}"
# FORCE_RESTART=1 : ignore les checkpoints existants et repart de zéro.
# Usage : sbatch train_vqvae_jeanzay.slurm vqvae_full "T1W T2W T2FLAIR" "0.1T 1.5T 3T 5T 7T" 1
FORCE_RESTART="${4:-0}"
NPROC_PER_NODE="${5:-4}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-1}"
USE_AMP="${USE_AMP:-0}"

CODE_DIR="$WORK/MRIX/mrixfields_2026"
DATA_ROOT="$WORK/MRIX/data"
OUTPUT_DIR="$WORK/MRIX/mrixfields_2026/outputs/vqvae3d/runs/${RUN_NAME}"

mkdir -p "$CODE_DIR/logs"

echo "======================================================================="
echo " Job ID    : $SLURM_JOB_ID"
echo " Nœud      : $SLURM_JOB_NODELIST"
echo " Run name  : $RUN_NAME"
echo " Modalités : $MODALITIES"
echo " Champs    : $FIELDS"
echo " Data      : $DATA_ROOT"
echo " Output    : $OUTPUT_DIR"
echo " NPROC     : $NPROC_PER_NODE"
echo " Use AMP   : $USE_AMP"
echo " AMP dtype : $AMP_DTYPE"
echo " Grad ckpt : $GRAD_CHECKPOINTING"
echo "======================================================================="

# ─── Modules Jean Zay ────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ─── Dépendances utilisateur ─────────────────────────────────────────────────
export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / NPROC_PER_NODE))

# Installation via pip --user --no-cache-dir (fonctionne depuis les nœuds de
# calcul Jean Zay selon la doc IDRIS). En cas d'échec réseau persistant,
# pré-installer depuis le nœud de login : bash src/slurm/setup_jeanzay.sh
_pip_install() {
    python3 -c "import $1" 2>/dev/null && return 0
    echo "[INFO] Installation de '$2'..."
    pip install --user --no-cache-dir "$2" || {
        echo "[ERREUR] Impossible d'installer '$2'."
        echo "  Pré-installez depuis le nœud de login : bash src/slurm/setup_jeanzay.sh"
        exit 1
    }
}
_pip_install nibabel nibabel
_pip_install einops  einops
_pip_install scipy   scipy

# ─── Cache PyTorch ───────────────────────────────────────────────────────────
export TORCH_HOME="$WORK/.cache/torch"

# ─── Vérification des données NIfTI ─────────────────────────────────────────
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERREUR] DATA_ROOT introuvable : $DATA_ROOT"
    exit 1
fi
N_NIFTI=$(find "$DATA_ROOT" -name "R_*.nii.gz" | wc -l)
echo "Données NIfTI  : OK ($N_NIFTI volumes dans $DATA_ROOT)"
echo "GPU(s)         :"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# ─── Gestion auto-requeue ─────────────────────────────────────────────────────
_requeue_handler() {
    echo "[$(date)] Timeout imminent — requeue du job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap '_requeue_handler' USR1

# ─── Détection du dernier checkpoint pour reprise ─────────────────────────────────
WEIGHTS_DIR="${OUTPUT_DIR}/weights"
RESUME_FLAG=""

if [ "$FORCE_RESTART" = "1" ]; then
    echo "FORCE_RESTART=1 : démarrage à zéro (checkpoints ignorés)"
elif [ -d "$WEIGHTS_DIR" ]; then
    # Priorité 1 : dernier checkpoint périodique vqvae_step_*.pth (le plus récent)
    LAST_STEP_CKPT=$(ls -t "${WEIGHTS_DIR}"/vqvae_step_*.pth 2>/dev/null | head -1)
    # Priorité 2 : model_final.pth
    # Priorité 3 : model_best.pth
    if [ -n "$LAST_STEP_CKPT" ]; then
        LAST_CKPT="$LAST_STEP_CKPT"
    elif [ -f "${WEIGHTS_DIR}/model_final.pth" ]; then
        LAST_CKPT="${WEIGHTS_DIR}/model_final.pth"
    elif [ -f "${WEIGHTS_DIR}/model_best.pth" ]; then
        LAST_CKPT="${WEIGHTS_DIR}/model_best.pth"
    else
        LAST_CKPT=""
    fi

    if [ -n "$LAST_CKPT" ]; then
        RESUME_FLAG="--resume $LAST_CKPT"
        echo "Reprise depuis  : $LAST_CKPT"
    else
        echo "Aucun checkpoint trouvé — démarrage à zéro."
    fi
else
    echo "Aucun répertoire weights — démarrage à zéro."
fi

# ─── Lancement DDP (4xH100 par défaut) ───────────────────────────────────────
cd "$CODE_DIR"

echo "======================================================================="
echo " Démarrage torchrun ($NPROC_PER_NODE GPU(s))"
echo "======================================================================="

GC_FLAG=""
if [ "$GRAD_CHECKPOINTING" = "1" ]; then
    GC_FLAG="--gradient-checkpointing"
fi

AMP_FLAG=""
if [ "$USE_AMP" = "1" ]; then
    AMP_FLAG="--use-amp --amp-dtype $AMP_DTYPE"
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" src/vae3d/train_vqvae.py \
    --data-root  "$DATA_ROOT" \
    --splits     retro_train \
    --modalities $MODALITIES \
    --fields     $FIELDS \
    --batch-size 1 \
    --num-workers 16 \
    --steps      20000 \
    --channel-multipliers 1 2 4 4 \
    $GC_FLAG \
    $AMP_FLAG \
    --output-dir "$OUTPUT_DIR" \
    $RESUME_FLAG

echo "======================================================================="
echo " Job terminé (ou interrompu pour requeue)"
echo "======================================================================="
