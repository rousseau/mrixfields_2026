#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Script SLURM — Entraînement VQ-VAE 3D multimodal sur Jean Zay (H100 SXM5 80GB)
#
# Partition gpu_p6 : H100 SXM5 80GB, 4 GPUs/nœud
# QOS : qos_gpu_h100-t4 (100h max)
#
# Lancement :
#   # Entraînement complet (toutes modalités + champs)
#   sbatch src/slurm/train_vqvae_jeanzay.slurm
#
#   # Override du run name / modalités
#   sbatch src/slurm/train_vqvae_jeanzay.slurm vqvae3d_t1w_only "T1W" "0.1T 1.5T 3T"
#
# Arguments positionnels :
#   $1 : RUN_NAME  (défaut : vqvae3d_hybrid)
#   $2 : MODALITIES espace-séparées entre guillemets  (défaut : "T1W T2W T2FLAIR")
#   $3 : FIELDS     espace-séparées entre guillemets  (défaut : "0.1T 1.5T 3T 5T 7T")
#
# Prérequis (via setup_jeanzay.sh) :
#   $WORK/MRIX/mrixfields_2026/   ← ce repo
#   $WORK/MRIX/data/              ← données NIfTI (format R_{MOD}_{FIELD}_{ID}.nii.gz)
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=vqvae3d
#SBATCH -A crp@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1               # torchrun gère les 4 sous-processus GPU
#SBATCH --cpus-per-task=64       # 16 CPUs × 4 GPUs (workers DataLoader)

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

CODE_DIR="$WORK/MRIX/mrixfields_2026"
DATA_ROOT="$WORK/MRIX/data"
OUTPUT_DIR="$WORK/MRIX/outputs/vqvae3d/${RUN_NAME}"  # Simplified output path structure

mkdir -p "$CODE_DIR/logs"

echo "======================================================================="
echo " Job ID    : $SLURM_JOB_ID"
echo " Nœud      : $SLURM_JOB_NODELIST"
echo " Run name  : $RUN_NAME"
echo " Modalités : $MODALITIES"
echo " Champs    : $FIELDS"
echo " Data      : $DATA_ROOT"
echo " Output    : $OUTPUT_DIR"
echo "======================================================================="

# ─── Modules Jean Zay ────────────────────────────────────────────────────────
module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

# ─── Dépendances utilisateur ─────────────────────────────────────────────────
export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

_install_if_missing() {
    python3 -c "import $1" 2>/dev/null || pip install --user --quiet "$2"
}
_install_if_missing nibabel nibabel
_install_if_missing einops  einops

# ─── Cache PyTorch ───────────────────────────────────────────────────────────
export TORCH_HOME="$WORK/.cache/torch"

# ─── Vérification des données NIfTI ─────────────────────────────────────────
if [ ! -d "$DATA_ROOT" ]; then
    echo "[ERREUR] DATA_ROOT introuvable : $DATA_ROOT"
    exit 1
fi
N_NIFTI=$(find "$DATA_ROOT" -name "R_*.nii.gz" | wc -l)
echo "Données NIfTI  : OK ($N_NIFTI volumes dans $DATA_ROOT)"
echo "GPU            : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ─── Gestion auto-requeue ─────────────────────────────────────────────────────
_requeue_handler() {
    echo "[$(date)] Timeout imminent — requeue du job ${SLURM_JOB_ID}"
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
}
trap '_requeue_handler' USR1

# ─── Détection du dernier checkpoint pour reprise ────────────────────────────
WEIGHTS_DIR="${OUTPUT_DIR}/weights"
RESUME_FLAG=""

if [ -d "$WEIGHTS_DIR" ]; then
    LAST_CKPT=$(ls -t "${WEIGHTS_DIR}"/checkpoint_*.pth 2>/dev/null | head -1 || true)
    if [ -n "$LAST_CKPT" ]; then
        RESUME_FLAG="--resume $LAST_CKPT"
        echo "Reprise depuis  : $LAST_CKPT"
    else
        echo "Aucun checkpoint trouvé — démarrage à zéro."
    fi
else
    echo "Aucun répertoire weights — démarrage à zéro."
fi

# ─── Lancement multi-GPU (torchrun DDP 4×H100) ───────────────────────────────
cd "$CODE_DIR"

echo "======================================================================="
echo " Démarrage torchrun (4 GPUs H100)"
echo "======================================================================="

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    src/train_vqvae.py \
    --data-root  "$DATA_ROOT" \
    --splits     retro_train \
    --modalities $MODALITIES \
    --fields     $FIELDS \
    --batch-size 1 \
    --num-workers 16 \
    --steps      20000 \
    --channel-multipliers 1 2 4 4 \
    --gradient-checkpointing \
    --use-amp \
    --output-dir "$OUTPUT_DIR" \
    $RESUME_FLAG

echo "======================================================================="
echo " Job terminé (ou interrompu pour requeue)"
echo "======================================================================="
