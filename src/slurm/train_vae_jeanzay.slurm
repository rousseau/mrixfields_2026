#!/bin/bash
# Unified SLURM launcher for VAE training on Jean Zay.
#
# Usage:
#   sbatch src/slurm/train_vae_jeanzay.slurm aekl
#   sbatch src/slurm/train_vae_jeanzay.slurm vqvae my_vq_run "T1W T2W T2FLAIR" "0.1T 1.5T 3T 5T 7T"
#   sbatch src/slurm/train_vae_jeanzay.slurm medvae my_med_run
#
# Positional args:
#   $1: VAE_TYPE   -> aekl | vqvae | medvae
#   $2: RUN_NAME   -> optional (default: <vae>_jeanzay)
#   $3: MODALITIES -> optional, quoted list, only for vqvae
#   $4: FIELDS     -> optional, quoted list, only for vqvae
#
# Common tunables via env at submit time:
#   --export=ALL,STEPS=20000,BATCH_SIZE=2,LR=1e-4,TIME_LIMIT=48:00:00

#SBATCH --job-name=vae_train
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=48:00:00
#SBATCH --output=logs/vae_%x_%j.out
#SBATCH --error=logs/vae_%x_%j.err

set -euo pipefail

VAE_TYPE="${1:-}"
RUN_NAME="${2:-${VAE_TYPE}_jeanzay}"
MODALITIES="${3:-T1W T2W T2FLAIR}"
FIELDS="${4:-0.1T 1.5T 3T 5T 7T}"

if [[ -z "$VAE_TYPE" ]]; then
  echo "[ERREUR] VAE_TYPE manquant. Valeurs: aekl | vqvae | medvae"
  exit 1
fi

case "$VAE_TYPE" in
  aekl|vqvae|medvae) ;;
  *)
    echo "[ERREUR] VAE_TYPE invalide: $VAE_TYPE"
    echo "Valeurs acceptées: aekl | vqvae | medvae"
    exit 1
    ;;
esac

CODE_DIR="$WORK/MRIX/mrixfields_2026"
DATA_ROOT="${DATA_ROOT:-$WORK/MRIX/data}"
OUT_ROOT="${OUT_ROOT:-$WORK/MRIX/outputs}"

STEPS="${STEPS:-20000}"
LR="${LR:-1e-4}"

DEFAULT_BATCH_SIZE=2
if [[ "$VAE_TYPE" == "vqvae" ]]; then
  DEFAULT_BATCH_SIZE=1
fi
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"

mkdir -p "$CODE_DIR/logs"
mkdir -p "$OUT_ROOT"

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"
export TORCH_HOME="$WORK/.cache/torch"
# Avoid CUDA memory fragmentation on long-running training
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

_install_if_missing() {
  python3 -c "import $1" >/dev/null 2>&1 || pip install --user --quiet "$2"
}

_require_module() {
  python3 -c "import $1" >/dev/null 2>&1 || {
    echo "[ERREUR] Module Python manquant: $1"
    echo "Installez-le sur login node puis relancez:"
    echo "  module purge && module load arch/h100 && module load pytorch-gpu/py3/2.5.0"
    echo "  export PYTHONUSERBASE=$WORK/.local && export PATH=$WORK/.local/bin:$PATH"
    echo "  pip install --user $1"
    exit 1
  }
}

_install_if_missing nibabel nibabel
_install_if_missing scipy scipy
_install_if_missing einops einops

if [[ "$VAE_TYPE" == "medvae" ]]; then
  # Compute nodes may not have external network access; require preinstalled package.
  _require_module medvae
fi

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[ERREUR] DATA_ROOT introuvable: $DATA_ROOT"
  exit 1
fi

cd "$CODE_DIR"

echo "======================================================================="
echo " Job ID    : ${SLURM_JOB_ID:-N/A}"
echo " VAE type  : $VAE_TYPE"
echo " Run name  : $RUN_NAME"
echo " Data root : $DATA_ROOT"
echo " Output    : $OUT_ROOT"
echo " Steps     : $STEPS"
echo " Batch     : $BATCH_SIZE"
echo " LR        : $LR"
echo "======================================================================="

if [[ "$VAE_TYPE" == "vqvae" ]]; then
  OUTPUT_DIR="$OUT_ROOT/vqvae3d/runs/$RUN_NAME"
  python3 src/train_vqvae.py \
    --data-root "$DATA_ROOT" \
    --splits retro_train \
    --modalities $MODALITIES \
    --fields $FIELDS \
    --batch-size "$BATCH_SIZE" \
    --num-workers 8 \
    --steps "$STEPS" \
    --lr "$LR" \
    --use-amp \
    --output-dir "$OUTPUT_DIR"
else
  OUTPUT_DIR="$OUT_ROOT/train_${VAE_TYPE}/$RUN_NAME"
  python3 src/train_vae.py \
    --vae "$VAE_TYPE" \
    --data-root "$DATA_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --lr "$LR" \
    --device cuda \
    --output-dir "$OUTPUT_DIR" \
    --use-amp
fi

echo "======================================================================="
echo " Training terminé pour $VAE_TYPE"
echo "======================================================================="
