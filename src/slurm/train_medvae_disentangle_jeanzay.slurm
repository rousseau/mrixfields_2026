#!/bin/bash
# SLURM - MedVAE disentanglement v1.2 (anatomie/modalite) sur Jean Zay
# v1.2: progressive paired-loss warmup + tighter adversarial schedule

#SBATCH --job-name=medvae_dis_v12
#SBATCH -A crp@h100
#SBATCH -C h100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu_p6
#SBATCH --qos=qos_gpu_h100-t4
#SBATCH --time=48:00:00
#SBATCH --output=logs/medvae_dis_v12_%x_%j.out
#SBATCH --error=logs/medvae_dis_v12_%x_%j.err

set -e

RUN_NAME="${1:-medvae_disentangle_v12}"
MODALITIES="${2:-T1W T2W T2FLAIR}"
FIELDS="${3:-0.1T 1.5T 3T 5T 7T}"

CODE_DIR="$WORK/MRIX/mrixfields_2026"
DATA_ROOT="$WORK/MRIX/data"
OUT_DIR="$WORK/MRIX/mrixfields_2026/outputs/medvae_disentangle_v1/runs/${RUN_NAME}"

mkdir -p "$CODE_DIR/logs"

module purge
module load arch/h100
module load pytorch-gpu/py3/2.5.0

export PYTHONUSERBASE="$WORK/.local"
export PATH="$WORK/.local/bin:$PATH"
export HF_HOME="$WORK/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1

# Les nœuds de calcul Jean Zay n'ont PAS accès à internet.
# Toutes les dépendances doivent être pré-installées : bash src/slurm/setup_jeanzay.sh
_check_import() {
    python3 -c "import $1" 2>/dev/null && return 0
    echo "[ERREUR] Package '$1' introuvable. Pré-installez : bash src/slurm/setup_jeanzay.sh"
    exit 1
}
_check_import medvae
_check_import nibabel
_check_import einops
_check_import scipy

cd "$CODE_DIR"

python3 src/vae3d/train_medvae_disentangle_v1.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT_DIR" \
  --splits retro_train \
  --modalities $MODALITIES \
  --fields $FIELDS \
  --batch-size 1 \
  --num-workers 16 \
  --steps 20000 \
  --save-every 200 \
  --print-every 20 \
  --use-amp \
  --amp-dtype bf16
